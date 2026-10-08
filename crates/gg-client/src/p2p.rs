//! Peer-assisted artifact distribution (ADR-0012), worker side.
//!
//! Two halves, both subordinate to the control plane:
//!
//! * [`PeerServer`] serves chunks of artifacts already in the local cache, and only to a reader
//!   presenting a ticket signed by the server for that exact artifact plus a request signed with
//!   the key named in that ticket. Sharing is off by default, bounded by a connection limit and a
//!   byte budget, and never exposes anything but cached platform artifacts.
//! * [`fetch_with_peers`] downloads an artifact chunk by chunk. Every chunk is verified against the
//!   server-signed manifest **before** it is written; a peer that returns wrong bytes is dropped
//!   for the rest of the transfer; a chunk no peer serves is fetched from the control plane, which
//!   remains the durable source. The whole-file hash is checked again at the end.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use gg_core::{NetworkId, NodeId, Sha256Digest, UnixMillis, WorkerId};
use gg_protocol::domain::Domain;
use gg_protocol::keys::{PublicKey, SecretKey};
use gg_protocol::messages::{
    ArtifactRef, ChunkManifestV1, ChunkRequestV1, PeerTicketV1, PushTicketV1,
};
use gg_protocol::signed::Signed;
use serde::Deserialize;

use crate::cache::ArtifactCache;
use crate::client::{Api, ClientError};

/// Header carrying the hex-encoded canonical `Signed<PeerTicketV1>`.
pub const TICKET_HEADER: &str = "x-gg-ticket";
/// Header carrying the hex-encoded canonical `Signed<ChunkRequestV1>`.
pub const REQUEST_HEADER: &str = "x-gg-request";
/// A chunk request older than this (either direction) is refused by the holder.
pub const REQUEST_FRESHNESS_SECS: i64 = 120;

/// Counters of the sharing side (exposed for tests, logs and the demo report).
#[derive(Debug, Default)]
pub struct ShareStats {
    /// Chunks actually served.
    pub chunks_served: AtomicU64,
    /// Bytes actually served.
    pub bytes_served: AtomicU64,
    /// Requests refused because ticket or signature did not verify.
    pub refused_unauthorized: AtomicU64,
    /// Requests refused because the artifact is not held locally.
    pub refused_missing: AtomicU64,
    /// Requests refused by the local budget or connection limit.
    pub refused_budget: AtomicU64,
    /// Chunks being served right now, and when the last one finished (milliseconds since the
    /// server started): a durable copy sent in the background yields the uplink to them (ADR-0026).
    pub in_flight: AtomicU64,
    /// See `in_flight`.
    pub last_served_ms: AtomicU64,
    /// Origin of `last_served_ms`.
    pub started: std::sync::OnceLock<std::time::Instant>,
}

impl ShareStats {
    fn now_ms(&self) -> u64 {
        self.started
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_millis() as u64
    }

    /// Whether a peer is being served, or was within the last `quiet_ms` milliseconds.
    pub fn busy(&self, quiet_ms: u64) -> bool {
        let last = self.last_served_ms.load(Ordering::Relaxed);
        self.in_flight.load(Ordering::Relaxed) > 0 || (last > 0 && self.now_ms() < last + quiet_ms)
    }

    /// Snapshot as (served chunks, served bytes, unauthorized, missing, budget).
    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.chunks_served.load(Ordering::Relaxed),
            self.bytes_served.load(Ordering::Relaxed),
            self.refused_unauthorized.load(Ordering::Relaxed),
            self.refused_missing.load(Ordering::Relaxed),
            self.refused_budget.load(Ordering::Relaxed),
        )
    }
}

impl crate::client::YieldTo for ShareStats {
    fn busy(&self, quiet_ms: u64) -> bool {
        ShareStats::busy(self, quiet_ms)
    }
}

/// Artifacts being downloaded right now, served chunk by chunk as they arrive (a swarm): workers
/// that start together on the same input exchange the chunks each already has instead of all
/// asking the control plane. A chunk is listed only once it matched the signed manifest and was
/// written.
#[derive(Clone, Default)]
pub struct Partials(Arc<std::sync::Mutex<std::collections::HashMap<Sha256Digest, Partial>>>);

struct Partial {
    path: std::path::PathBuf,
    size: u64,
    chunk_size: u32,
    have: Vec<bool>,
}

impl std::fmt::Debug for Partials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Partials")
    }
}

impl Partials {
    fn begin(&self, hash: Sha256Digest, path: &Path, size: u64, chunk_size: u32, count: usize) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).insert(
            hash,
            Partial {
                path: path.to_path_buf(),
                size,
                chunk_size,
                have: vec![false; count],
            },
        );
    }

    fn mark(&self, hash: &Sha256Digest, index: usize) {
        if let Some(p) = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(hash)
            && let Some(h) = p.have.get_mut(index)
        {
            *h = true;
        }
    }

    /// Stop sharing a download (failed, or now held by the cache).
    pub fn end(&self, hash: &Sha256Digest) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(hash);
    }

    /// The download is complete at `path`: keep serving it from there until the caller has put it
    /// in the cache and calls [`Partials::end`]. Readers still waiting for chunks this one was
    /// responsible for get them without a gap.
    fn complete(&self, hash: &Sha256Digest, path: &Path) {
        if let Some(p) = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(hash)
        {
            p.path = path.to_path_buf();
            p.have.iter_mut().for_each(|h| *h = true);
        }
    }

    /// Chunk `index` of `hash`, if it is being downloaded, already verified and written.
    fn read(&self, hash: &Sha256Digest, size: u64, chunk_size: u32, index: u32) -> Option<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let path = {
            let g = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let p = g.get(hash)?;
            if p.size != size || p.chunk_size != chunk_size || !*p.have.get(index as usize)? {
                return None;
            }
            p.path.clone()
        };
        let start = index as u64 * chunk_size as u64;
        let len = (size - start).min(chunk_size as u64);
        let mut f = std::fs::File::open(path).ok()?;
        f.seek(SeekFrom::Start(start)).ok()?;
        let mut buf = vec![0u8; len as usize];
        f.read_exact(&mut buf).ok()?;
        Some(buf)
    }
}

/// Artifacts this reader waits for by push (ADR-0041, A8: the reversed connection), with their
/// signed manifest: a pushed chunk is taken only if it matches it, and handed to the download.
#[derive(Clone, Default)]
pub struct Inbox(Arc<std::sync::Mutex<std::collections::HashMap<Sha256Digest, InboxEntry>>>);

struct InboxEntry {
    size: u64,
    chunk_size: u32,
    digests: Vec<Sha256Digest>,
    tx: tokio::sync::mpsc::UnboundedSender<(u32, Vec<u8>)>,
}

impl Inbox {
    /// Wait for pushes of the artifact `manifest` describes.
    fn open(
        &self,
        manifest: &ChunkManifestV1,
    ) -> tokio::sync::mpsc::UnboundedReceiver<(u32, Vec<u8>)> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.0.lock().unwrap_or_else(|e| e.into_inner()).insert(
            manifest.artifact_hash,
            InboxEntry {
                size: manifest.size,
                chunk_size: manifest.chunk_size,
                digests: manifest.chunks.clone(),
                tx,
            },
        );
        rx
    }

    fn close(&self, hash: &Sha256Digest) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(hash);
    }

    /// A pushed chunk: taken only for an artifact waited for, of the size the ticket and the
    /// manifest agree on, and whose digest is the manifest's.
    fn deliver(
        &self,
        ticket: &PushTicketV1,
        index: u32,
        bytes: Vec<u8>,
    ) -> Result<(), (StatusCode, &'static str)> {
        let g = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(e) = g.get(&ticket.artifact_hash) else {
            return Err((StatusCode::CONFLICT, "not waiting for this artifact"));
        };
        if e.size != ticket.size || e.chunk_size != ticket.chunk_size {
            return Err((StatusCode::CONFLICT, "ticket and manifest disagree"));
        }
        let Some(expected) = e.digests.get(index as usize) else {
            return Err((StatusCode::UNPROCESSABLE_ENTITY, "chunk out of range"));
        };
        if Sha256Digest::of_bytes(&bytes) != *expected {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "chunk does not match the signed manifest",
            ));
        }
        e.tx.send((index, bytes))
            .map_err(|_| (StatusCode::CONFLICT, "the download is over"))
    }
}

/// Header carrying a hex-encoded, server-signed [`PushTicketV1`].
pub const PUSH_TICKET_HEADER: &str = "x-gg-push-ticket";

struct ShareState {
    cache: ArtifactCache,
    /// Development fault: serve wrong bytes (the reader must catch it). Refused in public mode.
    corrupt: bool,
    network: NetworkId,
    server_key: PublicKey,
    stats: Arc<ShareStats>,
    budget_left: AtomicU64,
    conns: Arc<tokio::sync::Semaphore>,
    /// Simulated uplink (development measurements only).
    uplink: Option<Arc<crate::throttle::Uplink>>,
    /// Downloads in progress, served as they arrive.
    partials: Partials,
    /// Artifacts waited for by push.
    inbox: Inbox,
    /// This server's own transport key fingerprint: a push ticket must name it.
    fingerprint: Sha256Digest,
}

/// A running chunk server. Connections are TLS 1.3 with a raw public key (ADR-0013); readers pin
/// the fingerprint the control plane published for this worker.
pub struct PeerServer {
    inbox: Inbox,
    partials: Partials,
    addr: SocketAddr,
    fingerprint: Sha256Digest,
    stats: Arc<ShareStats>,
    shutdown: tokio::sync::watch::Sender<bool>,
    handle: tokio::task::JoinHandle<()>,
}

impl PeerServer {
    /// Bind `listen_addr` and serve chunks of artifacts present in `cache`.
    pub async fn start(
        listen_addr: &str,
        cache: ArtifactCache,
        network: NetworkId,
        server_key: PublicKey,
        max_conns: u32,
        upload_budget_bytes: u64,
    ) -> anyhow::Result<Self> {
        Self::start_with_faults(
            listen_addr,
            cache,
            network,
            server_key,
            max_conns,
            upload_budget_bytes,
            false,
            None,
        )
        .await
    }

    /// As [`PeerServer::start`], with the local development fault that serves wrong bytes.
    #[allow(clippy::too_many_arguments)]
    pub async fn start_with_faults(
        listen_addr: &str,
        cache: ArtifactCache,
        network: NetworkId,
        server_key: PublicKey,
        max_conns: u32,
        upload_budget_bytes: u64,
        corrupt: bool,
        uplink: Option<Arc<crate::throttle::Uplink>>,
    ) -> anyhow::Result<Self> {
        if corrupt {
            tracing::warn!("DEV FAULT: this worker serves corrupted chunks to peers");
        }
        let identity = crate::peer_tls::TlsIdentity::generate()?;
        let fingerprint = identity.fingerprint();
        let state = Arc::new(ShareState {
            cache,
            corrupt,
            network,
            server_key,
            stats: Arc::new(ShareStats::default()),
            budget_left: AtomicU64::new(upload_budget_bytes),
            conns: Arc::new(tokio::sync::Semaphore::new(max_conns.max(1) as usize)),
            uplink,
            partials: Partials::default(),
            inbox: Inbox::default(),
            fingerprint,
        });
        let app = axum::Router::new()
            .route("/p2p/v1/artifacts/{hash}/chunks/{index}", get(serve_chunk))
            .route(
                "/p2p/v1/artifacts/{hash}/push/{index}",
                axum::routing::put(receive_push)
                    .layer(axum::extract::DefaultBodyLimit::max(17 << 20)),
            )
            .with_state(state.clone());
        let tls_config = identity.server_config()?;
        let listener = bind_listener(listen_addr).await?;
        let addr = listener.local_addr()?;
        let (tx, rx) = tokio::sync::watch::channel(false);
        let handle = tokio::spawn(crate::peer_tls::serve_tls(
            listener, app, tls_config, max_conns, rx,
        ));
        tracing::info!(%addr, fingerprint = %fingerprint, "sharing cached artifacts with authorized peers over TLS");
        Ok(Self {
            inbox: state.inbox.clone(),
            partials: state.partials.clone(),
            addr,
            fingerprint,
            stats: state.stats.clone(),
            shutdown: tx,
            handle,
        })
    }

    /// SHA-256 of this server's TLS transport key, announced to the control plane so readers can
    /// pin it.
    pub fn fingerprint(&self) -> Sha256Digest {
        self.fingerprint
    }

    /// Address actually bound.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Counters.
    pub fn stats(&self) -> Arc<ShareStats> {
        self.stats.clone()
    }

    /// Artifacts this server waits for by push.
    pub fn inbox(&self) -> Inbox {
        self.inbox.clone()
    }

    /// Downloads in progress this server shares as their chunks arrive.
    pub fn partials(&self) -> Partials {
        self.partials.clone()
    }

    /// Stop serving.
    pub async fn stop(self) {
        let _ = self.shutdown.send(true);
        let _ = self.handle.await;
    }
}

fn decode_signed<T: serde::de::DeserializeOwned>(
    headers: &HeaderMap,
    name: &str,
) -> Option<Signed<T>> {
    let raw = headers.get(name)?.to_str().ok()?;
    if raw.len() > 8192 {
        return None;
    }
    let bytes = hex::decode(raw).ok()?;
    Signed::from_canonical_bytes(&bytes).ok()
}

/// Take one pushed chunk (A8): a server-signed push ticket naming this server's transport key and
/// the artifact, still valid; an artifact this reader waits for; a chunk matching its manifest.
async fn receive_push(
    State(state): State<Arc<ShareState>>,
    AxumPath((hash, index)): AxumPath<(String, u32)>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let refuse = |state: &ShareState| {
        state
            .stats
            .refused_unauthorized
            .fetch_add(1, Ordering::Relaxed);
        (
            StatusCode::FORBIDDEN,
            "not authorized to push this artifact",
        )
            .into_response()
    };
    let Ok(hash) = hash.parse::<Sha256Digest>() else {
        return (StatusCode::BAD_REQUEST, "bad artifact hash").into_response();
    };
    let Some(ticket) = decode_signed::<PushTicketV1>(&headers, PUSH_TICKET_HEADER) else {
        return refuse(&state);
    };
    if ticket.signer != state.server_key
        || ticket.verify(&state.network, Domain::PushTicket).is_err()
        || ticket.body.network != state.network
        || ticket.body.artifact_hash != hash
        || ticket.body.reader_fingerprint != state.fingerprint
        || ticket.body.expires_at.0 <= UnixMillis::now().0
    {
        return refuse(&state);
    }
    match state.inbox.deliver(&ticket.body, index, body.to_vec()) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err((status, why)) => (status, why).into_response(),
    }
}

/// Serve one chunk after checking the server-signed ticket and the reader's signature.
async fn serve_chunk(
    State(state): State<Arc<ShareState>>,
    AxumPath((hash, index)): AxumPath<(String, u32)>,
    headers: HeaderMap,
) -> Response {
    let unauthorized = |state: &ShareState| {
        state
            .stats
            .refused_unauthorized
            .fetch_add(1, Ordering::Relaxed);
        (StatusCode::FORBIDDEN, "not authorized for this artifact").into_response()
    };
    let Ok(hash) = hash.parse::<Sha256Digest>() else {
        return (StatusCode::BAD_REQUEST, "bad artifact hash").into_response();
    };
    let Some(ticket) = decode_signed::<PeerTicketV1>(&headers, TICKET_HEADER) else {
        return unauthorized(&state);
    };
    let Some(request) = decode_signed::<ChunkRequestV1>(&headers, REQUEST_HEADER) else {
        return unauthorized(&state);
    };
    // The ticket must be the server's, for this network, this artifact, and still valid.
    if ticket.signer != state.server_key
        || ticket.verify(&state.network, Domain::PeerTicket).is_err()
        || ticket.body.network != state.network
        || ticket.body.artifact_hash != hash
        || ticket.body.expires_at.0 <= UnixMillis::now().0
    {
        return unauthorized(&state);
    }
    // The request must be signed by the key the ticket names, for the same artifact and chunk.
    if request.signer != ticket.body.reader_key
        || request
            .verify(&state.network, Domain::ChunkRequest)
            .is_err()
        || request.body.network != state.network
        || request.body.artifact_hash != hash
        || request.body.chunk_index != index
        || (request.body.issued_at.0 - UnixMillis::now().0).abs() > REQUEST_FRESHNESS_SECS * 1000
    {
        return unauthorized(&state);
    }
    let Some((start, len)) = ticket.body.range_of(index) else {
        return (StatusCode::RANGE_NOT_SATISFIABLE, "chunk out of range").into_response();
    };
    // Local limits come before any disk read.
    let Ok(_permit) = state.conns.clone().try_acquire_owned() else {
        state.stats.refused_budget.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::TOO_MANY_REQUESTS, "too many peer connections").into_response();
    };
    if state
        .budget_left
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
            left.checked_sub(len)
        })
        .is_err()
    {
        state.stats.refused_budget.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::TOO_MANY_REQUESTS, "upload budget exhausted").into_response();
    }
    // Held entirely (cache), or being downloaded with this chunk already verified (swarm).
    let read = match state.cache.read_range(&hash, ticket.body.size, start, len) {
        Ok(None) => Ok(state
            .partials
            .read(&hash, ticket.body.size, ticket.body.chunk_size, index)),
        other => other,
    };
    match read {
        Ok(Some(mut bytes)) => {
            state.stats.in_flight.fetch_add(1, Ordering::Relaxed);
            if let Some(u) = &state.uplink {
                u.take(bytes.len() as u64).await;
            }
            state.stats.in_flight.fetch_sub(1, Ordering::Relaxed);
            state
                .stats
                .last_served_ms
                .store(state.stats.now_ms().max(1), Ordering::Relaxed);
            if state.corrupt
                && let Some(b) = bytes.first_mut()
            {
                *b ^= 0xff;
            }
            state.stats.chunks_served.fetch_add(1, Ordering::Relaxed);
            state
                .stats
                .bytes_served
                .fetch_add(bytes.len() as u64, Ordering::Relaxed);

            (
                StatusCode::OK,
                [("content-type", "application/octet-stream")],
                bytes,
            )
                .into_response()
        }
        Ok(None) => {
            state.budget_left.fetch_add(len, Ordering::Relaxed);
            state.stats.refused_missing.fetch_add(1, Ordering::Relaxed);
            (StatusCode::NOT_FOUND, "artifact not held").into_response()
        }
        Err(e) => {
            state.budget_left.fetch_add(len, Ordering::Relaxed);
            state.stats.refused_missing.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(error = %e, %hash, "failed to read a cached chunk");
            (StatusCode::NOT_FOUND, "artifact not readable").into_response()
        }
    }
}

/// One peer as advertised by the control plane.
#[derive(Clone, Debug, Deserialize)]
pub struct PeerEntry {
    /// Holder, when a worker holds it (informational).
    #[serde(default)]
    pub worker_id: Option<WorkerId>,
    /// Lab node holding it, when the scientist's node does (ADR-0041; informational).
    #[serde(default)]
    pub node_id: Option<NodeId>,
    /// `ip:port`.
    pub addr: String,
    /// SHA-256 of the holder's TLS transport key, as recorded by the control plane. Pinned by the
    /// reader: no certificate authority, no hostname, no plaintext fallback (ADR-0013).
    pub tls_fingerprint: Sha256Digest,
}

/// Tracker answer: holders plus this reader's ticket.
#[derive(Clone, Debug, Deserialize)]
pub struct PeersResponse {
    /// Holders.
    pub peers: Vec<PeerEntry>,
    /// Ticket signed by the server.
    pub ticket: Signed<PeerTicketV1>,
    /// Whether the control plane serves the artifact itself. When it does not (an artifact held
    /// by a lab node, ADR-0041), a chunk no peer has is asked of the peers again, for up to
    /// `request_timeout_secs`, before the control plane is tried (and answers 409).
    #[serde(default = "server_holds_default")]
    pub server_holds: bool,
}

fn server_holds_default() -> bool {
    true
}

/// What a peer-assisted download actually did. Reported in logs and demo checks; never used to
/// decide correctness (that is the manifest's job).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FetchStats {
    /// Chunks obtained from peers.
    pub chunks_from_peers: u32,
    /// Chunks obtained from the control plane (fallback or no peer).
    pub chunks_from_http: u32,
    /// Chunks a peer served with wrong bytes.
    pub bad_chunks: u32,
    /// Peers dropped for the rest of the transfer after serving wrong bytes.
    pub peers_dropped: Vec<String>,
    /// TLS connections opened to peers: one per peer and transfer, unless one breaks.
    pub peer_connections: u32,
}

/// Settings of the fetching side.
#[derive(Clone, Debug)]
pub struct FetchConfig {
    /// Maximum peers tried for one artifact.
    pub max_peers: usize,
    /// Accept loopback/private peer addresses (single-host demos, laboratory networks).
    pub allow_private_peers: bool,
    /// Per-chunk timeout.
    pub request_timeout_secs: u64,
    /// Chunks requested at once. One at a time costs a round trip per chunk; several in flight
    /// fill the link instead, and bound what waits in memory for its turn to be written.
    pub parallel_chunks: usize,
    /// Where to list this download while it runs, so this worker's chunk server shares each chunk
    /// as soon as it is verified (swarm). `None`: nothing is shared before the end.
    pub partials: Option<Partials>,
    /// This worker's transport key fingerprint, as peers see it. With it, the chunks of an artifact
    /// are shared out between the readers that fetch it together: each asks the control plane only
    /// for the chunks it is responsible for, and takes the others from their reader.
    pub self_fingerprint: Option<Sha256Digest>,
    /// How long to wait for a chunk from the reader responsible for it before asking the control
    /// plane anyway (that reader may be slow or gone).
    pub swarm_patience_ms: u64,
    /// Route prefix of the reader's principal for manifests and peer lists: `/worker`, or
    /// `/node` for a lab node (ADR-0041).
    pub tracker: String,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            max_peers: 8,
            allow_private_peers: false,
            request_timeout_secs: 20,
            parallel_chunks: 4,
            partials: None,
            self_fingerprint: None,
            swarm_patience_ms: 2000,
            tracker: "/worker".into(),
        }
    }
}

/// Whether a peer address may be contacted (mirror of the server-side policy: a worker never
/// connects to an address the local policy refuses, whatever the tracker says).
pub fn peer_addr_allowed(addr: &SocketAddr, allow_private: bool) -> bool {
    use std::net::IpAddr;
    if addr.port() == 0 {
        return false;
    }
    match addr.ip() {
        IpAddr::V4(v4) => {
            if v4.is_unspecified() || v4.is_multicast() || v4.is_broadcast() || v4.is_link_local() {
                return false;
            }
            if v4.is_loopback() || v4.is_private() {
                return allow_private;
            }
            true
        }
        IpAddr::V6(v6) => {
            if v6.is_unspecified() || v6.is_multicast() {
                return false;
            }
            let seg = v6.segments()[0];
            if v6.is_loopback() || seg & 0xffc0 == 0xfe80 || seg & 0xfe00 == 0xfc00 {
                return allow_private;
            }
            true
        }
    }
}

/// Fetch and verify the signed chunk manifest of an artifact, from the routes under `tracker`
/// (`/worker` or `/node`).
pub async fn fetch_manifest(
    api: &Api,
    token: &str,
    tracker: &str,
    network: &NetworkId,
    server_key: &PublicKey,
    artifact: &ArtifactRef,
) -> Result<ChunkManifestV1, ClientError> {
    #[derive(Deserialize)]
    struct Resp {
        manifest: Signed<ChunkManifestV1>,
    }
    let resp: Resp = api
        .get_json(
            token,
            &format!("{tracker}/artifacts/{}/manifest", artifact.hash),
        )
        .await?;
    let m = resp.manifest;
    if m.signer != *server_key {
        return Err(ClientError::Integrity(
            "chunk manifest is not signed by the server key".into(),
        ));
    }
    m.verify(network, Domain::ChunkManifest)?;
    if m.body.network != *network
        || m.body.artifact_hash != artifact.hash
        || m.body.size != artifact.size
    {
        return Err(ClientError::Integrity(
            "chunk manifest does not describe the requested artifact".into(),
        ));
    }
    m.body
        .validate_structure()
        .map_err(|e| ClientError::Integrity(format!("chunk manifest: {e}")))?;
    Ok(m.body)
}

/// Download `artifact` to `dest`, preferring authorized peers and falling back to the control
/// plane chunk by chunk. Returns what each source provided.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_with_peers(
    api: &Api,
    token: &str,
    key: &SecretKey,
    network: &NetworkId,
    server_key: &PublicKey,
    artifact: &ArtifactRef,
    http_path: &str,
    dest: &Path,
    cfg: &FetchConfig,
) -> Result<FetchStats, ClientError> {
    let manifest = fetch_manifest(api, token, &cfg.tracker, network, server_key, artifact).await?;
    let answer: PeersResponse = api
        .get_json(
            token,
            &format!("{}/artifacts/{}/peers", cfg.tracker, artifact.hash),
        )
        .await?;
    if answer.ticket.signer != *server_key {
        return Err(ClientError::Integrity(
            "peer ticket is not signed by the server key".into(),
        ));
    }
    answer.ticket.verify(network, Domain::PeerTicket)?;
    if answer.ticket.body.artifact_hash != artifact.hash
        || answer.ticket.body.network != *network
        || answer.ticket.body.chunk_size != manifest.chunk_size
        || answer.ticket.body.size != manifest.size
    {
        return Err(ClientError::Integrity(
            "peer ticket does not match the manifest".into(),
        ));
    }
    let ticket_hex = hex::encode(
        answer
            .ticket
            .to_canonical_bytes()
            .map_err(ClientError::Protocol)?,
    );
    let allowed = |list: Vec<PeerEntry>| -> Vec<PeerEntry> {
        list.into_iter()
            .filter(|p| match p.addr.parse::<SocketAddr>() {
                Ok(a) => peer_addr_allowed(&a, cfg.allow_private_peers),
                Err(_) => false,
            })
            .take(cfg.max_peers)
            .collect()
    };
    let peers = std::sync::Mutex::new((allowed(answer.peers), std::time::Instant::now()));
    let server_holds = std::sync::atomic::AtomicBool::new(answer.server_holds);

    let timeout = std::time::Duration::from_secs(cfg.request_timeout_secs);
    // A suffix, not a replaced extension: `in.0` and `in.1` would both become `in.part`.
    let tmp = crate::client::part_path(dest);
    let file = tokio::fs::File::create(&tmp).await?;
    file.set_len(artifact.size).await?;
    let file = tokio::sync::Mutex::new(file);
    let count = manifest.chunks.len();
    // Listed while it runs: this worker's chunk server shares each chunk once verified and written.
    if let Some(partials) = &cfg.partials {
        partials.begin(
            artifact.hash,
            &tmp,
            manifest.size,
            manifest.chunk_size,
            count,
        );
    }
    // Each reader goes through the chunks in its own order, drawn from its key: workers that start
    // together fetch different chunks first, then exchange them, instead of all asking for the same
    // ones at the same moment (a shifted start is not enough: two nearby starts collide).
    let order: Vec<usize> = {
        let mut keyed: Vec<(Sha256Digest, usize)> = (0..count)
            .map(|i| {
                let mut seed = key.public_key().0.to_vec();
                seed.extend_from_slice(artifact.hash.as_bytes());
                seed.extend_from_slice(&(i as u64).to_le_bytes());
                (Sha256Digest::of_bytes(&seed), i)
            })
            .collect();
        keyed.sort();
        keyed.into_iter().map(|(_, i)| i).collect()
    };

    // Several chunks in flight (`parallel_chunks`), each from a peer or, failing that, from the
    // control plane, each checked against the signed manifest as soon as it arrives and written at
    // its place. At most `parallel` chunks wait in memory.
    let parallel = cfg.parallel_chunks.clamp(1, 16);
    let next = std::sync::atomic::AtomicUsize::new(0);
    // Chunks no peer held yet when first asked: set aside, retried once the fresh ones run out,
    // when the others have moved on. Only then does a chunk nobody holds go to the control plane.
    let deferred =
        std::sync::Mutex::new(std::collections::VecDeque::<(usize, std::time::Instant)>::new());
    // Rendezvous: the reader responsible for chunk `i` is the one, among this reader and the
    // peers listed, whose transport key gives the smallest digest with `i`. Every reader computes
    // the same answer from the same list; the list may differ briefly, which only costs a chunk
    // fetched twice.
    let responsible_is_me = |i: usize| -> bool {
        let Some(me) = cfg.self_fingerprint else {
            return true;
        };
        let score = |fp: &Sha256Digest| {
            let mut seed = fp.as_bytes().to_vec();
            seed.extend_from_slice(artifact.hash.as_bytes());
            seed.extend_from_slice(&(i as u64).to_le_bytes());
            Sha256Digest::of_bytes(&seed)
        };
        let mine = score(&me);
        let g = peers.lock().unwrap_or_else(|e| e.into_inner());
        g.0.iter()
            .filter(|p| p.tls_fingerprint != me)
            .all(|p| score(&p.tls_fingerprint) > mine)
    };
    let failed = std::sync::atomic::AtomicBool::new(false);
    let window = tokio::sync::Semaphore::new(parallel);
    let shared = std::sync::Mutex::new((FetchStats::default(), HashSet::<String>::new()));
    let (tx, mut rx) =
        tokio::sync::mpsc::unbounded_channel::<(usize, Result<Vec<u8>, ClientError>)>();
    let refresh_peers = || async {
        // Before falling back to the control plane: someone may have started holding the chunk
        // since the list was read (a worker that started with us). At most every 250 ms.
        let due = {
            let g = peers.lock().unwrap_or_else(|e| e.into_inner());
            g.1.elapsed() >= std::time::Duration::from_millis(250)
        };
        if !due {
            return false;
        }
        match api
            .get_json::<PeersResponse>(
                token,
                &format!("{}/artifacts/{}/peers", cfg.tracker, artifact.hash),
            )
            .await
        {
            Ok(fresh) => {
                server_holds.store(fresh.server_holds, std::sync::atomic::Ordering::SeqCst);
                let mut g = peers.lock().unwrap_or_else(|e| e.into_inner());
                *g = (allowed(fresh.peers), std::time::Instant::now());
                true
            }
            Err(_) => false,
        }
    };
    let slot = |_slot: usize| {
        let tx = tx.clone();
        let (next, failed, window, shared, peers) = (&next, &failed, &window, &shared, &peers);
        let (deferred, order, responsible_is_me) = (&deferred, &order, &responsible_is_me);
        let (manifest, ticket_hex, refresh_peers) = (&manifest, &ticket_hex, &refresh_peers);
        let server_holds = &server_holds;
        async move {
            let mut sessions: std::collections::HashMap<String, crate::peer_tls::PeerSession> =
                std::collections::HashMap::new();
            loop {
                match window.acquire().await {
                    Ok(p) => p.forget(),
                    Err(_) => return,
                }
                if failed.load(std::sync::atomic::Ordering::SeqCst) {
                    window.add_permits(1);
                    return;
                }
                let n_th = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let (i, waiting_since) = if n_th < count {
                    (order[n_th], None)
                } else {
                    match deferred
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .pop_front()
                    {
                        Some((i, since)) => (i, Some(since)),
                        None => {
                            window.add_permits(1);
                            return;
                        }
                    }
                };
                let index = i as u32;
                let expected = manifest.chunks[i];
                let Some((start, len)) = manifest.range_of(index) else {
                    let _ = tx.send((
                        i,
                        Err(ClientError::Integrity(
                            "manifest range out of bounds".into(),
                        )),
                    ));
                    return;
                };
                let mut bytes: Option<Vec<u8>> = None;
                for attempt in 0..2 {
                    let list: Vec<PeerEntry> =
                        peers.lock().unwrap_or_else(|e| e.into_inner()).0.clone();
                    let n = list.len();
                    // Peers first, starting with a different one per chunk to spread the load
                    // (consecutive chunks start on consecutive peers), skipping those already
                    // caught lying.
                    for step in 0..n {
                        let peer = &list[(i + step) % n];
                        if shared
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .1
                            .contains(&peer.addr)
                        {
                            continue;
                        }
                        if !sessions.contains_key(&peer.addr) {
                            match crate::peer_tls::PeerSession::connect(
                                &peer.addr,
                                peer.tls_fingerprint,
                                timeout,
                            )
                            .await
                            {
                                Ok(session) => {
                                    shared
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .0
                                        .peer_connections += 1;
                                    sessions.insert(peer.addr.clone(), session);
                                }
                                Err(e) => {
                                    tracing::debug!(peer = %peer.addr, error = %e, "peer unreachable, trying the next source");
                                    continue;
                                }
                            }
                        }
                        let session = sessions
                            .get_mut(&peer.addr)
                            .expect("session opened just above");
                        match fetch_chunk_from_peer(
                            session,
                            &peer.addr,
                            &artifact.hash,
                            index,
                            len,
                            key,
                            network,
                            ticket_hex,
                            cfg.request_timeout_secs,
                        )
                        .await
                        {
                            Ok(candidate) if Sha256Digest::of_bytes(&candidate) == expected => {
                                shared
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .0
                                    .chunks_from_peers += 1;
                                bytes = Some(candidate);
                                break;
                            }
                            Ok(_) => {
                                // Wrong bytes: this peer is out for the rest of the transfer.
                                // Nothing it sent is written, and the control plane still has
                                // the artifact.
                                let mut g = shared.lock().unwrap_or_else(|e| e.into_inner());
                                g.0.bad_chunks += 1;
                                if g.1.insert(peer.addr.clone()) {
                                    g.0.peers_dropped.push(peer.addr.clone());
                                }
                                tracing::warn!(peer = %peer.addr, chunk = index, artifact = %artifact.hash, "peer served bytes that do not match the signed manifest: dropping it");
                            }
                            Err(ClientError::Status { .. }) => {
                                // Not held (yet), or refused: the connection itself is fine.
                            }
                            Err(e) => {
                                // The connection may be what failed: the next chunk opens a
                                // fresh one.
                                sessions.remove(&peer.addr);
                                tracing::debug!(peer = %peer.addr, chunk = index, error = %e, "peer chunk failed, trying the next source");
                            }
                        }
                    }
                    if bytes.is_some() || attempt == 1 || !refresh_peers().await {
                        break;
                    }
                }
                if bytes.is_none() && !responsible_is_me(i) {
                    // Another reader is responsible for it and will have it soon: come back to it
                    // later, within `swarm_patience_ms`; only then ask the control plane anyway. It
                    // holds no memory meanwhile, so its place in the window is given back.
                    let since = waiting_since.unwrap_or_else(std::time::Instant::now);
                    if since.elapsed() < std::time::Duration::from_millis(cfg.swarm_patience_ms) {
                        if waiting_since.is_some() {
                            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                        }
                        deferred
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push_back((i, since));
                        window.add_permits(1);
                        continue;
                    }
                }
                if bytes.is_none() && !server_holds.load(std::sync::atomic::Ordering::SeqCst) {
                    // Every holder known served bytes that contradict the signed manifest: waiting
                    // brings nothing, and the control plane has no copy to fall back on.
                    let all_lied = {
                        let list = peers.lock().unwrap_or_else(|e| e.into_inner());
                        let dropped = shared.lock().unwrap_or_else(|e| e.into_inner());
                        !list.0.is_empty() && list.0.iter().all(|p| dropped.1.contains(&p.addr))
                    };
                    if all_lied {
                        failed.store(true, std::sync::atomic::Ordering::SeqCst);
                        let _ = tx.send((
                            i,
                            Err(ClientError::Integrity(format!(
                                "every holder of chunk {index} served bytes that do not match the signed manifest"
                            ))),
                        ));
                        return;
                    }
                    // The control plane has no copy (a lab node holds it, ADR-0041): the holder
                    // may be restarting or its announcement renewing. Ask the peers again, for up
                    // to `request_timeout_secs`, before trying the control plane.
                    let since = waiting_since.unwrap_or_else(std::time::Instant::now);
                    if since.elapsed() < timeout {
                        if waiting_since.is_some() {
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                        deferred
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push_back((i, since));
                        window.add_permits(1);
                        continue;
                    }
                }
                let result = match bytes {
                    Some(b) => Ok(b),
                    None => match api.download_range(token, http_path, start, len).await {
                        Ok(b) if Sha256Digest::of_bytes(&b) == expected => {
                            shared
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .0
                                .chunks_from_http += 1;
                            Ok(b)
                        }
                        Ok(_) => Err(ClientError::Integrity(format!(
                            "chunk {index} from the control plane does not match its signed manifest"
                        ))),
                        Err(e) => Err(e),
                    },
                };
                if result.is_err() {
                    failed.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                if tx.send((i, result)).is_err() {
                    return;
                }
            }
        }
    };
    let slots = futures_util::future::join_all((0..parallel).map(slot));
    let writer = async {
        let outcome: Result<(), ClientError> = async {
            for _ in 0..count {
                let (i, r) = rx
                    .recv()
                    .await
                    .ok_or_else(|| ClientError::Integrity("transfer ended early".into()))?;
                let chunk = r?;
                {
                    use tokio::io::{AsyncSeekExt, AsyncWriteExt};
                    let mut f = file.lock().await;
                    f.seek(std::io::SeekFrom::Start(
                        i as u64 * manifest.chunk_size as u64,
                    ))
                    .await?;
                    f.write_all(&chunk).await?;
                    f.flush().await?;
                }
                if let Some(partials) = &cfg.partials {
                    partials.mark(&artifact.hash, i);
                }
                window.add_permits(1);
            }
            Ok(())
        }
        .await;
        if outcome.is_err() {
            // Unblock every slot waiting for room in the window; they see `failed` and stop.
            failed.store(true, std::sync::atomic::Ordering::SeqCst);
            window.add_permits(parallel);
        }
        outcome
    };
    let (_, outcome) = tokio::join!(slots, writer);
    drop(tx);
    let file = file.into_inner();
    if let Err(e) = outcome {
        if let Some(partials) = &cfg.partials {
            partials.end(&artifact.hash);
        }
        drop(file);
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    let stats = shared.into_inner().unwrap_or_else(|e| e.into_inner()).0;
    file.sync_all().await?;
    drop(file);
    // Every chunk matched the signed manifest; the whole is hashed once more, in order, against the
    // hash of the artifact itself.
    let (hash, written) = {
        use tokio::io::AsyncReadExt;
        let mut f = tokio::fs::File::open(&tmp).await?;
        let mut hasher = Sha256Digest::hasher();
        let mut buf = vec![0u8; 1 << 20];
        let mut total = 0u64;
        loop {
            let n = f.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            total += n as u64;
        }
        (hasher.finalize(), total)
    };
    if written != artifact.size || hash != artifact.hash {
        if let Some(partials) = &cfg.partials {
            partials.end(&artifact.hash);
        }
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(ClientError::Integrity(format!(
            "assembled artifact mismatch: got {hash} ({written} bytes)"
        )));
    }
    tokio::fs::rename(&tmp, dest).await?;
    if let Some(partials) = &cfg.partials {
        partials.complete(&artifact.hash, dest);
    }
    Ok(stats)
}

/// One chunk from one holder over TLS 1.3 with its transport key pinned (ADR-0013). The reply is
/// bounded to the chunk size before anything is buffered.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_chunk_from_peer(
    session: &mut crate::peer_tls::PeerSession,
    addr: &str,
    hash: &Sha256Digest,
    index: u32,
    len: u64,
    key: &SecretKey,
    network: &NetworkId,
    ticket_hex: &str,
    timeout_secs: u64,
) -> Result<Vec<u8>, ClientError> {
    let request = ChunkRequestV1 {
        protocol_version: gg_protocol::PROTOCOL_VERSION,
        network: network.clone(),
        artifact_hash: *hash,
        chunk_index: index,
        issued_at: UnixMillis::now(),
    };
    let signed = Signed::sign(key, network, Domain::ChunkRequest, request)?;
    let request_hex = hex::encode(signed.to_canonical_bytes()?);
    let deadline = std::time::Duration::from_secs(timeout_secs);
    // The handshake only succeeds against the transport key the control plane published, and the
    // reply can never be larger than one chunk.
    let (status, bytes) = session
        .get(
            &format!("/p2p/v1/artifacts/{hash}/chunks/{index}"),
            &[
                (TICKET_HEADER, ticket_hex.to_owned()),
                (REQUEST_HEADER, request_hex),
            ],
            len as usize,
            deadline,
        )
        .await
        .map_err(ClientError::Peer)?;
    if status != 200 {
        return Err(ClientError::Status {
            status,
            message: format!("peer {addr} refused chunk {index}"),
        });
    }
    if bytes.len() as u64 != len {
        return Err(ClientError::Integrity(format!(
            "peer sent {} bytes for a {len}-byte chunk",
            bytes.len()
        )));
    }
    Ok(bytes)
}

/// Bind the chunk server. `[::]:port` is the useful default: the socket accepts IPv6 and IPv4 as
/// well (dual stack, set explicitly: Windows keeps IPv6 sockets IPv6-only by default). A machine
/// without IPv6 falls back to IPv4 instead of refusing to share.
async fn bind_listener(listen_addr: &str) -> std::io::Result<tokio::net::TcpListener> {
    let bound = match listen_addr.parse::<std::net::SocketAddr>() {
        Ok(addr) if addr.is_ipv6() && addr.ip().is_unspecified() => dual_stack(addr),
        _ => tokio::net::TcpListener::bind(listen_addr).await,
    };
    match bound {
        Ok(l) => Ok(l),
        Err(e) if listen_addr.starts_with("[::]") => {
            let port = listen_addr.rsplit(':').next().unwrap_or("0");
            let fallback = format!("0.0.0.0:{port}");
            tracing::info!(error = %e, %fallback, "no IPv6 socket available, sharing over IPv4 only");
            tokio::net::TcpListener::bind(fallback).await
        }
        Err(e) => Err(e),
    }
}

/// An IPv6 listener that also accepts IPv4 (mapped addresses).
fn dual_stack(addr: std::net::SocketAddr) -> std::io::Result<tokio::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    let s = Socket::new(Domain::IPV6, Type::STREAM, Some(Protocol::TCP))?;
    s.set_only_v6(false)?;
    // As tokio's own bind does on Unix: a restarted worker gets its port back at once.
    #[cfg(unix)]
    s.set_reuse_address(true)?;
    s.bind(&addr.into())?;
    s.listen(1024)?;
    s.set_nonblocking(true)?;
    tokio::net::TcpListener::from_std(s.into())
}

/// Receive `artifact` by push, when no holder can be reached but this reader can (ADR-0041, A8:
/// the reversed connection). The reader asks the control plane (`{tracker}/artifacts/{hash}/reverse`)
/// to have the holders connect to it at one of `addrs` (the first the control plane can dial back);
/// the holders push the chunks to `server`, which takes each one only if it matches the signed
/// manifest; the whole is checked against its hash before it lands at `dest`. `wait`: how long to
/// wait for the chunks.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_by_push(
    api: &Api,
    token: &str,
    tracker: &str,
    network: &NetworkId,
    server_key: &PublicKey,
    artifact: &ArtifactRef,
    dest: &Path,
    server: &PeerServer,
    addrs: &[String],
    wait: std::time::Duration,
) -> Result<(), ClientError> {
    let manifest = fetch_manifest(api, token, tracker, network, server_key, artifact).await?;
    let inbox = server.inbox();
    let mut rx = inbox.open(&manifest);
    let outcome = async {
        let mut asked = Err(ClientError::Integrity(
            "no address of this reader to be pushed to".into(),
        ));
        for addr in addrs {
            let body = serde_json::json!({ "addr": addr, "tls_fingerprint": server.fingerprint() });
            match api
                .post_json::<serde_json::Value>(
                    token,
                    &format!("{tracker}/artifacts/{}/reverse", artifact.hash),
                    &body,
                )
                .await
            {
                Ok(_) => {
                    asked = Ok(());
                    break;
                }
                Err(e) => {
                    tracing::debug!(%addr, error = %e, "the control plane cannot push to this address");
                    asked = Err(e);
                }
            }
        }
        asked?;
        let count = manifest.chunks.len();
        let tmp = crate::client::part_path(dest);
        let file = tokio::fs::File::create(&tmp).await?;
        file.set_len(artifact.size).await?;
        let mut file = file;
        let mut have = vec![false; count];
        let mut got = 0usize;
        let deadline = tokio::time::Instant::now() + wait;
        while got < count {
            let next = tokio::time::timeout_at(deadline, rx.recv()).await;
            let Ok(Some((index, bytes))) = next else {
                drop(file);
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(ClientError::Integrity(format!(
                    "{got} of {count} chunks pushed in time"
                )));
            };
            let i = index as usize;
            if have[i] {
                continue;
            }
            use tokio::io::{AsyncSeekExt, AsyncWriteExt};
            file.seek(std::io::SeekFrom::Start(
                index as u64 * manifest.chunk_size as u64,
            ))
            .await?;
            file.write_all(&bytes).await?;
            have[i] = true;
            got += 1;
        }
        file.sync_all().await?;
        drop(file);
        let hash = crate::cache::hash_file(&tmp)?;
        if hash != artifact.hash {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(ClientError::Integrity(format!(
                "assembled artifact mismatch: got {hash}"
            )));
        }
        tokio::fs::rename(&tmp, dest).await?;
        Ok(())
    }
    .await;
    inbox.close(&artifact.hash);
    outcome
}

/// A push the control plane asks a holder for (A8): connect to the reader and send it the chunks.
#[derive(Clone, Debug, Deserialize)]
pub struct PushRequest {
    /// Artifact to push.
    pub hash: Sha256Digest,
    /// Its size.
    pub size: u64,
    /// Chunk size of its manifest.
    pub chunk_size: u32,
    /// Where the reader listens (dialled back by the control plane).
    pub reader_addr: String,
    /// The reader's transport key, pinned.
    pub reader_fingerprint: Sha256Digest,
    /// Hex of the server-signed [`PushTicketV1`].
    pub ticket: String,
}

/// Push every chunk of `req.hash` from `cache` to the reader, over one TLS connection pinned to its
/// transport key. Stops as soon as the reader says it no longer waits.
pub async fn push_artifact(
    cache: &ArtifactCache,
    req: &PushRequest,
    timeout: std::time::Duration,
) -> Result<u32, String> {
    let mut session =
        crate::peer_tls::PeerSession::connect(&req.reader_addr, req.reader_fingerprint, timeout)
            .await?;
    let chunk = req.chunk_size as u64;
    let count = if req.size == 0 {
        1
    } else {
        req.size.div_ceil(chunk)
    };
    let mut sent = 0;
    for i in 0..count {
        let start = i * chunk;
        let len = chunk.min(req.size - start);
        let bytes = cache
            .read_range(&req.hash, req.size, start, len)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("{} is not held here", req.hash))?;
        let (status, body) = session
            .put(
                &format!("/p2p/v1/artifacts/{}/push/{i}", req.hash),
                &[(PUSH_TICKET_HEADER, req.ticket.clone())],
                bytes,
                timeout,
            )
            .await?;
        match status {
            204 | 200 => sent += 1,
            // The reader no longer waits (it got the artifact from elsewhere, or gave up).
            409 => break,
            other => {
                return Err(format!(
                    "reader refused chunk {i}: {other} {}",
                    String::from_utf8_lossy(&body)
                ));
            }
        }
    }
    Ok(sent)
}
