//! TLS 1.3 between peers with RFC 7250 raw public keys (ADR-0013).
//!
//! A holder generates an **ephemeral** key pair when it starts sharing — never its protocol identity
//! key — and announces the SHA-256 of its `SubjectPublicKeyInfo` to the control plane. A reader gets
//! that fingerprint from the tracker (over its authenticated session) and pins it: the TLS handshake
//! succeeds only against the exact key the control plane vouched for. There is no certificate
//! authority, no hostname, and no plaintext fallback.
//!
//! TLS gives confidentiality and peer authentication. It is never what makes the bytes correct: every
//! chunk is still checked against the server-signed manifest (see [`crate::p2p`]).

use std::sync::Arc;

use gg_core::Sha256Digest;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{
    CertificateDer, PrivatePkcs8KeyDer, ServerName, SubjectPublicKeyInfoDer, UnixTime,
};
use rustls::server::AlwaysResolvesServerRawPublicKeys;
use rustls::sign::CertifiedKey;
use rustls::{
    ClientConfig, DigitallySignedStruct, Error as TlsError, ServerConfig, SignatureScheme,
};

/// Name presented in the handshake. Raw public keys carry no name; the verifier ignores it, and it
/// never leaves the machine.
const PEER_SERVER_NAME: &str = "peer.gpu-goblin.invalid";

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::aws_lc_rs::default_provider())
}

/// Ephemeral transport identity of a sharing worker.
#[derive(Debug)]
pub struct TlsIdentity {
    key_der: Vec<u8>,
    spki: Vec<u8>,
    fingerprint: Sha256Digest,
}

impl TlsIdentity {
    /// Generate a fresh Ed25519 transport key. Distinct from the worker's protocol key: a transport
    /// key lives as long as the process and signs nothing but TLS handshakes.
    pub fn generate() -> anyhow::Result<Self> {
        let pkcs8 = aws_lc_rs::signature::Ed25519KeyPair::generate_pkcs8(
            &aws_lc_rs::rand::SystemRandom::new(),
        )
        .map_err(|_| anyhow::anyhow!("failed to generate a transport key"))?;
        let key_der = pkcs8.as_ref().to_vec();
        let signing = provider()
            .key_provider
            .load_private_key(PrivatePkcs8KeyDer::from(key_der.clone()).into())?;
        let spki = signing
            .public_key()
            .ok_or_else(|| anyhow::anyhow!("transport key exposes no public key"))?
            .as_ref()
            .to_vec();
        let fingerprint = Sha256Digest::of_bytes(&spki);
        Ok(Self {
            key_der,
            spki,
            fingerprint,
        })
    }

    /// SHA-256 of the `SubjectPublicKeyInfo`: what the tracker publishes and readers pin.
    pub fn fingerprint(&self) -> Sha256Digest {
        self.fingerprint
    }

    /// Raw `SubjectPublicKeyInfo` bytes.
    pub fn spki(&self) -> &[u8] {
        &self.spki
    }

    /// TLS 1.3 server configuration presenting this raw public key.
    pub fn server_config(&self) -> anyhow::Result<Arc<ServerConfig>> {
        let p = provider();
        let signing = p
            .key_provider
            .load_private_key(PrivatePkcs8KeyDer::from(self.key_der.clone()).into())?;
        let certified = CertifiedKey::new(vec![CertificateDer::from(self.spki.clone())], signing);
        let cfg = ServerConfig::builder_with_provider(p)
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(AlwaysResolvesServerRawPublicKeys::new(Arc::new(
                certified,
            ))));
        Ok(Arc::new(cfg))
    }
}

/// Verifier that accepts exactly one raw public key, identified by the fingerprint the control plane
/// published for that holder.
#[derive(Debug)]
struct PinnedPeerKey {
    expected: Sha256Digest,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinnedPeerKey {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        // With raw public keys, `end_entity` carries the SubjectPublicKeyInfo itself.
        if Sha256Digest::of_bytes(end_entity.as_ref()) != self.expected {
            return Err(TlsError::InvalidCertificate(
                rustls::CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        // The configuration only offers TLS 1.3; reaching this is a protocol violation.
        Err(TlsError::PeerIncompatible(
            rustls::PeerIncompatible::Tls13RequiredForQuic,
        ))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls13_signature_with_raw_key(
            message,
            &SubjectPublicKeyInfoDer::from(cert.as_ref()),
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }

    fn requires_raw_public_keys(&self) -> bool {
        true
    }
}

/// TLS 1.3 client configuration pinned to one holder's transport key.
pub fn client_config(expected: Sha256Digest) -> anyhow::Result<Arc<ClientConfig>> {
    let p = provider();
    let cfg = ClientConfig::builder_with_provider(p.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinnedPeerKey {
            expected,
            provider: p,
        }))
        .with_no_client_auth();
    Ok(Arc::new(cfg))
}

/// Server name used in peer handshakes (ignored by the pinning verifier).
pub fn peer_server_name() -> ServerName<'static> {
    ServerName::try_from(PEER_SERVER_NAME).expect("static name is valid")
}

/// Serve `app` over TLS 1.3 on `listener` until `shutdown` flips, bounding live connections.
/// Used by the chunk server and by tests that need a peer with different behaviour.
pub async fn serve_tls(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    config: Arc<ServerConfig>,
    max_conns: u32,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    let acceptor = tokio_rustls::TlsAcceptor::from(config);
    let slots = Arc::new(tokio::sync::Semaphore::new((max_conns.max(1) * 2) as usize));
    loop {
        let accepted = tokio::select! {
            _ = shutdown.wait_for(|v| *v) => break,
            a = listener.accept() => a,
        };
        let Ok((stream, _from)) = accepted else {
            continue;
        };
        // A chunk goes out in several writes (TLS, headers, body): without TCP_NODELAY, the
        // reader pays for the delayed acknowledgment on every chunk.
        let _ = stream.set_nodelay(true);
        let Ok(slot) = slots.clone().try_acquire_owned() else {
            continue; // over the connection limit: drop it immediately
        };
        let acceptor = acceptor.clone();
        let app = app.clone();
        tokio::spawn(async move {
            let _slot = slot;
            let tls = match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                acceptor.accept(stream),
            )
            .await
            {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    tracing::debug!(error = %e, "peer TLS handshake failed");
                    return;
                }
                Err(_) => return,
            };
            let io = hyper_util::rt::TokioIo::new(tls);
            let svc = hyper_util::service::TowerToHyperService::new(app);
            // A reader keeps its connection for a whole transfer (one per peer, not per chunk);
            // one left idle between requests is closed after 30 s, so a place is never held
            // by a connection that asks nothing.
            let mut builder =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new());
            builder
                .http1()
                .timer(hyper_util::rt::TokioTimer::new())
                .header_read_timeout(std::time::Duration::from_secs(30));
            if let Err(e) = builder.serve_connection(io, svc).await {
                tracing::debug!(error = %e, "peer connection ended");
            }
        });
    }
}

/// A connection to one peer, TLS pinned to its published transport key, reused for every chunk
/// of a transfer. Opening one costs a TCP handshake and a TLS 1.3 handshake — two round trips —
/// which a connection per chunk would pay again for each mebibyte.
pub struct PeerSession {
    addr: String,
    sender: hyper::client::conn::http1::SendRequest<http_body_util::Full<hyper::body::Bytes>>,
    driver: tokio::task::JoinHandle<()>,
}

impl Drop for PeerSession {
    fn drop(&mut self) {
        self.driver.abort();
    }
}

impl PeerSession {
    /// Connect to `addr`; the handshake succeeds only against `fingerprint`.
    pub async fn connect(
        addr: &str,
        fingerprint: Sha256Digest,
        timeout: std::time::Duration,
    ) -> Result<Self, String> {
        let tcp = tokio::time::timeout(timeout, tokio::net::TcpStream::connect(addr))
            .await
            .map_err(|_| format!("connect to {addr} timed out"))?
            .map_err(|e| format!("connect to {addr}: {e}"))?;
        let _ = tcp.set_nodelay(true);
        let connector = tokio_rustls::TlsConnector::from(
            client_config(fingerprint).map_err(|e| e.to_string())?,
        );
        let tls = tokio::time::timeout(timeout, connector.connect(peer_server_name(), tcp))
            .await
            .map_err(|_| format!("TLS handshake with {addr} timed out"))?
            .map_err(|e| format!("TLS handshake with {addr}: {e}"))?;
        let io = hyper_util::rt::TokioIo::new(tls);
        let (sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .map_err(|e| format!("http handshake with {addr}: {e}"))?;
        let driver = tokio::spawn(async move {
            let _ = conn.await;
        });
        Ok(Self {
            addr: addr.to_owned(),
            sender,
            driver,
        })
    }

    /// One GET on this connection, with explicit headers. Returns the status and at most
    /// `max_body` bytes.
    pub async fn get(
        &mut self,
        path: &str,
        headers: &[(&str, String)],
        max_body: usize,
        timeout: std::time::Duration,
    ) -> Result<(u16, Vec<u8>), String> {
        self.request(
            hyper::Method::GET,
            path,
            headers,
            Vec::new(),
            max_body,
            timeout,
        )
        .await
    }

    /// One PUT of `body` on this connection (a chunk pushed to a reader, ADR-0041 A8).
    pub async fn put(
        &mut self,
        path: &str,
        headers: &[(&str, String)],
        body: Vec<u8>,
        timeout: std::time::Duration,
    ) -> Result<(u16, Vec<u8>), String> {
        self.request(hyper::Method::PUT, path, headers, body, 4096, timeout)
            .await
    }

    async fn request(
        &mut self,
        method: hyper::Method,
        path: &str,
        headers: &[(&str, String)],
        body: Vec<u8>,
        max_body: usize,
        timeout: std::time::Duration,
    ) -> Result<(u16, Vec<u8>), String> {
        use http_body_util::BodyExt;

        let addr = self.addr.clone();
        tokio::time::timeout(timeout, self.sender.ready())
            .await
            .map_err(|_| format!("{addr}: connection not ready in time"))?
            .map_err(|e| format!("{addr}: connection closed: {e}"))?;
        let mut builder = hyper::Request::builder()
            .method(method)
            .uri(path)
            .header(hyper::header::HOST, addr.as_str());
        for (name, value) in headers {
            builder = builder.header(*name, value);
        }
        let request = builder
            .body(http_body_util::Full::new(hyper::body::Bytes::from(body)))
            .map_err(|e| e.to_string())?;
        let response = tokio::time::timeout(timeout, self.sender.send_request(request))
            .await
            .map_err(|_| format!("{addr} did not answer in time"))?
            .map_err(|e| format!("request to {addr}: {e}"))?;
        let status = response.status().as_u16();
        if let Some(declared) = response
            .headers()
            .get(hyper::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok())
            && declared > max_body
        {
            return Err(format!(
                "peer announced {declared} bytes, limit is {max_body}"
            ));
        }
        let body = http_body_util::Limited::new(response.into_body(), max_body);
        let collected = tokio::time::timeout(timeout, body.collect())
            .await
            .map_err(|_| format!("{addr} stalled while sending a body"))?
            .map_err(|e| format!("body from {addr}: {e}"))?;
        Ok((status, collected.to_bytes().to_vec()))
    }
}

/// One GET to a peer over TLS pinned to `fingerprint`, with explicit headers, on a fresh
/// connection. Returns the status and at most `max_body` bytes. The chunk fetcher and the tests go
/// through [`PeerSession`] the same way, so no test can exercise a path production does not take.
pub async fn peer_get(
    addr: &str,
    fingerprint: Sha256Digest,
    path: &str,
    headers: &[(&str, String)],
    max_body: usize,
    timeout: std::time::Duration,
) -> Result<(u16, Vec<u8>), String> {
    let mut session = PeerSession::connect(addr, fingerprint, timeout).await?;
    session.get(path, headers, max_body, timeout).await
}
