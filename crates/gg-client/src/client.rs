//! HTTP client for the worker routes (TLS verified by reqwest/rustls defaults).

use std::path::Path;

use gg_core::*;
use gg_protocol::domain::Domain;
use gg_protocol::keys::{PublicKey, SecretKey};
use gg_protocol::messages::{LoginChallengeV1, ResultSubmissionV1};
use gg_protocol::signed::Signed;
use serde::Deserialize;
use serde_json::Value;

/// Check the scheme of a control-plane URL: `https`, or `http` to this machine, or `http` anywhere
/// when `plain_http_anywhere` is set.
pub fn check_transport(base: &str, plain_http_anywhere: bool) -> Result<(), ClientError> {
    let url = reqwest::Url::parse(base)
        .map_err(|e| ClientError::Config(format!("server URL '{base}': {e}")))?;
    match url.scheme() {
        "https" => Ok(()),
        "http" if plain_http_anywhere || is_loopback(&url) => Ok(()),
        "http" => Err(ClientError::Config(format!(
            "server URL '{base}' uses plain http to another machine: use https (plain http is \
             accepted only to this machine)"
        ))),
        other => Err(ClientError::Config(format!(
            "server URL '{base}': unsupported scheme '{other}'"
        ))),
    }
}

fn is_loopback(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Traffic that a background upload gives way to: on a worker, the chunks it is serving to its
/// peers, which sit on the critical path of the next stage (ADR-0026).
pub trait YieldTo: Send + Sync {
    /// Whether that traffic is running now, or finished less than `quiet_ms` milliseconds ago.
    fn busy(&self, quiet_ms: u64) -> bool;
}

/// Client errors.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// Transport.
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    /// Non-success status with the server's error body.
    #[error("server returned {status}: {message}")]
    Status {
        /// HTTP status.
        status: u16,
        /// Message.
        message: String,
    },
    /// Protocol.
    #[error(transparent)]
    Protocol(#[from] gg_protocol::ProtocolError),
    /// Peer transport (TLS, connection, framing).
    #[error("peer: {0}")]
    Peer(String),
    /// Integrity.
    #[error("integrity: {0}")]
    Integrity(String),
    /// Configuration refused before any request (for example, plain http to another machine).
    #[error("configuration: {0}")]
    Config(String),
    /// I/O.
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    /// JSON.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Server discovery info.
#[derive(Clone, Debug, Deserialize)]
pub struct ServerInfo {
    /// Network id.
    pub network_id: NetworkId,
    /// Protocol version.
    pub protocol_version: u32,
    /// Catalog version.
    pub catalog_version: u32,
    /// Server public key (hex).
    pub server_public_key: PublicKey,
    /// Dev mode flag.
    pub dev_mode: bool,
    /// Chunk size of peer manifests; `0` when the server does not distribute by peers (or predates
    /// ADR-0026), in which case activations are uploaded before submission as before.
    #[serde(default)]
    pub p2p_chunk_bytes: u32,
    /// Largest output of a job a lab node holds that the server accepts as an upload (ADR-0041).
    #[serde(default)]
    pub node_relay_max_bytes: u64,
    /// How long a declared output waits for its holder.
    #[serde(default)]
    pub pending_output_secs: i64,
}

/// Answer to an announcement: what the control plane observed and recorded.
#[derive(Clone, Debug, Deserialize)]
pub struct Announced {
    /// Source address the control plane saw. A worker behind a NAT that preserves the port can
    /// announce this address instead of guessing its own (ADR-0013).
    pub observed_ip: String,
    /// Address recorded.
    pub addr: String,
    /// Seconds before the announcement expires.
    pub ttl_secs: i64,
}

/// Registration response.
#[derive(Clone, Debug, Deserialize)]
pub struct Registered {
    /// Worker id.
    pub worker_id: WorkerId,
    /// Authorized flag.
    pub authorized: bool,
}

/// Session response.
#[derive(Clone, Debug, Deserialize)]
pub struct Session {
    /// Token.
    pub token: String,
    /// TTL.
    pub expires_in_secs: i64,
    /// Principal id (worker or validator record).
    pub principal_id: uuid::Uuid,
    /// `worker` or `validator`.
    pub principal_kind: String,
}

/// Base client.
#[derive(Clone)]
pub struct Api {
    http: reqwest::Client,
    base: String,
    /// Simulated uplink applied to uploads (development measurements only).
    uplink: Option<std::sync::Arc<crate::throttle::Uplink>>,
}

async fn check(r: reqwest::Response) -> Result<reqwest::Response, ClientError> {
    let status = r.status();
    if status.is_success() {
        return Ok(r);
    }
    let text = r.text().await.unwrap_or_default();
    let message = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(str::to_owned))
        .unwrap_or(text);
    Err(ClientError::Status {
        status: status.as_u16(),
        message,
    })
}

/// Longest a request waits in total for the server to accept it again (429, 503).
const BACKPRESSURE_BUDGET: std::time::Duration = std::time::Duration::from_secs(60);

/// How long the server asks to wait (`Retry-After`, in seconds), bounded; otherwise a short
/// backoff growing with the attempt.
fn retry_after(r: &reqwest::Response, attempt: u32) -> std::time::Duration {
    r.headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(std::time::Duration::from_secs)
        .unwrap_or(std::time::Duration::from_millis(200 * u64::from(attempt)))
        .min(std::time::Duration::from_secs(10))
}

impl Api {
    /// Send the request `make` builds, and replay it while the server pushes back: 429 (rate
    /// limit, `Retry-After` honoured) or 503 (a transaction the database aborted and that can be
    /// replayed as is). A push-back is not a failure: turning it into one made a validator give up a
    /// verification it could not even release, which then held its lease for ten minutes.
    async fn send(
        &self,
        make: impl Fn() -> reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, ClientError> {
        let mut waited = std::time::Duration::ZERO;
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let r = make().send().await?;
            let status = r.status().as_u16();
            if matches!(status, 429 | 503) && waited < BACKPRESSURE_BUDGET {
                let wait = retry_after(&r, attempt);
                tracing::debug!(
                    status,
                    wait_ms = wait.as_millis() as u64,
                    "server pushed back, replaying the request"
                );
                tokio::time::sleep(wait).await;
                waited += wait;
                continue;
            }
            return check(r).await;
        }
    }

    /// New client for `base`. The control plane must be reached over `https://`; plain `http://`
    /// is accepted only to this machine (`localhost`, `127.0.0.0/8`, `::1`), where nothing crosses
    /// a network. Tokens, session keys and signed results travel on this connection.
    pub fn new(base: &str) -> Result<Self, ClientError> {
        check_transport(base, false)?;
        Self::build(base)
    }

    /// Like [`Api::new`], but also accepts plain `http://` to another machine. Only for a
    /// development grid on a private network: the caller decides it explicitly (a worker or
    /// validator outside public mode, or `GG_ALLOW_INSECURE_HTTP=1` for the CLI).
    pub fn new_allowing_plain_http(base: &str) -> Result<Self, ClientError> {
        check_transport(base, true)?;
        Self::build(base)
    }

    fn build(base: &str) -> Result<Self, ClientError> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_owned(),
            uplink: None,
        })
    }

    /// Send uploads through a simulated uplink (development measurements only).
    pub fn with_uplink(mut self, uplink: Option<std::sync::Arc<crate::throttle::Uplink>>) -> Self {
        self.uplink = uplink;
        self
    }

    /// Shared HTTP client (peer requests reuse the same connection pool and TLS settings).
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    fn url(&self, path: &str) -> String {
        format!("{}/v1{path}", self.base)
    }

    /// PUT raw bytes (a relayed chunk, ADR-0041); the answer's body is ignored.
    pub async fn put_bytes(
        &self,
        token: &str,
        path: &str,
        bytes: Vec<u8>,
    ) -> Result<(), ClientError> {
        let r = self
            .send(|| {
                self.http
                    .put(self.url(path))
                    .bearer_auth(token)
                    .header("content-type", "application/octet-stream")
                    .body(bytes.clone())
            })
            .await?;
        check(r).await.map(|_| ())
    }

    /// GET /v1/info.
    pub async fn info(&self) -> Result<ServerInfo, ClientError> {
        Ok(self
            .send(|| self.http.get(self.url("/info")))
            .await?
            .json()
            .await?)
    }

    /// GET json with a bearer token.
    pub async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        token: &str,
        path: &str,
    ) -> Result<T, ClientError> {
        Ok(self
            .send(|| self.http.get(self.url(path)).bearer_auth(token))
            .await?
            .json()
            .await?)
    }

    /// POST json, expecting json (or `None` on 204).
    pub async fn post_json<T: serde::de::DeserializeOwned>(
        &self,
        token: &str,
        path: &str,
        body: &Value,
    ) -> Result<Option<T>, ClientError> {
        let r = self
            .send(|| self.http.post(self.url(path)).bearer_auth(token).json(body))
            .await?;
        if r.status().as_u16() == 204 {
            return Ok(None);
        }
        Ok(Some(r.json().await?))
    }

    /// Register a worker key.
    pub async fn register_worker(
        &self,
        key: &PublicKey,
        display_name: &str,
    ) -> Result<Registered, ClientError> {
        let body = serde_json::json!({ "public_key": key.to_hex(), "display_name": display_name });
        Ok(self
            .post_json("", "/worker/register", &body)
            .await?
            .expect("json"))
    }

    /// Challenge-based login for a worker, validator or lab node key.
    pub async fn login(
        &self,
        key: &SecretKey,
        network: &NetworkId,
        action: &str,
    ) -> Result<Session, ClientError> {
        let prefix = match action {
            "worker-login" => "/worker",
            "node-login" => "/node",
            _ => "/validator",
        };
        let ch: Value = self
            .post_json(
                "",
                &format!("{prefix}/challenge"),
                &serde_json::json!({ "public_key": key.public_key().to_hex() }),
            )
            .await?
            .expect("json");
        let nonce = hex::decode(ch["nonce"].as_str().unwrap_or_default())
            .map_err(|e| ClientError::Integrity(e.to_string()))?;
        let body = LoginChallengeV1 {
            protocol_version: gg_protocol::PROTOCOL_VERSION,
            network: network.clone(),
            action: action.to_owned(),
            nonce,
            public_key: key.public_key(),
            issued_at: UnixMillis::now(),
        };
        let signed = Signed::sign(key, network, Domain::Login, body)?;
        Ok(self
            .post_json(
                "",
                &format!("{prefix}/session"),
                &serde_json::to_value(&signed)?,
            )
            .await?
            .expect("json"))
    }

    /// Download an artifact to `dest`, verifying hash and size (never more than `max` bytes).
    /// Interrupted transfers resume with `Range` requests (up to 5 attempts); the hash is computed
    /// over the bytes actually received, so a server that ignores ranges restarts the download.
    /// `download` with bounded retries (6 tries, exponential backoff capped at 10 s) on transient
    /// failures: quota `429`, `502/503/504`, transport errors. Other errors are returned at once.
    pub async fn download_retry(
        &self,
        token: &str,
        path: &str,
        expected: &gg_protocol::messages::ArtifactRef,
        max: u64,
        dest: &Path,
    ) -> Result<(), ClientError> {
        let mut delay = std::time::Duration::from_secs(1);
        for attempt in 1..=6u32 {
            match self.download(token, path, expected, max, dest).await {
                Ok(()) => return Ok(()),
                Err(e) if attempt < 6 && is_transient(&e) => {
                    tracing::warn!(error = %e, attempt, wait_secs = delay.as_secs(), "transient download failure, retrying");
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(std::time::Duration::from_secs(10));
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("bounded retry loop")
    }

    pub async fn download(
        &self,
        token: &str,
        path: &str,
        expected: &gg_protocol::messages::ArtifactRef,
        max: u64,
        dest: &Path,
    ) -> Result<(), ClientError> {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;
        if expected.size > max {
            return Err(ClientError::Integrity(format!(
                "artifact size {} exceeds local limit {max}",
                expected.size
            )));
        }
        // A suffix, not a replaced extension: `in.0` and `in.1` would both become `in.part`.
        let tmp = part_path(dest);
        let mut received: u64 = 0;
        let mut hasher = Sha256Digest::hasher();
        let mut file = tokio::fs::File::create(&tmp).await?;
        let mut attempts = 0u32;
        loop {
            attempts += 1;
            let mut req = self.http.get(self.url(path)).bearer_auth(token);
            if received > 0 {
                req = req.header("range", format!("bytes={received}-"));
            }
            let r = match req.send().await {
                Ok(r) if r.status().is_success() => r,
                // A refusal keeps its status and message (`409`: not stored here, read it from
                // peers), as every other call of this client reports it.
                Ok(r) => {
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(check(r).await.expect_err("status is not a success"));
                }
                Err(e) if attempts < 5 && (e.is_connect() || e.is_timeout() || e.is_request()) => {
                    tokio::time::sleep(std::time::Duration::from_millis(200 * attempts as u64))
                        .await;
                    continue;
                }
                Err(e) => {
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(e.into());
                }
            };
            if received > 0 && r.status().as_u16() != 206 {
                // Server ignored the range: restart from scratch.
                received = 0;
                hasher = Sha256Digest::hasher();
                file = tokio::fs::File::create(&tmp).await?;
            } else if received > 0 {
                let cr = r
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                if !cr.starts_with(&format!("bytes {received}-")) {
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(ClientError::Integrity(format!(
                        "unexpected content-range '{cr}' after {received} bytes"
                    )));
                }
            }
            let mut stream = r.bytes_stream();
            let mut interrupted = false;
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        if attempts < 5 {
                            tracing::warn!(error = %e, received, "download interrupted, resuming");
                            interrupted = true;
                            break;
                        }
                        drop(file);
                        let _ = tokio::fs::remove_file(&tmp).await;
                        return Err(e.into());
                    }
                };
                received += chunk.len() as u64;
                if received > expected.size {
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(ClientError::Integrity(
                        "server sent more bytes than announced".into(),
                    ));
                }
                hasher.update(&chunk);
                file.write_all(&chunk).await?;
            }
            if interrupted || (received < expected.size && attempts < 5) {
                if !interrupted {
                    tracing::warn!(received, expected = expected.size, "short read, resuming");
                }
                tokio::time::sleep(std::time::Duration::from_millis(200 * attempts as u64)).await;
                continue;
            }
            break;
        }
        file.sync_all().await?;
        drop(file);
        let hash = hasher.finalize();
        if received != expected.size || hash != expected.hash {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(ClientError::Integrity(format!(
                "downloaded artifact mismatch: got {hash} ({received} bytes)"
            )));
        }
        tokio::fs::rename(&tmp, dest).await?;
        Ok(())
    }

    /// Announce that this worker holds an artifact and serves its chunks at `addr`.
    pub async fn announce_artifact(
        &self,
        token: &str,
        hash: &gg_core::Sha256Digest,
        addr: &str,
        tls_fingerprint: &gg_core::Sha256Digest,
    ) -> Result<Announced, ClientError> {
        let body = serde_json::json!({ "addr": addr, "tls_fingerprint": tls_fingerprint });
        Ok(self
            .send(|| {
                self.http
                    .post(self.url(&format!("/worker/artifacts/{hash}/announce")))
                    .bearer_auth(token)
                    .json(&body)
            })
            .await?
            .json()
            .await?)
    }

    /// Fetch `len` bytes at `start` of an artifact with a `Range` request. Used as the per-chunk
    /// fallback when no peer serves a chunk: the control plane is always the durable source.
    pub async fn download_range(
        &self,
        token: &str,
        path: &str,
        start: u64,
        len: u64,
    ) -> Result<Vec<u8>, ClientError> {
        let end = start + len - 1;
        let r = self
            .send(|| {
                self.http
                    .get(self.url(path))
                    .bearer_auth(token)
                    .header("range", format!("bytes={start}-{end}"))
            })
            .await?;
        let status = r.status().as_u16();
        let bytes = r.bytes().await?;
        if bytes.len() as u64 != len {
            return Err(ClientError::Integrity(format!(
                "range {start}-{end} returned {} bytes (status {status})",
                bytes.len()
            )));
        }
        Ok(bytes.to_vec())
    }

    /// Upload a file as an artifact.
    pub async fn upload(
        &self,
        token: &str,
        path: &str,
        file: &Path,
        hash: &Sha256Digest,
        size: u64,
        schema: &str,
    ) -> Result<Value, ClientError> {
        self.upload_yielding(token, path, file, hash, size, schema, None, 0)
            .await
    }

    /// Upload in the background, pausing between blocks while `yield_to` is serving peers: the
    /// next stage reading an activation from this worker is on the critical path, its durable copy
    /// is not (ADR-0026). Yielding stops after `max_yield_secs` in total.
    #[allow(clippy::too_many_arguments)]
    pub async fn upload_yielding(
        &self,
        token: &str,
        path: &str,
        file: &Path,
        hash: &Sha256Digest,
        size: u64,
        schema: &str,
        yield_to: Option<std::sync::Arc<dyn YieldTo>>,
        max_yield_secs: u64,
    ) -> Result<Value, ClientError> {
        if !path.ends_with(&hash.to_hex()) {
            return Err(ClientError::Integrity(
                "upload path does not end with the artifact hash".into(),
            ));
        }
        let url = format!("{}?size={size}&schema={schema}", self.url(path));
        let mut waited = std::time::Duration::ZERO;
        let mut attempt = 0u32;
        let r = loop {
            attempt += 1;
            let f = tokio::fs::File::open(file).await?;
            let stream = tokio_util::io::ReaderStream::with_capacity(f, 64 * 1024);
            let body = if self.uplink.is_none() && yield_to.is_none() {
                reqwest::Body::wrap_stream(stream)
            } else {
                use futures_util::StreamExt;
                let uplink = self.uplink.clone();
                let yield_to = yield_to.clone();
                let budget = std::time::Duration::from_secs(max_yield_secs);
                let yielded = std::sync::Arc::new(std::sync::Mutex::new(std::time::Duration::ZERO));
                reqwest::Body::wrap_stream(stream.then(move |chunk| {
                    let (uplink, yield_to, yielded) =
                        (uplink.clone(), yield_to.clone(), yielded.clone());
                    async move {
                        if let Some(stats) = &yield_to {
                            let t0 = std::time::Instant::now();
                            let spent = *yielded.lock().unwrap_or_else(|e| e.into_inner());
                            while stats.busy(100) && spent + t0.elapsed() < budget {
                                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                            }
                            *yielded.lock().unwrap_or_else(|e| e.into_inner()) += t0.elapsed();
                        }
                        if let (Some(u), Ok(b)) = (&uplink, &chunk) {
                            u.take(b.len() as u64).await;
                        }
                        chunk
                    }
                }))
            };
            let r = self
                .http
                .put(&url)
                .bearer_auth(token)
                .header("content-length", size)
                .body(body)
                .send()
                .await?;
            if matches!(r.status().as_u16(), 429 | 503) && waited < BACKPRESSURE_BUDGET {
                // Pushed back (rate limit, transaction to replay): the body is a stream, so the
                // file is read again from the start.
                let wait = retry_after(&r, attempt);
                tokio::time::sleep(wait).await;
                waited += wait;
                continue;
            }
            break check(r).await?;
        };
        Ok(r.json().await?)
    }

    /// Submit a signed result.
    pub async fn submit_result(
        &self,
        token: &str,
        signed: &Signed<ResultSubmissionV1>,
    ) -> Result<Value, ClientError> {
        let path = format!("/worker/attempts/{}/result", signed.body.attempt_id);
        Ok(self
            .post_json(token, &path, &serde_json::to_value(signed)?)
            .await?
            .expect("json"))
    }

    /// Heartbeat.
    pub async fn heartbeat(&self, token: &str, attempt: AttemptId) -> Result<Value, ClientError> {
        Ok(self
            .post_json(
                token,
                &format!("/worker/attempts/{attempt}/heartbeat"),
                &serde_json::json!({}),
            )
            .await?
            .expect("json"))
    }

    /// Abandon.
    pub async fn abandon(
        &self,
        token: &str,
        attempt: AttemptId,
        reason: &str,
    ) -> Result<Value, ClientError> {
        Ok(self
            .post_json(
                token,
                &format!("/worker/attempts/{attempt}/abandon"),
                &serde_json::json!({ "reason": reason }),
            )
            .await?
            .expect("json"))
    }
}

/// Whether a failure is worth retrying: rate limiting, upstream/transient server errors, transport.
/// Temporary path of a download into `dest`: `dest` with `.part` appended.
pub fn part_path(dest: &Path) -> std::path::PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(".part");
    std::path::PathBuf::from(name)
}

pub fn is_transient(e: &ClientError) -> bool {
    match e {
        ClientError::Status { status, .. } => matches!(status, 429 | 502 | 503 | 504),
        ClientError::Http(e) => match e.status() {
            Some(s) => s.as_u16() == 429 || s.is_server_error(),
            None => e.is_connect() || e.is_timeout() || e.is_request() || e.is_body(),
        },
        _ => false,
    }
}

#[cfg(test)]
mod transport_tests {
    use super::check_transport;

    #[test]
    fn plain_http_is_accepted_only_to_this_machine() {
        for ok in [
            "https://grid.example.org",
            "http://127.0.0.1:8080",
            "http://127.1.2.3",
            "http://localhost:8080",
            "http://[::1]:8080",
        ] {
            assert!(check_transport(ok, false).is_ok(), "{ok}");
        }
        for refused in [
            "http://grid.example.org",
            "http://192.168.1.10:8080",
            "http://[2001:db8::1]:8080",
            "http://localhost.example.org",
        ] {
            let e = check_transport(refused, false).unwrap_err().to_string();
            assert!(e.contains("use https"), "{refused}: {e}");
        }
        assert!(check_transport("http://192.168.1.10:8080", true).is_ok());
        assert!(check_transport("ftp://grid.example.org", true).is_err());
        assert!(check_transport("not a url", true).is_err());
    }
}
