//! Resumable downloads: a server that cuts the connection mid-transfer must not defeat the client,
//! and the reassembled bytes must hash correctly.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::get;
use gg_client::client::Api;
use gg_core::Sha256Digest;
use gg_protocol::messages::ArtifactRef;

#[derive(Clone)]
struct Flaky {
    data: Arc<Vec<u8>>,
    requests: Arc<AtomicUsize>,
}

async fn serve(State(f): State<Flaky>, headers: HeaderMap) -> Response {
    let n = f.requests.fetch_add(1, Ordering::SeqCst);
    let size = f.data.len() as u64;
    let start: u64 = headers
        .get("range")
        .and_then(|v| v.to_str().ok())
        .and_then(|r| r.strip_prefix("bytes="))
        .and_then(|r| r.split('-').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let slice = f.data[start as usize..].to_vec();
    // First request: send only the first third, then end the body early (short read).
    let body = if n == 0 {
        slice[..slice.len() / 3].to_vec()
    } else {
        slice
    };
    let mut b = Response::builder()
        .header("content-length", size - start)
        .header("accept-ranges", "bytes");
    if start > 0 {
        b = b.status(StatusCode::PARTIAL_CONTENT).header(
            "content-range",
            format!("bytes {start}-{}/{size}", size - 1),
        );
    }
    b.body(Body::from(body)).unwrap()
}

#[tokio::test]
async fn download_resumes_after_a_short_read_and_verifies_hash() {
    let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
    let hash = Sha256Digest::of_bytes(&data);
    let flaky = Flaky {
        data: Arc::new(data.clone()),
        requests: Arc::new(AtomicUsize::new(0)),
    };
    let app = Router::new()
        .route("/v1/blob", get(serve))
        .with_state(flaky.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let api = Api::new(&format!("http://{addr}")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("blob");
    api.download(
        "t",
        "/blob",
        &ArtifactRef {
            hash,
            size: data.len() as u64,
        },
        1 << 20,
        &dest,
    )
    .await
    .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(
        flaky.requests.load(Ordering::SeqCst) >= 2,
        "a resume request must have happened"
    );
    assert!(!dir.path().join("blob.part").exists());

    // A wrong announced hash is detected after reassembly.
    let dest2 = dir.path().join("blob2");
    let err = api
        .download(
            "t",
            "/blob",
            &ArtifactRef {
                hash: Sha256Digest::zero(),
                size: data.len() as u64,
            },
            1 << 20,
            &dest2,
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("mismatch"), "{err}");
    assert!(!dest2.exists() && !dir.path().join("blob2.part").exists());
}

/// A refusal keeps its status and the server's message: a `409` (not stored here, read it from
/// peers) is what makes a worker or a validator turn to peers (ADR-0026, ADR-0041).
#[tokio::test]
async fn a_refused_download_reports_its_status_and_message() {
    let app = Router::new().route(
        "/v1/blob",
        get(|| async {
            (
                StatusCode::CONFLICT,
                [("content-type", "application/json")],
                r#"{"message":"artifact held by a lab node: fetch it from peers"}"#,
            )
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let api = Api::new(&format!("http://{addr}")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("blob");
    let r = ArtifactRef {
        hash: Sha256Digest::of_bytes(b"x"),
        size: 1,
    };
    for result in [
        api.download("t", "/blob", &r, 1 << 20, &dest).await,
        api.download_retry("t", "/blob", &r, 1 << 20, &dest).await,
    ] {
        match result {
            Err(gg_client::client::ClientError::Status { status, message }) => {
                assert_eq!(status, 409);
                assert!(message.contains("lab node"), "{message}");
            }
            other => panic!("expected a 409 status, got {other:?}"),
        }
    }
    assert!(!dest.exists());
}
