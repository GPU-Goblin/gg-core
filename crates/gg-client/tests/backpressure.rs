//! A control plane that pushes a request back (429, quota; 503, transaction to replay) does not
//! make the client fail: it waits for `Retry-After` and replays. Without this, a validator gave up
//! on a verification it could not even submit, which then stayed reserved for ten minutes.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use gg_client::client::{Api, ClientError};

async fn serve(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn pushed_back_requests_are_replayed_and_other_errors_are_not() {
    let calls = Arc::new(AtomicUsize::new(0));
    async fn flaky(State(calls): State<Arc<AtomicUsize>>) -> Response {
        match calls.fetch_add(1, Ordering::SeqCst) {
            0 => (
                StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", "1")],
                "rate limit exceeded",
            )
                .into_response(),
            1 => (StatusCode::SERVICE_UNAVAILABLE, "retry").into_response(),
            _ => axum::Json(serde_json::json!({ "ok": true })).into_response(),
        }
    }
    async fn refused() -> Response {
        (
            StatusCode::CONFLICT,
            axum::Json(serde_json::json!({ "message": "lease expired" })),
        )
            .into_response()
    }
    let refusals = Arc::new(AtomicUsize::new(0));
    let r2 = refusals.clone();
    let app = Router::new()
        .route("/v1/flaky", post(flaky))
        .with_state(calls.clone())
        .route(
            "/v1/refused",
            post(move || {
                r2.fetch_add(1, Ordering::SeqCst);
                refused()
            }),
        );
    let api = Api::new(&serve(app).await).unwrap();

    let t0 = std::time::Instant::now();
    let v: serde_json::Value = api
        .post_json("t", "/flaky", &serde_json::json!({}))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert!(
        t0.elapsed() >= std::time::Duration::from_secs(1),
        "Retry-After honoured"
    );

    match api
        .post_json::<serde_json::Value>("t", "/refused", &serde_json::json!({}))
        .await
    {
        Err(ClientError::Status { status: 409, .. }) => {}
        other => panic!("a 409 is an answer, not a push-back: {other:?}"),
    }
    assert_eq!(refusals.load(Ordering::SeqCst), 1);
}
