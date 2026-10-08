//! Client side of the control plane, shared by the worker, the validator and the `gg` CLI: the
//! authenticated HTTP API with backpressure and resumable transfers, and the local artifact cache
//! keyed by SHA-256.

#![deny(unsafe_code)]

pub mod cache;
pub mod client;
#[cfg(feature = "p2p")]
pub mod nat_probe;
#[cfg(feature = "p2p")]
pub mod natpmp;
#[cfg(feature = "p2p")]
pub mod p2p;
#[cfg(feature = "p2p")]
pub mod peer_tls;
#[cfg(feature = "p2p")]
pub mod reach;
pub mod signals;
pub mod throttle;
#[cfg(feature = "p2p")]
pub mod upnp;
pub mod worker_control;
