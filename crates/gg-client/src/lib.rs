//! Client side of the control plane, shared by the worker, the validator and the `gg` CLI: the
//! authenticated HTTP API with backpressure and resumable transfers, and the local artifact cache
//! keyed by SHA-256.

#![deny(unsafe_code)]

pub mod cache;
pub mod client;
pub mod signals;
pub mod throttle;
