//! Domain model of GPU Goblin: pure types and invariants, no I/O, no framework dependency.
//!
//! Everything here is deterministic and testable without a database, a network or a GPU.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod credits;
pub mod digest;
pub mod ids;
pub mod limits;
pub mod network;
#[cfg(feature = "openapi")]
mod openapi;
pub mod state;
pub mod time;

pub use credits::{Credits, CreditsError};
pub use digest::{DigestParseError, Sha256Digest};
pub use ids::*;
pub use limits::ResourceLimits;
pub use network::{NetworkId, NetworkIdError};
pub use state::*;
pub use time::UnixMillis;
