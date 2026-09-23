//! Wire protocol of GPU Goblin.
//!
//! - [`canonical`]: deterministic DAG-CBOR encoding (ADR-0003).
//! - [`domain`]: domain-separated, network-bound hashing (ADR-0005).
//! - [`keys`] and [`signed`]: Ed25519 identities and signed envelopes.
//! - [`messages`]: versioned messages (task spec, assignment, result, decision, receipt, login).
//! - [`tensor_file`]: bounded binary tensor format `GGT1` (ADR-0007).
//! - [`drbg`]: deterministic byte/float expansion used for protocol-fixed data and weights.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod canonical;
pub mod domain;
pub mod drbg;
pub mod error;
pub mod keys;
pub mod messages;
pub mod release;
pub mod signed;
pub mod tensor_file;

pub use error::ProtocolError;
pub use gg_core;

/// Protocol version carried by every message and bound into every signature.
pub const PROTOCOL_VERSION: u32 = 1;
