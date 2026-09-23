//! Domain separation and network binding for hashes and signatures.

use gg_core::{NetworkId, Sha256Digest};
use serde::Serialize;

use crate::ProtocolError;

/// Prefix of every tagged message; includes the protocol version.
pub const TAG_PREFIX: &str = "gpu-goblin/v1\n";

/// Signing/hashing domains. Each domain yields incompatible byte strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Scientific definition of a task (identity-free).
    TaskDefinition,
    /// Task specification (definition + instance ids).
    TaskSpec,
    /// Attempt assignment envelope (signed by the server).
    Assignment,
    /// Result submission (signed by the worker agent).
    Result,
    /// Verification decision (signed by a validator).
    Verification,
    /// Contribution receipt (signed by the server).
    Receipt,
    /// Worker / validator login challenge.
    Login,
    /// Chunk manifest of an artifact (signed by the server, phase P5).
    ChunkManifest,
    /// Peer read ticket (signed by the server, phase P5).
    PeerTicket,
    /// Chunk request from a reader to a holder (signed by the reader, phase P5).
    ChunkRequest,
    /// Signed release manifest of the distributed executables (phase P8).
    Release,
    /// Merkle leaf (reward batches, phase P7).
    MerkleLeaf,
    /// Merkle inner node.
    MerkleNode,
}

impl Domain {
    /// Stable textual tag.
    pub const fn as_str(self) -> &'static str {
        match self {
            Domain::TaskDefinition => "task-definition",
            Domain::TaskSpec => "task-spec",
            Domain::Assignment => "assignment",
            Domain::Result => "result",
            Domain::Verification => "verification",
            Domain::Receipt => "receipt",
            Domain::Login => "worker-login",
            Domain::ChunkManifest => "chunk-manifest",
            Domain::PeerTicket => "peer-ticket",
            Domain::ChunkRequest => "chunk-request",
            Domain::Release => "release-manifest",
            Domain::MerkleLeaf => "merkle-leaf",
            Domain::MerkleNode => "merkle-node",
        }
    }
}

/// Build the tagged message: `prefix || network || "\n" || domain || "\n" || bytes`.
pub fn tagged(network: &NetworkId, domain: Domain, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(TAG_PREFIX.len() + 64 + bytes.len());
    out.extend_from_slice(TAG_PREFIX.as_bytes());
    out.extend_from_slice(network.as_str().as_bytes());
    out.push(b'\n');
    out.extend_from_slice(domain.as_str().as_bytes());
    out.push(b'\n');
    out.extend_from_slice(bytes);
    out
}

/// Domain-separated hash of already-canonical bytes.
pub fn hash_bytes(network: &NetworkId, domain: Domain, canonical: &[u8]) -> Sha256Digest {
    Sha256Digest::of_bytes(&tagged(network, domain, canonical))
}

/// Domain-separated hash of a message (encoded canonically first).
pub fn hash_message<T: Serialize>(
    network: &NetworkId,
    domain: Domain,
    msg: &T,
) -> Result<Sha256Digest, ProtocolError> {
    let bytes = crate::canonical::to_bytes(msg)?;
    Ok(hash_bytes(network, domain, &bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_domains_and_networks_differ() {
        let n1 = NetworkId::new("gg-dev-local").unwrap();
        let n2 = NetworkId::new("gg-other").unwrap();
        let a = hash_bytes(&n1, Domain::Result, b"x");
        let b = hash_bytes(&n1, Domain::Receipt, b"x");
        let c = hash_bytes(&n2, Domain::Result, b"x");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(
            tagged(&n1, Domain::Result, b"x"),
            b"gpu-goblin/v1\ngg-dev-local\nresult\nx".to_vec()
        );
    }
}
