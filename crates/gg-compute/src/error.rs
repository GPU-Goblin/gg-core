//! Compute errors.

use gg_protocol::tensor_file::TensorFileError;

/// Errors of the catalog, execution and verification.
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    /// Compute kind not in the catalog.
    #[error("unknown compute kind: {0}")]
    UnknownKind(String),
    /// Kind exists in the protocol but is not available in this catalog version.
    #[error("compute kind {0} is not available in catalog version {1}")]
    KindNotAvailable(&'static str, u32),
    /// Parameters out of bounds.
    #[error("invalid parameters: {0}")]
    InvalidParams(String),
    /// Missing or unexpected input.
    #[error("inputs: {0}")]
    Inputs(String),
    /// Numeric profile not admitted.
    #[error("numeric profile not admitted: {0}")]
    ProfileNotAdmitted(String),
    /// Input tensor invalid.
    #[error("input '{name}': {source}")]
    InputTensor {
        /// Input name.
        name: String,
        /// Cause.
        #[source]
        source: TensorFileError,
    },
    /// Engine unavailable (e.g. GPU requested but not compiled in or not present).
    #[error("engine unavailable: {0}")]
    EngineUnavailable(String),
    /// Output exceeds limits.
    #[error("output limit exceeded: {0}")]
    OutputLimit(String),
    /// The task needs more memory than this executor may use (C1).
    #[error("memory limit: {0}")]
    MemoryLimit(String),
    /// Protocol error.
    #[error(transparent)]
    Protocol(#[from] gg_protocol::ProtocolError),
    /// I/O.
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
}
