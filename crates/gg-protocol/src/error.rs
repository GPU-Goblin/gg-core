//! Protocol errors.

/// Errors raised by encoding, hashing, signing and parsing.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    /// Encoding failed (non-finite float, unsupported type, integer out of range).
    #[error("canonical encoding failed: {0}")]
    Encode(String),
    /// Decoding failed or input was not canonical.
    #[error("canonical decoding failed: {0}")]
    Decode(String),
    /// Signature verification failed.
    #[error("invalid signature")]
    InvalidSignature,
    /// Malformed key or signature bytes.
    #[error("malformed key material: {0}")]
    KeyMaterial(String),
    /// Message field out of bounds.
    #[error("message validation failed: {0}")]
    Validation(String),
    /// Tensor file error.
    #[error("tensor file: {0}")]
    TensorFile(#[from] crate::tensor_file::TensorFileError),
    /// I/O.
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
}
