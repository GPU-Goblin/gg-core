//! SHA-256 digests used for artifacts and domain-separated message hashes.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// A SHA-256 digest (32 bytes). Human-readable formats use lowercase hex; binary formats use raw bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sha256Digest(pub [u8; 32]);

/// Error parsing a digest from text.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DigestParseError {
    /// Not 64 hex characters.
    #[error("digest must be 64 hex characters")]
    Length,
    /// Invalid hex.
    #[error("digest is not valid hex")]
    Hex,
}

impl Sha256Digest {
    /// Hash raw bytes (used for artifacts, no domain tag).
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(bytes);
        Self(h.finalize().into())
    }

    /// Incremental hasher.
    pub fn hasher() -> Sha256Hasher {
        Sha256Hasher(Sha256::new())
    }

    /// Lowercase hex.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Raw bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// All-zero digest (fixtures only).
    pub const fn zero() -> Self {
        Self([0u8; 32])
    }
}

/// Streaming SHA-256 hasher.
pub struct Sha256Hasher(Sha256);

impl Sha256Hasher {
    /// Feed bytes.
    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    /// Finish.
    pub fn finalize(self) -> Sha256Digest {
        Sha256Digest(self.0.finalize().into())
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.to_hex())
    }
}

impl FromStr for Sha256Digest {
    type Err = DigestParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.strip_prefix("sha256:").unwrap_or(s);
        if s.len() != 64 {
            return Err(DigestParseError::Length);
        }
        let v = hex::decode(s).map_err(|_| DigestParseError::Hex)?;
        let mut out = [0u8; 32];
        out.copy_from_slice(&v);
        Ok(Self(out))
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            serializer.serialize_str(&self.to_hex())
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            s.parse().map_err(serde::de::Error::custom)
        } else {
            let b: serde_bytes_compat::ByteBuf = Deserialize::deserialize(deserializer)?;
            let b = b.0;
            if b.len() != 32 {
                return Err(serde::de::Error::custom("digest must be 32 bytes"));
            }
            let mut out = [0u8; 32];
            out.copy_from_slice(&b);
            Ok(Self(out))
        }
    }
}

/// Minimal byte-buffer visitor so `gg-core` does not depend on `serde_bytes`.
mod serde_bytes_compat {
    use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};
    use std::fmt;

    pub struct ByteBuf(pub Vec<u8>);

    impl<'de> Deserialize<'de> for ByteBuf {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = ByteBuf;
                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("bytes")
                }
                fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<ByteBuf, E> {
                    Ok(ByteBuf(v.to_vec()))
                }
                fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<ByteBuf, E> {
                    Ok(ByteBuf(v))
                }
                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<ByteBuf, A::Error> {
                    let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(64));
                    while let Some(b) = seq.next_element::<u8>()? {
                        out.push(b);
                    }
                    Ok(ByteBuf(out))
                }
            }
            d.deserialize_bytes(V)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        // NIST "abc"
        let d = Sha256Digest::of_bytes(b"abc");
        assert_eq!(
            d.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(d.to_string().parse::<Sha256Digest>().unwrap(), d);
        assert_eq!(
            format!("sha256:{}", d.to_hex())
                .parse::<Sha256Digest>()
                .unwrap(),
            d
        );
    }

    #[test]
    fn json_is_hex() {
        let d = Sha256Digest::of_bytes(b"");
        let j = serde_json::to_string(&d).unwrap();
        assert_eq!(j, format!("\"{}\"", d.to_hex()));
        assert_eq!(serde_json::from_str::<Sha256Digest>(&j).unwrap(), d);
        assert!(serde_json::from_str::<Sha256Digest>("\"abc\"").is_err());
    }
}
