//! Logical network identifier bound into every signature and hash.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

/// Network identifier: 1..=32 characters in `[a-z0-9-]`.
#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct NetworkId(String);

/// Invalid network id.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("network id must be 1..=32 characters in [a-z0-9-]")]
pub struct NetworkIdError;

impl NetworkId {
    /// Validate and build.
    pub fn new(s: &str) -> Result<Self, NetworkIdError> {
        if s.is_empty() || s.len() > 32 {
            return Err(NetworkIdError);
        }
        if !s
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err(NetworkIdError);
        }
        Ok(Self(s.to_owned()))
    }

    /// As text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NetworkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for NetworkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NetworkId({})", self.0)
    }
}

impl FromStr for NetworkId {
    type Err = NetworkIdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl<'de> Deserialize<'de> for NetworkId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        NetworkId::new(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_charset() {
        assert!(NetworkId::new("gg-dev-local").is_ok());
        assert!(NetworkId::new("").is_err());
        assert!(NetworkId::new("Bad").is_err());
        assert!(NetworkId::new("a\nb").is_err());
        assert!(NetworkId::new(&"a".repeat(33)).is_err());
    }
}
