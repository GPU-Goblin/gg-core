//! Integer credits with an explicit unit (`TC`, test credits). No monetary value is promised.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Unit label for credits.
pub const CREDIT_UNIT: &str = "TC";

/// An amount of test credits. Integer, never fractional. JSON representation is a decimal string so
/// that no precision is lost in JavaScript clients; binary formats use an unsigned integer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Credits(pub u64);

/// Arithmetic error on credits.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CreditsError {
    /// Overflow.
    #[error("credit amount overflow")]
    Overflow,
    /// Insufficient amount for a subtraction.
    #[error("insufficient credits: have {have}, need {need}")]
    Insufficient {
        /// Available amount.
        have: u64,
        /// Requested amount.
        need: u64,
    },
    /// Parse error.
    #[error("invalid credit amount")]
    Parse,
}

impl Credits {
    /// Zero.
    pub const ZERO: Credits = Credits(0);

    /// Build from a raw amount.
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    /// Raw amount.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Checked addition.
    pub fn checked_add(self, other: Credits) -> Result<Credits, CreditsError> {
        self.0
            .checked_add(other.0)
            .map(Credits)
            .ok_or(CreditsError::Overflow)
    }

    /// Checked subtraction.
    pub fn checked_sub(self, other: Credits) -> Result<Credits, CreditsError> {
        self.0
            .checked_sub(other.0)
            .map(Credits)
            .ok_or(CreditsError::Insufficient {
                have: self.0,
                need: other.0,
            })
    }

    /// Checked multiplication by a count.
    pub fn checked_mul(self, n: u64) -> Result<Credits, CreditsError> {
        self.0
            .checked_mul(n)
            .map(Credits)
            .ok_or(CreditsError::Overflow)
    }

    /// Is zero.
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Credits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, CREDIT_UNIT)
    }
}

impl fmt::Debug for Credits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Credits({})", self.0)
    }
}

impl FromStr for Credits {
    type Err = CreditsError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let s = s.strip_suffix(CREDIT_UNIT).map(str::trim).unwrap_or(s);
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(CreditsError::Parse);
        }
        s.parse::<u64>()
            .map(Credits)
            .map_err(|_| CreditsError::Parse)
    }
}

impl Serialize for Credits {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            serializer.serialize_str(&self.0.to_string())
        } else {
            serializer.serialize_u64(self.0)
        }
    }
}

impl<'de> Deserialize<'de> for Credits {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            s.parse().map_err(serde::de::Error::custom)
        } else {
            u64::deserialize(deserializer).map(Credits)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_is_checked() {
        assert_eq!(Credits(1).checked_add(Credits(2)).unwrap(), Credits(3));
        assert_eq!(
            Credits(u64::MAX).checked_add(Credits(1)),
            Err(CreditsError::Overflow)
        );
        assert_eq!(
            Credits(1).checked_sub(Credits(2)),
            Err(CreditsError::Insufficient { have: 1, need: 2 })
        );
    }

    #[test]
    fn json_is_lossless_string() {
        let c = Credits(18_446_744_073_709_551_615);
        let j = serde_json::to_string(&c).unwrap();
        assert_eq!(j, "\"18446744073709551615\"");
        assert_eq!(serde_json::from_str::<Credits>(&j).unwrap(), c);
        assert!(serde_json::from_str::<Credits>("\"-1\"").is_err());
        assert!(serde_json::from_str::<Credits>("\"1.5\"").is_err());
        assert_eq!("42 TC".parse::<Credits>().unwrap(), Credits(42));
    }
}
