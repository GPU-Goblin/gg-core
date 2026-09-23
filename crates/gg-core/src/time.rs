//! Protocol timestamps: integer milliseconds since the Unix epoch (UTC).

use std::fmt;

use serde::{Deserialize, Serialize};

/// Milliseconds since 1970-01-01T00:00:00Z. Always an integer so that binary and JSON encodings agree.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct UnixMillis(pub i64);

impl UnixMillis {
    /// Current wall-clock time (server or agent clock; only the server clock is authoritative).
    pub fn now() -> Self {
        let d = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Self(i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
    }

    /// Add milliseconds (saturating).
    pub fn plus_millis(self, ms: i64) -> Self {
        Self(self.0.saturating_add(ms))
    }

    /// Add seconds (saturating).
    pub fn plus_secs(self, s: i64) -> Self {
        self.plus_millis(s.saturating_mul(1000))
    }

    /// Raw value.
    pub const fn get(self) -> i64 {
        self.0
    }
}

impl fmt::Display for UnixMillis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ms", self.0)
    }
}

impl fmt::Debug for UnixMillis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnixMillis({})", self.0)
    }
}
