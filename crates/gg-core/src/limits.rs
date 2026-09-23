//! Resource limits attached to a task (conservative, enforced by the executor and the catalog).

use serde::{Deserialize, Serialize};

/// Resource limits for one attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum resident memory for the executor process, in bytes.
    pub max_ram_bytes: u64,
    /// Maximum scratch disk usage, in bytes.
    pub max_disk_bytes: u64,
    /// Maximum wall-clock duration of the attempt, in seconds.
    pub max_duration_secs: u32,
    /// Maximum total size of produced outputs, in bytes.
    pub max_output_bytes: u64,
}

impl ResourceLimits {
    /// Validate that the limits are within global bounds.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_ram_bytes == 0 || self.max_ram_bytes > 64 << 30 {
            return Err("max_ram_bytes out of range (1..=64GiB)");
        }
        if self.max_disk_bytes == 0 || self.max_disk_bytes > 256 << 30 {
            return Err("max_disk_bytes out of range (1..=256GiB)");
        }
        if self.max_duration_secs == 0 || self.max_duration_secs > 24 * 3600 {
            return Err("max_duration_secs out of range (1..=86400)");
        }
        if self.max_output_bytes == 0 || self.max_output_bytes > 16 << 30 {
            return Err("max_output_bytes out of range (1..=16GiB)");
        }
        Ok(())
    }
}
