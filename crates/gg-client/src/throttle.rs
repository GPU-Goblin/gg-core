//! Simulated uplink, for measurement (development only).
//!
//! A home connection shares its upstream bandwidth among everything leaving the machine: here,
//! uploads to the control plane and chunks served to peers. `Uplink` models it as a queue: each
//! transfer occupies the link for `bytes / rate` seconds, after the ones queued before it. This is
//! what makes it possible to measure, on a single machine, what the order of transfers changes
//! (ADR-0026). Setting `dev_faults.uplink_bytes_per_sec`, refused in public mode.

use std::time::Duration;

use tokio::time::Instant;

/// An uplink with a fixed rate, shared by all of the worker's transfers.
#[derive(Debug)]
pub struct Uplink {
    bytes_per_sec: u64,
    next_free: tokio::sync::Mutex<Instant>,
}

impl Uplink {
    /// A link of `bytes_per_sec` bytes per second.
    pub fn new(bytes_per_sec: u64) -> Self {
        Self {
            bytes_per_sec: bytes_per_sec.max(1),
            next_free: tokio::sync::Mutex::new(Instant::now()),
        }
    }

    /// Wait until `bytes` bytes have crossed the link, after those already queued.
    pub async fn take(&self, bytes: u64) {
        let end = {
            let mut next = self.next_free.lock().await;
            let start = (*next).max(Instant::now());
            let end = start + Duration::from_secs_f64(bytes as f64 / self.bytes_per_sec as f64);
            *next = end;
            end
        };
        tokio::time::sleep_until(end).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn two_transfers_share_the_link() {
        // 1 MB/s: 100 kB take 100 ms; the second transfer waits its turn.
        let u = std::sync::Arc::new(Uplink::new(1_000_000));
        let t0 = Instant::now();
        let (a, b) = (u.clone(), u.clone());
        let (ta, tb) = tokio::join!(
            async move {
                a.take(100_000).await;
                t0.elapsed()
            },
            async move {
                b.take(100_000).await;
                t0.elapsed()
            }
        );
        let (first, second) = if ta < tb { (ta, tb) } else { (tb, ta) };
        assert!(first >= Duration::from_millis(99), "{first:?}");
        assert!(second >= Duration::from_millis(199), "{second:?}");
        assert!(second < Duration::from_millis(600), "{second:?}");
    }
}
