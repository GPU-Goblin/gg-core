//! Deterministic expansion `RG-DRBG-SHA256-CTR v1`.
//!
//! ```text
//! key      = SHA-256("gpu-goblin/drbg/v1\n" || domain || "\n" || seed as u64 LE)
//! block(i) = SHA-256(key || i as u64 LE)          (i = 0, 1, 2, ...)
//! stream   = block(0) || block(1) || ...
//! u32      = next 4 stream bytes, little-endian
//! unit f32 = (u32 >> 8) * 2^-24                   (in [0, 1), 24 bits of precision, exact in f32)
//! sym f32  = 2 * unit - 1                         (in [-1, 1))
//! ```
//!
//! This is a specification device for protocol-fixed data (synthetic datasets, initial weights,
//! example order); it is not used for key generation.

use sha2::{Digest, Sha256};

/// Deterministic byte stream generator.
#[derive(Clone, Debug)]
pub struct Drbg {
    key: [u8; 32],
    counter: u64,
    block: [u8; 32],
    pos: usize,
}

impl Drbg {
    /// Create a generator for `domain` and `seed`.
    pub fn new(domain: &str, seed: u64) -> Self {
        let mut h = Sha256::new();
        h.update(b"gpu-goblin/drbg/v1\n");
        h.update(domain.as_bytes());
        h.update(b"\n");
        h.update(seed.to_le_bytes());
        let key: [u8; 32] = h.finalize().into();
        let mut d = Self {
            key,
            counter: 0,
            block: [0; 32],
            pos: 32,
        };
        d.refill();
        d
    }

    fn refill(&mut self) {
        let mut h = Sha256::new();
        h.update(self.key);
        h.update(self.counter.to_le_bytes());
        self.block = h.finalize().into();
        self.counter = self.counter.wrapping_add(1);
        self.pos = 0;
    }

    /// Next byte.
    pub fn next_u8(&mut self) -> u8 {
        if self.pos >= 32 {
            self.refill();
        }
        let b = self.block[self.pos];
        self.pos += 1;
        b
    }

    /// Next u32 (little-endian from the stream).
    pub fn next_u32(&mut self) -> u32 {
        let b = [
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
        ];
        u32::from_le_bytes(b)
    }

    /// Next u64.
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        lo | (hi << 32)
    }

    /// Uniform in [0, 1) with 24 bits of precision.
    pub fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform in [-1, 1).
    pub fn next_sym_f32(&mut self) -> f32 {
        2.0 * self.next_unit_f32() - 1.0
    }

    /// Uniform in [lo, hi).
    pub fn next_range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit_f32()
    }

    /// Fill a vector with symmetric values.
    pub fn sym_f32_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_sym_f32()).collect()
    }

    /// Uniform index in `0..n` (rejection sampling, unbiased). `n` must be > 0.
    pub fn next_index(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        let zone = u32::MAX - (u32::MAX % n);
        loop {
            let v = self.next_u32();
            if v < zone {
                return v % n;
            }
        }
    }

    /// Fisher–Yates permutation of `0..n` (deterministic).
    pub fn permutation(&mut self, n: u32) -> Vec<u32> {
        let mut p: Vec<u32> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.next_index(i + 1) as usize;
            p.swap(i as usize, j);
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_domain_separated() {
        let a: Vec<u32> = (0..8)
            .map({
                let mut d = Drbg::new("test", 1);
                move |_| d.next_u32()
            })
            .collect();
        let b: Vec<u32> = (0..8)
            .map({
                let mut d = Drbg::new("test", 1);
                move |_| d.next_u32()
            })
            .collect();
        let c: Vec<u32> = (0..8)
            .map({
                let mut d = Drbg::new("test", 2);
                move |_| d.next_u32()
            })
            .collect();
        let e: Vec<u32> = (0..8)
            .map({
                let mut d = Drbg::new("other", 1);
                move |_| d.next_u32()
            })
            .collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, e);
    }

    #[test]
    fn first_block_matches_spec() {
        // Recompute the specification by hand.
        let mut h = Sha256::new();
        h.update(b"gpu-goblin/drbg/v1\n");
        h.update(b"vec");
        h.update(b"\n");
        h.update(7u64.to_le_bytes());
        let key: [u8; 32] = h.finalize().into();
        let mut h = Sha256::new();
        h.update(key);
        h.update(0u64.to_le_bytes());
        let block: [u8; 32] = h.finalize().into();
        let mut d = Drbg::new("vec", 7);
        let got: Vec<u8> = (0..32).map(|_| d.next_u8()).collect();
        assert_eq!(got, block.to_vec());
        assert_eq!(d.next_u32(), {
            let mut h = Sha256::new();
            h.update(key);
            h.update(1u64.to_le_bytes());
            let b2: [u8; 32] = h.finalize().into();
            u32::from_le_bytes([b2[0], b2[1], b2[2], b2[3]])
        });
    }

    #[test]
    fn floats_in_range_and_permutation_valid() {
        let mut d = Drbg::new("f", 3);
        for _ in 0..10_000 {
            let u = d.next_unit_f32();
            assert!((0.0..1.0).contains(&u));
            let s = d.next_sym_f32();
            assert!((-1.0..1.0).contains(&s));
        }
        let p = d.permutation(100);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        assert_ne!(p, (0..100).collect::<Vec<_>>());
    }
}
