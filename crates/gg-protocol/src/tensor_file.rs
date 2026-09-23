//! `GGT1` bounded binary tensor format (ADR-0007).
//!
//! ```text
//! magic "GGT1" | u8 dtype (1 = f32 LE, 2 = bf16 LE) | u8 ndim (1..=4) | u64 LE shape[ndim] | data
//! ```
//!
//! bf16 is the upper half of an f32, rounded to nearest even (ADR-0035). It carries gradients only:
//! a reader always states the element type it expects, and a file of the other type is refused.

use std::io::Read;
use std::path::Path;

/// Magic bytes.
pub const MAGIC: &[u8; 4] = b"GGT1";
/// dtype code for little-endian f32.
pub const DTYPE_F32: u8 = 1;
/// dtype code for little-endian bfloat16.
pub const DTYPE_BF16: u8 = 2;

/// Element type of a tensor file. Decoded tensors are always f32: bf16 widens exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dtype {
    /// 4 bytes per element.
    #[default]
    F32,
    /// 2 bytes per element: sign, 8 exponent bits, 7 mantissa bits.
    Bf16,
}

impl Dtype {
    /// Code in the header.
    pub fn code(self) -> u8 {
        match self {
            Dtype::F32 => DTYPE_F32,
            Dtype::Bf16 => DTYPE_BF16,
        }
    }

    /// Bytes per element.
    pub fn width(self) -> u64 {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 => 2,
        }
    }
}

/// `x` rounded to the nearest bf16, ties to even (as PyTorch's conversion). `x` must be finite; the
/// result may overflow to an infinity, which the caller checks.
pub fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

/// The f32 value of bf16 bits (exact).
pub fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
/// Maximum number of dimensions.
pub const MAX_NDIM: usize = 4;
/// Default maximum number of elements (2^26 → 256 MiB of f32).
pub const DEFAULT_MAX_ELEMENTS: u64 = 1 << 26;

/// Errors of the tensor file format.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TensorFileError {
    /// Bad magic.
    #[error("bad magic")]
    Magic,
    /// Unsupported dtype.
    #[error("unsupported dtype {0}")]
    Dtype(u8),
    /// Bad rank.
    #[error("ndim {0} out of range 1..=4")]
    Ndim(u8),
    /// Too many elements or overflow.
    #[error("tensor too large: {0} elements (max {1})")]
    TooLarge(u64, u64),
    /// Truncated or oversized payload.
    #[error("size mismatch: expected {expected} bytes, got {actual}")]
    Size {
        /// Expected size.
        expected: u64,
        /// Actual size.
        actual: u64,
    },
    /// Non-finite value at index.
    #[error("non-finite value at index {0}")]
    NonFinite(usize),
    /// Bad bundle entry name.
    #[error("bad bundle entry name: {0}")]
    BadName(String),
    /// Shape mismatch.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    Shape {
        /// Expected shape.
        expected: Vec<u64>,
        /// Actual shape.
        actual: Vec<u64>,
    },
}

/// Limits applied when decoding.
#[derive(Clone, Copy, Debug)]
pub struct TensorLimits {
    /// Maximum number of elements.
    pub max_elements: u64,
}

impl Default for TensorLimits {
    fn default() -> Self {
        Self {
            max_elements: DEFAULT_MAX_ELEMENTS,
        }
    }
}

/// A decoded f32 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorF32 {
    /// Shape.
    pub shape: Vec<u64>,
    /// Row-major data.
    pub data: Vec<f32>,
}

impl TensorF32 {
    /// Build, checking that the shape matches the data length.
    pub fn new(shape: Vec<u64>, data: Vec<f32>) -> Result<Self, TensorFileError> {
        let n = num_elements(&shape, u64::MAX)?;
        if n != data.len() as u64 {
            return Err(TensorFileError::Size {
                expected: n,
                actual: data.len() as u64,
            });
        }
        Ok(Self { shape, data })
    }

    /// Encode to bytes (f32).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(6 + 8 * self.shape.len() + 4 * self.data.len());
        self.encode_into(&mut out);
        out
    }

    fn header_into(&self, out: &mut Vec<u8>, dtype: Dtype) {
        out.extend_from_slice(MAGIC);
        out.push(dtype.code());
        out.push(self.shape.len() as u8);
        for d in &self.shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        self.header_into(out, Dtype::F32);
        for v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Encode to bytes as `dtype`. bf16 rounds each element to nearest; an element that is not
    /// finite, or overflows once rounded, is refused.
    pub fn encode_as(&self, dtype: Dtype) -> Result<Vec<u8>, TensorFileError> {
        let mut out = Vec::with_capacity(Self::encoded_size_as(&self.shape, dtype) as usize);
        self.encode_as_into(&mut out, dtype)?;
        Ok(out)
    }

    fn encode_as_into(&self, out: &mut Vec<u8>, dtype: Dtype) -> Result<(), TensorFileError> {
        match dtype {
            Dtype::F32 => self.encode_into(out),
            Dtype::Bf16 => {
                self.header_into(out, dtype);
                for (i, v) in self.data.iter().enumerate() {
                    let b = bf16_bits(*v);
                    if !v.is_finite() || !bf16_value(b).is_finite() {
                        return Err(TensorFileError::NonFinite(i));
                    }
                    out.extend_from_slice(&b.to_le_bytes());
                }
            }
        }
        Ok(())
    }

    /// Decode an f32 file with limits; the whole buffer must be consumed exactly.
    pub fn decode(bytes: &[u8], limits: &TensorLimits) -> Result<Self, TensorFileError> {
        Self::decode_as(bytes, limits, Dtype::F32)
    }

    /// Decode a file of element type `dtype` (any other is refused) into f32.
    pub fn decode_as(
        bytes: &[u8],
        limits: &TensorLimits,
        dtype: Dtype,
    ) -> Result<Self, TensorFileError> {
        if bytes.len() < 6 || &bytes[..4] != MAGIC {
            return Err(TensorFileError::Magic);
        }
        if bytes[4] != dtype.code() {
            return Err(TensorFileError::Dtype(bytes[4]));
        }
        let ndim = bytes[5];
        if ndim == 0 || ndim as usize > MAX_NDIM {
            return Err(TensorFileError::Ndim(ndim));
        }
        let header = 6 + 8 * ndim as usize;
        if bytes.len() < header {
            return Err(TensorFileError::Size {
                expected: header as u64,
                actual: bytes.len() as u64,
            });
        }
        let mut shape = Vec::with_capacity(ndim as usize);
        for i in 0..ndim as usize {
            let mut d = [0u8; 8];
            d.copy_from_slice(&bytes[6 + 8 * i..14 + 8 * i]);
            shape.push(u64::from_le_bytes(d));
        }
        let n = num_elements(&shape, limits.max_elements)?;
        let expected = header as u64 + n * dtype.width();
        if bytes.len() as u64 != expected {
            return Err(TensorFileError::Size {
                expected,
                actual: bytes.len() as u64,
            });
        }
        let mut data = Vec::with_capacity(n as usize);
        match dtype {
            Dtype::F32 => {
                for chunk in bytes[header..].chunks_exact(4) {
                    data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            Dtype::Bf16 => {
                for chunk in bytes[header..].chunks_exact(2) {
                    data.push(bf16_value(u16::from_le_bytes([chunk[0], chunk[1]])));
                }
            }
        }
        Ok(Self { shape, data })
    }

    /// Shape and payload size announced by a header, without decoding the data. Used to check a
    /// large dataset before reading it whole.
    pub fn peek_shape(
        bytes: &[u8],
        limits: &TensorLimits,
    ) -> Result<(Vec<u64>, u64), TensorFileError> {
        if bytes.len() < 6 || &bytes[..4] != MAGIC {
            return Err(TensorFileError::Magic);
        }
        if bytes[4] != DTYPE_F32 {
            return Err(TensorFileError::Dtype(bytes[4]));
        }
        let ndim = bytes[5];
        if ndim == 0 || ndim as usize > MAX_NDIM {
            return Err(TensorFileError::Ndim(ndim));
        }
        let header = 6 + 8 * ndim as usize;
        if bytes.len() < header {
            return Err(TensorFileError::Size {
                expected: header as u64,
                actual: bytes.len() as u64,
            });
        }
        let mut shape = Vec::with_capacity(ndim as usize);
        for i in 0..ndim as usize {
            let mut d = [0u8; 8];
            d.copy_from_slice(&bytes[6 + 8 * i..14 + 8 * i]);
            shape.push(u64::from_le_bytes(d));
        }
        let n = num_elements(&shape, limits.max_elements)?;
        Ok((shape, header as u64 + n * 4))
    }

    /// Exact encoded size of an f32 tensor of this shape.
    pub fn encoded_size(shape: &[u64]) -> u64 {
        Self::encoded_size_as(shape, Dtype::F32)
    }

    /// Exact encoded size of a tensor of this shape and element type.
    pub fn encoded_size_as(shape: &[u64], dtype: Dtype) -> u64 {
        6 + 8 * shape.len() as u64 + dtype.width() * shape.iter().product::<u64>()
    }

    /// Read a file, refusing to read more than the size implied by `limits`.
    pub fn read_from_path(path: &Path, limits: &TensorLimits) -> Result<Self, TensorFileError> {
        let max_bytes = 6 + 8 * MAX_NDIM as u64 + limits.max_elements.saturating_mul(4);
        let meta = std::fs::metadata(path).map_err(|_| TensorFileError::Magic)?;
        if meta.len() > max_bytes {
            return Err(TensorFileError::TooLarge(
                meta.len() / 4,
                limits.max_elements,
            ));
        }
        let mut buf = Vec::with_capacity(meta.len() as usize);
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .map_err(|_| TensorFileError::Magic)?;
        Self::decode(&buf, limits)
    }

    /// Fail if any element is NaN or infinite.
    pub fn check_finite(&self) -> Result<(), TensorFileError> {
        match self.data.iter().position(|v| !v.is_finite()) {
            Some(i) => Err(TensorFileError::NonFinite(i)),
            None => Ok(()),
        }
    }

    /// Fail unless the shape is exactly `expected`.
    pub fn check_shape(&self, expected: &[u64]) -> Result<(), TensorFileError> {
        if self.shape.as_slice() != expected {
            return Err(TensorFileError::Shape {
                expected: expected.to_vec(),
                actual: self.shape.clone(),
            });
        }
        Ok(())
    }
}

/// Magic bytes of a bundle.
pub const BUNDLE_MAGIC: &[u8; 4] = b"GGB1";
/// Maximum entries in a bundle.
pub const MAX_BUNDLE_ENTRIES: usize = 4096;
/// Maximum entry name length.
pub const MAX_BUNDLE_NAME: usize = 64;

/// `GGB1` bundle: several named `GGT1` tensors in one artifact, so a checkpoint of hundreds of
/// parameters is one reference, one hash and one download.
///
/// ```text
/// magic "GGB1" | u32 LE count | count × ( u16 LE name_len | name | u64 LE tensor_len ) | count × GGT1
/// ```
///
/// Entries are sorted by name, so the encoding of a given set of tensors is unique.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BundleF32 {
    /// Tensors by name.
    pub entries: std::collections::BTreeMap<String, TensorF32>,
}

impl BundleF32 {
    /// Build from named tensors.
    pub fn new(
        entries: std::collections::BTreeMap<String, TensorF32>,
    ) -> Result<Self, TensorFileError> {
        if entries.len() > MAX_BUNDLE_ENTRIES {
            return Err(TensorFileError::TooLarge(
                entries.len() as u64,
                MAX_BUNDLE_ENTRIES as u64,
            ));
        }
        for name in entries.keys() {
            if name.is_empty() || name.len() > MAX_BUNDLE_NAME {
                return Err(TensorFileError::BadName(name.clone()));
            }
        }
        Ok(Self { entries })
    }

    /// Encode to bytes (f32).
    pub fn encode(&self) -> Vec<u8> {
        self.encode_as(Dtype::F32)
            .expect("f32 encoding never fails")
    }

    /// Encode to bytes with every tensor as `dtype` (see [`TensorF32::encode_as`]). Written in
    /// one buffer of the exact size: a SmolLM2 gradient is 270 MB in bf16.
    pub fn encode_as(&self, dtype: Dtype) -> Result<Vec<u8>, TensorFileError> {
        Self::encode_entries_as(&self.entries, dtype)
    }

    /// Encode named tensors as a bundle without taking or copying them (a server step encodes
    /// 1.6 GB of weights and optimizer state per round for SmolLM2); same checks as [`Self::new`].
    pub fn encode_entries_as(
        entries: &std::collections::BTreeMap<String, TensorF32>,
        dtype: Dtype,
    ) -> Result<Vec<u8>, TensorFileError> {
        if entries.len() > MAX_BUNDLE_ENTRIES {
            return Err(TensorFileError::TooLarge(
                entries.len() as u64,
                MAX_BUNDLE_ENTRIES as u64,
            ));
        }
        if let Some(name) = entries
            .keys()
            .find(|n| n.is_empty() || n.len() > MAX_BUNDLE_NAME)
        {
            return Err(TensorFileError::BadName(name.clone()));
        }
        let size = Self::encoded_size_as(
            entries
                .iter()
                .map(|(n, t)| (n.as_str(), t.shape.as_slice())),
            dtype,
        );
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(BUNDLE_MAGIC);
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (name, t) in entries {
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&TensorF32::encoded_size_as(&t.shape, dtype).to_le_bytes());
        }
        for t in entries.values() {
            if dtype == Dtype::F32 {
                t.encode_into(&mut out);
            } else {
                t.encode_as_into(&mut out, dtype)?;
            }
        }
        debug_assert_eq!(out.len() as u64, size);
        Ok(out)
    }

    /// Exact encoded size of an f32 bundle of named shapes.
    pub fn encoded_size<'a>(shapes: impl IntoIterator<Item = (&'a str, &'a [u64])>) -> u64 {
        Self::encoded_size_as(shapes, Dtype::F32)
    }

    /// Exact encoded size of a bundle of named shapes whose tensors are `dtype`.
    pub fn encoded_size_as<'a>(
        shapes: impl IntoIterator<Item = (&'a str, &'a [u64])>,
        dtype: Dtype,
    ) -> u64 {
        let mut total = 8u64;
        for (name, shape) in shapes {
            total += 2 + name.len() as u64 + 8 + TensorF32::encoded_size_as(shape, dtype);
        }
        total
    }

    /// Decode an f32 bundle with limits applied to every tensor and to the total element count;
    /// the whole buffer must be consumed exactly and names must be sorted and unique.
    pub fn decode(bytes: &[u8], limits: &TensorLimits) -> Result<Self, TensorFileError> {
        Self::decode_as(bytes, limits, Dtype::F32)
    }

    /// Same, for a bundle whose tensors are all `dtype`.
    pub fn decode_as(
        bytes: &[u8],
        limits: &TensorLimits,
        dtype: Dtype,
    ) -> Result<Self, TensorFileError> {
        if bytes.len() < 8 || &bytes[..4] != BUNDLE_MAGIC {
            return Err(TensorFileError::Magic);
        }
        let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        if count > MAX_BUNDLE_ENTRIES {
            return Err(TensorFileError::TooLarge(
                count as u64,
                MAX_BUNDLE_ENTRIES as u64,
            ));
        }
        let mut pos = 8usize;
        let mut index: Vec<(String, u64)> = Vec::with_capacity(count);
        let short = |pos: usize| TensorFileError::Size {
            expected: pos as u64,
            actual: bytes.len() as u64,
        };
        for _ in 0..count {
            if bytes.len() < pos + 2 {
                return Err(short(pos + 2));
            }
            let name_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;
            if name_len == 0 || name_len > MAX_BUNDLE_NAME {
                return Err(TensorFileError::BadName(format!("length {name_len}")));
            }
            if bytes.len() < pos + name_len + 8 {
                return Err(short(pos + name_len + 8));
            }
            let name = std::str::from_utf8(&bytes[pos..pos + name_len])
                .map_err(|_| TensorFileError::BadName("not utf-8".into()))?
                .to_owned();
            pos += name_len;
            let mut l = [0u8; 8];
            l.copy_from_slice(&bytes[pos..pos + 8]);
            pos += 8;
            if let Some((prev, _)) = index.last()
                && prev.as_str() >= name.as_str()
            {
                return Err(TensorFileError::BadName(format!(
                    "'{name}' out of order after '{prev}'"
                )));
            }
            index.push((name, u64::from_le_bytes(l)));
        }
        let mut entries = std::collections::BTreeMap::new();
        let mut remaining_elements = limits.max_elements;
        for (name, len) in index {
            let len = usize::try_from(len).map_err(|_| short(usize::MAX))?;
            if bytes.len() < pos + len {
                return Err(short(pos + len));
            }
            let t = TensorF32::decode_as(
                &bytes[pos..pos + len],
                &TensorLimits {
                    max_elements: remaining_elements,
                },
                dtype,
            )?;
            remaining_elements = remaining_elements.saturating_sub(t.data.len() as u64);
            pos += len;
            entries.insert(name, t);
        }
        if pos != bytes.len() {
            return Err(TensorFileError::Size {
                expected: pos as u64,
                actual: bytes.len() as u64,
            });
        }
        Ok(Self { entries })
    }

    /// Read an f32 bundle file, refusing to read more than `limits` allows.
    pub fn read_from_path(path: &Path, limits: &TensorLimits) -> Result<Self, TensorFileError> {
        Self::read_from_path_as(path, limits, Dtype::F32)
    }

    /// Same, for a bundle whose tensors are all `dtype`.
    pub fn read_from_path_as(
        path: &Path,
        limits: &TensorLimits,
        dtype: Dtype,
    ) -> Result<Self, TensorFileError> {
        let max_bytes = 8
            + MAX_BUNDLE_ENTRIES as u64
                * (2 + MAX_BUNDLE_NAME as u64 + 8 + 6 + 8 * MAX_NDIM as u64)
            + limits.max_elements.saturating_mul(4);
        let meta = std::fs::metadata(path).map_err(|_| TensorFileError::Magic)?;
        if meta.len() > max_bytes {
            return Err(TensorFileError::TooLarge(
                meta.len() / 4,
                limits.max_elements,
            ));
        }
        let mut buf = Vec::with_capacity(meta.len() as usize);
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .map_err(|_| TensorFileError::Magic)?;
        Self::decode_as(&buf, limits, dtype)
    }

    /// Fail if any tensor holds a NaN or an infinity.
    pub fn check_finite(&self) -> Result<(), TensorFileError> {
        for t in self.entries.values() {
            t.check_finite()?;
        }
        Ok(())
    }
}

fn num_elements(shape: &[u64], max: u64) -> Result<u64, TensorFileError> {
    let mut n: u64 = 1;
    for d in shape {
        if *d == 0 {
            return Err(TensorFileError::TooLarge(0, max));
        }
        n = n
            .checked_mul(*d)
            .ok_or(TensorFileError::TooLarge(u64::MAX, max))?;
        if n > max {
            return Err(TensorFileError::TooLarge(n, max));
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_roundtrip_layout_and_bounds() {
        let mut m = std::collections::BTreeMap::new();
        m.insert(
            "b".to_string(),
            TensorF32::new(vec![2], vec![1.0, 2.0]).unwrap(),
        );
        m.insert(
            "a".to_string(),
            TensorF32::new(vec![1, 2], vec![3.0, 4.0]).unwrap(),
        );
        let bundle = BundleF32::new(m).unwrap();
        let bytes = bundle.encode();
        assert_eq!(&bytes[..4], b"GGB1");
        assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
        // Index: 'a' before 'b' whatever the insertion order.
        assert_eq!(&bytes[8..10], &1u16.to_le_bytes());
        assert_eq!(&bytes[10..11], b"a");
        let expected = BundleF32::encoded_size([("a", &[1u64, 2][..]), ("b", &[2u64][..])]);
        assert_eq!(bytes.len() as u64, expected);
        assert_eq!(
            BundleF32::decode(&bytes, &TensorLimits::default()).unwrap(),
            bundle
        );
        // One byte too many, one byte too few, an unordered index, a limit set too low: all rejected.
        let mut plus = bytes.clone();
        plus.push(0);
        assert!(matches!(
            BundleF32::decode(&plus, &TensorLimits::default()),
            Err(TensorFileError::Size { .. })
        ));
        assert!(matches!(
            BundleF32::decode(&bytes[..bytes.len() - 1], &TensorLimits::default()),
            Err(TensorFileError::Size { .. })
        ));
        let mut swapped = bytes.clone();
        swapped[10] = b'c';
        assert!(matches!(
            BundleF32::decode(&swapped, &TensorLimits::default()),
            Err(TensorFileError::BadName(_))
        ));
        assert!(matches!(
            BundleF32::decode(&bytes, &TensorLimits { max_elements: 3 }),
            Err(TensorFileError::TooLarge(..))
        ));
        let (shape, size) =
            TensorF32::peek_shape(&bundle.entries["a"].encode(), &TensorLimits::default()).unwrap();
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(size, TensorF32::encoded_size(&[1, 2]));
    }

    #[test]
    fn roundtrip_and_layout() {
        let t = TensorF32::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = t.encode();
        assert_eq!(&b[..4], b"GGT1");
        assert_eq!(b[4], 1);
        assert_eq!(b[5], 2);
        assert_eq!(b.len(), 6 + 16 + 24);
        assert_eq!(TensorF32::decode(&b, &TensorLimits::default()).unwrap(), t);
        assert_eq!(&b[6..14], &2u64.to_le_bytes());
        assert_eq!(&b[22..26], &1.0f32.to_le_bytes());
    }

    #[test]
    fn rejects_bad_inputs() {
        let t = TensorF32::new(vec![2, 2], vec![0.0; 4]).unwrap();
        let mut b = t.encode();
        assert!(matches!(
            TensorF32::decode(&b[..b.len() - 1], &TensorLimits::default()),
            Err(TensorFileError::Size { .. })
        ));
        b.push(0);
        assert!(matches!(
            TensorF32::decode(&b, &TensorLimits::default()),
            Err(TensorFileError::Size { .. })
        ));
        let mut bad = t.encode();
        bad[0] = b'X';
        assert_eq!(
            TensorF32::decode(&bad, &TensorLimits::default()),
            Err(TensorFileError::Magic)
        );
        let mut bad = t.encode();
        bad[4] = 7;
        assert_eq!(
            TensorF32::decode(&bad, &TensorLimits::default()),
            Err(TensorFileError::Dtype(7))
        );
        let mut bad = t.encode();
        bad[5] = 5;
        assert_eq!(
            TensorF32::decode(&bad, &TensorLimits::default()),
            Err(TensorFileError::Ndim(5))
        );
        // huge declared shape must not allocate
        let mut huge = Vec::new();
        huge.extend_from_slice(MAGIC);
        huge.push(1);
        huge.push(2);
        huge.extend_from_slice(&u64::MAX.to_le_bytes());
        huge.extend_from_slice(&2u64.to_le_bytes());
        assert!(matches!(
            TensorF32::decode(&huge, &TensorLimits::default()),
            Err(TensorFileError::TooLarge(..))
        ));
        // element limit
        let big = TensorF32::new(vec![10, 10], vec![0.0; 100])
            .unwrap()
            .encode();
        assert!(matches!(
            TensorF32::decode(&big, &TensorLimits { max_elements: 99 }),
            Err(TensorFileError::TooLarge(..))
        ));
        // zero dimension
        let mut zero = Vec::new();
        zero.extend_from_slice(MAGIC);
        zero.push(1);
        zero.push(1);
        zero.extend_from_slice(&0u64.to_le_bytes());
        assert!(TensorF32::decode(&zero, &TensorLimits::default()).is_err());
    }

    #[test]
    fn bf16_rounds_to_nearest_even_and_is_typed() {
        // Exactly representable values pass unchanged; others round to nearest, ties to even.
        assert_eq!(bf16_value(bf16_bits(1.0)), 1.0);
        assert_eq!(bf16_value(bf16_bits(-3.5)), -3.5);
        let one_plus_half_ulp = f32::from_bits(0x3F80_8000); // halfway between 1 and 1 + 2^-7
        assert_eq!(bf16_bits(one_plus_half_ulp), 0x3F80); // tie → even (1.0)
        let above = f32::from_bits(0x3F80_8001);
        assert_eq!(bf16_bits(above), 0x3F81);
        let three_halves_tie = f32::from_bits(0x3F81_8000); // tie between odd 0x3F81 and 0x3F82
        assert_eq!(bf16_bits(three_halves_tie), 0x3F82);
        // Relative error at most 2^-9 for normal values.
        for x in [0.1f32, 1e-20, 12345.678, -7.77e-3, 3.0e38] {
            let r = bf16_value(bf16_bits(x));
            assert!(((r - x) / x).abs() <= 1.0 / 512.0, "{x} → {r}");
        }

        let t = TensorF32::new(vec![2, 2], vec![1.0, -0.1, 3.0e-5, 65504.0]).unwrap();
        let b = t.encode_as(Dtype::Bf16).unwrap();
        assert_eq!(b[4], DTYPE_BF16);
        assert_eq!(
            b.len() as u64,
            TensorF32::encoded_size_as(&[2, 2], Dtype::Bf16)
        );
        assert_eq!(b.len(), 6 + 16 + 8);
        let back = TensorF32::decode_as(&b, &TensorLimits::default(), Dtype::Bf16).unwrap();
        for (x, y) in t.data.iter().zip(&back.data) {
            assert_eq!(*y, bf16_value(bf16_bits(*x)));
        }
        // The reader states the type: a bf16 file is not an f32 one, nor the other way round.
        assert_eq!(
            TensorF32::decode(&b, &TensorLimits::default()),
            Err(TensorFileError::Dtype(DTYPE_BF16))
        );
        assert_eq!(
            TensorF32::decode_as(&t.encode(), &TensorLimits::default(), Dtype::Bf16),
            Err(TensorFileError::Dtype(DTYPE_F32))
        );
        assert!(matches!(
            TensorF32::decode_as(&b[..b.len() - 1], &TensorLimits::default(), Dtype::Bf16),
            Err(TensorFileError::Size { .. })
        ));
        // Not finite, or overflowing once rounded: refused.
        let nan = TensorF32::new(vec![2], vec![1.0, f32::NAN]).unwrap();
        assert_eq!(
            nan.encode_as(Dtype::Bf16),
            Err(TensorFileError::NonFinite(1))
        );
        let huge = TensorF32::new(vec![1], vec![f32::MAX]).unwrap();
        assert_eq!(
            huge.encode_as(Dtype::Bf16),
            Err(TensorFileError::NonFinite(0))
        );

        // A bundle: exact size, same index layout, typed decoding.
        let mut m = std::collections::BTreeMap::new();
        m.insert("w".to_string(), t.clone());
        m.insert(
            "b".to_string(),
            TensorF32::new(vec![3], vec![0.5, 0.25, -2.0]).unwrap(),
        );
        let bundle = BundleF32::new(m).unwrap();
        let bytes = bundle.encode_as(Dtype::Bf16).unwrap();
        let shapes = [("b", &[3u64][..]), ("w", &[2u64, 2][..])];
        assert_eq!(
            bytes.len() as u64,
            BundleF32::encoded_size_as(shapes, Dtype::Bf16)
        );
        assert!(bytes.len() < bundle.encode().len());
        let decoded = BundleF32::decode_as(&bytes, &TensorLimits::default(), Dtype::Bf16).unwrap();
        assert_eq!(decoded.entries["b"], bundle.entries["b"]);
        assert_eq!(decoded.entries["w"], back);
        assert_eq!(
            BundleF32::decode(&bytes, &TensorLimits::default()),
            Err(TensorFileError::Dtype(DTYPE_BF16))
        );
        assert_eq!(bundle.encode_as(Dtype::F32).unwrap(), bundle.encode());
    }

    #[test]
    fn finite_and_shape_checks() {
        let t = TensorF32::new(vec![2], vec![1.0, f32::NAN]).unwrap();
        assert_eq!(t.check_finite(), Err(TensorFileError::NonFinite(1)));
        assert!(t.check_shape(&[2]).is_ok());
        assert!(t.check_shape(&[1, 2]).is_err());
    }
}
