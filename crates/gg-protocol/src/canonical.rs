//! Deterministic encoding: DAG-CBOR via `serde_ipld_dagcbor` (strict decoding enabled).

use serde::{Serialize, de::DeserializeOwned};

use crate::ProtocolError;

/// Encode a value into canonical DAG-CBOR bytes.
pub fn to_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ProtocolError> {
    serde_ipld_dagcbor::to_vec(value).map_err(|e| ProtocolError::Encode(e.to_string()))
}

/// Decode a value from DAG-CBOR bytes. Non-canonical input (unsorted keys, non-minimal integers,
/// indefinite lengths, non-finite floats) is rejected by the strict decoder.
pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ProtocolError> {
    serde_ipld_dagcbor::from_slice(bytes).map_err(|e| ProtocolError::Decode(e.to_string()))
}

/// Check that `bytes` is exactly the canonical encoding of the value it decodes to.
pub fn is_canonical<T: DeserializeOwned + Serialize>(bytes: &[u8]) -> Result<bool, ProtocolError> {
    let v: T = from_bytes(bytes)?;
    Ok(to_bytes(&v)? == bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[test]
    fn rfc8949_integer_vectors() {
        // RFC 8949 Appendix A
        assert_eq!(to_bytes(&0u64).unwrap(), hex::decode("00").unwrap());
        assert_eq!(to_bytes(&23u64).unwrap(), hex::decode("17").unwrap());
        assert_eq!(to_bytes(&24u64).unwrap(), hex::decode("1818").unwrap());
        assert_eq!(to_bytes(&1000u64).unwrap(), hex::decode("1903e8").unwrap());
        assert_eq!(
            to_bytes(&1_000_000u64).unwrap(),
            hex::decode("1a000f4240").unwrap()
        );
        assert_eq!(
            to_bytes(&1_000_000_000_000u64).unwrap(),
            hex::decode("1b000000e8d4a51000").unwrap()
        );
        assert_eq!(
            to_bytes(&u64::MAX).unwrap(),
            hex::decode("1bffffffffffffffff").unwrap()
        );
        assert_eq!(to_bytes(&(-1i64)).unwrap(), hex::decode("20").unwrap());
        assert_eq!(
            to_bytes(&(-1000i64)).unwrap(),
            hex::decode("3903e7").unwrap()
        );
        assert_eq!(to_bytes(&"a").unwrap(), hex::decode("6161").unwrap());
        assert_eq!(to_bytes(&true).unwrap(), hex::decode("f5").unwrap());
        assert_eq!(
            to_bytes(&Option::<u8>::None).unwrap(),
            hex::decode("f6").unwrap()
        );
        // floats are always 64-bit in DAG-CBOR
        assert_eq!(
            to_bytes(&1.1f64).unwrap(),
            hex::decode("fb3ff199999999999a").unwrap()
        );
        assert_eq!(
            to_bytes(&1.0f64).unwrap(),
            hex::decode("fb3ff0000000000000").unwrap()
        );
    }

    #[test]
    fn map_keys_are_sorted_length_first() {
        #[derive(Serialize)]
        struct S {
            zz: u8,
            a: u8,
            bbb: u8,
            b: u8,
        }
        let bytes = to_bytes(&S {
            zz: 1,
            a: 2,
            bbb: 3,
            b: 4,
        })
        .unwrap();
        // a4 61 61 02 | 61 62 04 | 62 7a 7a 01 | 63 62 62 62 03
        assert_eq!(hex::encode(bytes), "a4616102616204627a7a016362626203");
    }

    #[test]
    fn non_finite_floats_are_rejected() {
        assert!(to_bytes(&f64::NAN).is_err());
        assert!(to_bytes(&f64::INFINITY).is_err());
        assert!(to_bytes(&f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn non_canonical_input_is_rejected() {
        // non-minimal integer 1 encoded as 0x18 0x01
        assert!(from_bytes::<u64>(&hex::decode("1801").unwrap()).is_err());
        // unsorted map keys {"b":1,"a":2}
        let unsorted = hex::decode("a2616201616102").unwrap();
        assert!(from_bytes::<BTreeMap<String, u8>>(&unsorted).is_err());
        // indefinite-length array
        assert!(from_bytes::<Vec<u8>>(&hex::decode("9f01ff").unwrap()).is_err());
        // duplicate keys
        assert!(
            from_bytes::<BTreeMap<String, u8>>(&hex::decode("a2616101616102").unwrap()).is_err()
        );
        // trailing bytes
        assert!(from_bytes::<u64>(&hex::decode("0100").unwrap()).is_err());
    }

    #[test]
    fn bytes_roundtrip_as_byte_strings() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct B {
            #[serde(with = "serde_bytes")]
            data: Vec<u8>,
        }
        let b = B {
            data: vec![1, 2, 3],
        };
        let bytes = to_bytes(&b).unwrap();
        assert_eq!(hex::encode(&bytes), "a164646174614301020".to_owned() + "3");
        assert_eq!(from_bytes::<B>(&bytes).unwrap(), b);
    }
}
