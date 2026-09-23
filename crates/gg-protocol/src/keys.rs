//! Ed25519 identities: secret keys (never serialized in messages), public keys and signatures.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use zeroize::Zeroizing;

use crate::ProtocolError;

/// An Ed25519 public key (32 bytes). Hex in human-readable formats, raw bytes otherwise.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PublicKey(pub [u8; 32]);

/// An Ed25519 signature (64 bytes).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Signature(pub [u8; 64]);

/// An Ed25519 secret key. `Debug` is redacted; there is no `Serialize`.
pub struct SecretKey(SigningKey);

impl SecretKey {
    /// Generate a fresh key from the operating system CSPRNG.
    pub fn generate() -> Result<Self, ProtocolError> {
        let mut secret = Zeroizing::new([0u8; 32]);
        getrandom::fill(secret.as_mut()).map_err(|e| ProtocolError::KeyMaterial(e.to_string()))?;
        Ok(Self(SigningKey::from_bytes(&secret)))
    }

    /// Build from 32 secret bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        Self(SigningKey::from_bytes(bytes))
    }

    /// Secret bytes (zeroized on drop).
    pub fn to_bytes(&self) -> Zeroizing<[u8; 32]> {
        Zeroizing::new(self.0.to_bytes())
    }

    /// Corresponding public key.
    pub fn public_key(&self) -> PublicKey {
        PublicKey(self.0.verifying_key().to_bytes())
    }

    /// Sign arbitrary bytes (callers must pass domain-tagged bytes; see [`crate::signed`]).
    pub fn sign_raw(&self, msg: &[u8]) -> Signature {
        Signature(self.0.sign(msg).to_bytes())
    }

    /// Load from a key file containing 64 hex characters (created by [`SecretKey::save_to_file`]).
    pub fn load_from_file(path: &Path) -> Result<Self, ProtocolError> {
        let text = Zeroizing::new(std::fs::read_to_string(path)?);
        let trimmed = text.trim();
        if trimmed.len() != 64 {
            return Err(ProtocolError::KeyMaterial(
                "key file must contain 64 hex characters".into(),
            ));
        }
        let raw = Zeroizing::new(
            hex::decode(trimmed)
                .map_err(|_| ProtocolError::KeyMaterial("key file is not hex".into()))?,
        );
        let mut bytes = Zeroizing::new([0u8; 32]);
        bytes.copy_from_slice(&raw);
        Ok(Self::from_bytes(&bytes))
    }

    /// Save to a new file with mode 0600 (fails if the file exists).
    pub fn save_to_file(&self, path: &Path) -> Result<(), ProtocolError> {
        use std::io::Write;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(path)?;
        let hex = Zeroizing::new(hex::encode(self.to_bytes().as_ref()));
        f.write_all(hex.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
        Ok(())
    }

    /// Load the key if the file exists, otherwise generate and save it.
    pub fn load_or_generate(path: &Path) -> Result<Self, ProtocolError> {
        if path.exists() {
            Self::load_from_file(path)
        } else {
            let k = Self::generate()?;
            k.save_to_file(path)?;
            Ok(k)
        }
    }
}

impl fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretKey(<redacted>, pub={})", self.public_key())
    }
}

impl PublicKey {
    /// Verify a signature over raw (already tagged) bytes using `verify_strict`.
    pub fn verify_raw(&self, msg: &[u8], sig: &Signature) -> Result<(), ProtocolError> {
        let vk = VerifyingKey::from_bytes(&self.0)
            .map_err(|_| ProtocolError::KeyMaterial("invalid public key".into()))?;
        let sig = ed25519_dalek::Signature::from_bytes(&sig.0);
        vk.verify_strict(msg, &sig)
            .map_err(|_| ProtocolError::InvalidSignature)
    }

    /// Lowercase hex.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PublicKey({})", self.to_hex())
    }
}

impl FromStr for PublicKey {
    type Err = ProtocolError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = hex::decode(s)
            .map_err(|_| ProtocolError::KeyMaterial("public key is not hex".into()))?;
        let arr: [u8; 32] = v
            .try_into()
            .map_err(|_| ProtocolError::KeyMaterial("public key must be 32 bytes".into()))?;
        Ok(Self(arr))
    }
}

impl Signature {
    /// Lowercase hex.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signature({})", self.to_hex())
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl FromStr for Signature {
    type Err = ProtocolError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = hex::decode(s)
            .map_err(|_| ProtocolError::KeyMaterial("signature is not hex".into()))?;
        let arr: [u8; 64] = v
            .try_into()
            .map_err(|_| ProtocolError::KeyMaterial("signature must be 64 bytes".into()))?;
        Ok(Self(arr))
    }
}

fn ser_fixed<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
    if s.is_human_readable() {
        s.serialize_str(&hex::encode(bytes))
    } else {
        s.serialize_bytes(bytes)
    }
}

fn de_fixed<'de, D: Deserializer<'de>, const N: usize>(d: D) -> Result<[u8; N], D::Error> {
    let v: Vec<u8> = if d.is_human_readable() {
        let s = String::deserialize(d)?;
        hex::decode(s).map_err(serde::de::Error::custom)?
    } else {
        serde_bytes::ByteBuf::deserialize(d)?.into_vec()
    };
    v.try_into()
        .map_err(|_| serde::de::Error::custom(format!("expected {N} bytes")))
}

impl Serialize for PublicKey {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        ser_fixed(&self.0, s)
    }
}

impl<'de> Deserialize<'de> for PublicKey {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        de_fixed::<D, 32>(d).map(PublicKey)
    }
}

impl Serialize for Signature {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        ser_fixed(&self.0, s)
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        de_fixed::<D, 64>(d).map(Signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc8032_test_vector_1() {
        // RFC 8032 §7.1 TEST 1
        let secret: [u8; 32] =
            hex::decode("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
                .unwrap()
                .try_into()
                .unwrap();
        let sk = SecretKey::from_bytes(&secret);
        assert_eq!(
            sk.public_key().to_hex(),
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        );
        let sig = sk.sign_raw(b"");
        assert_eq!(
            sig.to_hex(),
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        );
        sk.public_key().verify_raw(b"", &sig).unwrap();
        assert!(sk.public_key().verify_raw(b"x", &sig).is_err());
    }

    #[test]
    fn key_file_roundtrip_is_0600() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.key");
        let k = SecretKey::generate().unwrap();
        k.save_to_file(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let k2 = SecretKey::load_from_file(&path).unwrap();
        assert_eq!(k.public_key(), k2.public_key());
        assert!(k.save_to_file(&path).is_err(), "must not overwrite");
        let k3 = SecretKey::load_or_generate(&path).unwrap();
        assert_eq!(k3.public_key(), k.public_key());
    }

    #[test]
    fn json_hex_forms() {
        let k = SecretKey::generate().unwrap();
        let pk = k.public_key();
        let j = serde_json::to_string(&pk).unwrap();
        assert_eq!(j.len(), 66);
        assert_eq!(serde_json::from_str::<PublicKey>(&j).unwrap(), pk);
        let cbor = crate::canonical::to_bytes(&pk).unwrap();
        assert_eq!(cbor.len(), 34); // 0x58 0x20 + 32 bytes
        assert_eq!(
            crate::canonical::from_bytes::<PublicKey>(&cbor).unwrap(),
            pk
        );
        assert!(format!("{k:?}").contains("redacted"));
    }
}
