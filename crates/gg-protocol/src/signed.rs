//! Signed envelope binding a body to a signer, a domain, a network and the protocol version.

use gg_core::{NetworkId, Sha256Digest};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::ProtocolError;
use crate::canonical;
use crate::domain::{self, Domain};
use crate::keys::{PublicKey, SecretKey, Signature};

/// A message with its signer and signature. The signature covers the domain-tagged canonical
/// encoding of `body` (see docs/PROTOCOL.md).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Signed<T> {
    /// Message body.
    pub body: T,
    /// Signer public key. Cryptographic validity is not authorization.
    pub signer: PublicKey,
    /// Ed25519 signature over `tagged(network, domain, canonical(body))`.
    pub signature: Signature,
}

impl<T: Serialize> Signed<T> {
    /// Sign `body` for `network` in `domain`.
    pub fn sign(
        key: &SecretKey,
        network: &NetworkId,
        domain: Domain,
        body: T,
    ) -> Result<Self, ProtocolError> {
        let bytes = canonical::to_bytes(&body)?;
        let msg = domain::tagged(network, domain, &bytes);
        Ok(Self {
            body,
            signer: key.public_key(),
            signature: key.sign_raw(&msg),
        })
    }

    /// Verify the signature for `network`/`domain` and return the body hash.
    pub fn verify(
        &self,
        network: &NetworkId,
        domain: Domain,
    ) -> Result<Sha256Digest, ProtocolError> {
        let bytes = canonical::to_bytes(&self.body)?;
        let msg = domain::tagged(network, domain, &bytes);
        self.signer.verify_raw(&msg, &self.signature)?;
        Ok(Sha256Digest::of_bytes(&msg))
    }

    /// Domain-separated hash of the body (without verifying).
    pub fn body_hash(
        &self,
        network: &NetworkId,
        domain: Domain,
    ) -> Result<Sha256Digest, ProtocolError> {
        domain::hash_message(network, domain, &self.body)
    }

    /// Canonical bytes of the whole envelope (for storage and export).
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        canonical::to_bytes(self)
    }
}

impl<T: DeserializeOwned> Signed<T> {
    /// Parse an envelope from canonical bytes (strict).
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ProtocolError> {
        canonical::from_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
    struct Body {
        a: u32,
        b: String,
    }

    #[test]
    fn sign_verify_and_domain_binding() {
        let key = SecretKey::generate().unwrap();
        let net = NetworkId::new("gg-dev-local").unwrap();
        let s = Signed::sign(
            &key,
            &net,
            Domain::Result,
            Body {
                a: 1,
                b: "x".into(),
            },
        )
        .unwrap();
        s.verify(&net, Domain::Result).unwrap();
        assert!(s.verify(&net, Domain::Receipt).is_err(), "domain binding");
        assert!(
            s.verify(&NetworkId::new("gg-other").unwrap(), Domain::Result)
                .is_err(),
            "network binding"
        );
        let mut tampered = s.clone();
        tampered.body.a = 2;
        assert!(tampered.verify(&net, Domain::Result).is_err());
        let mut wrong_signer = s.clone();
        wrong_signer.signer = SecretKey::generate().unwrap().public_key();
        assert!(wrong_signer.verify(&net, Domain::Result).is_err());
        let bytes = s.to_canonical_bytes().unwrap();
        let back: Signed<Body> = Signed::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(back, s);
        back.verify(&net, Domain::Result).unwrap();
    }
}
