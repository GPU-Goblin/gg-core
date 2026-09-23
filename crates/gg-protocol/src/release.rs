//! Signed release manifests of the distributed executables (P8).
//!
//! A contributor runs binaries that talk to a control plane and execute other people's tasks. The
//! only thing that makes those binaries trustworthy is a signature by a key the contributor decided
//! to trust, checked before the files are used. This module is that check, and nothing more:
//!
//! * updates are **manual and verifiable**: nothing here downloads, installs or replaces anything,
//!   and no peer-supplied content is ever executed on the strength of this manifest;
//! * the trust root is a list of keys **the verifier supplies**, not a list carried by the manifest;
//! * a revoked key fails even when its signature is valid, because revocation is a decision about
//!   the key, not about the bytes.

use std::collections::BTreeSet;
use std::path::Path;

use gg_core::{NetworkId, Sha256Digest, UnixMillis};
use serde::{Deserialize, Serialize};

use crate::ProtocolError;
use crate::domain::Domain;
use crate::keys::PublicKey;
use crate::signed::Signed;

/// One distributed file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseArtifactV1 {
    /// File name, without any path: the verifier decides where files live.
    pub name: String,
    /// Size in bytes.
    pub size: u64,
    /// SHA-256 of the bytes.
    pub sha256: Sha256Digest,
}

/// What a release publishes, in one signed object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseManifestV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network this build is meant for.
    pub network: NetworkId,
    /// Version of the workspace.
    pub version: String,
    /// Commit the build came from.
    pub git_commit: String,
    /// Toolchain used.
    pub toolchain: String,
    /// Build time.
    pub built_at: UnixMillis,
    /// Files, sorted by name.
    pub artifacts: Vec<ReleaseArtifactV1>,
    /// Free-text notes (no secrets).
    pub notes: String,
}

/// Why a release was refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ReleaseError {
    /// The signature does not verify for this network and domain.
    #[error("invalid signature: {0}")]
    Signature(String),
    /// The signer is not in the trust root supplied by the verifier.
    #[error("the manifest is signed by a key that is not trusted here")]
    Untrusted,
    /// The signer was revoked.
    #[error("the signing key is revoked")]
    Revoked,
    /// The manifest is structurally wrong.
    #[error("malformed manifest: {0}")]
    Malformed(String),
    /// A file on disk does not match what the manifest says.
    #[error("{name}: {detail}")]
    Mismatch {
        /// File.
        name: String,
        /// What differs.
        detail: String,
    },
    /// A file named by the manifest is missing.
    #[error("{0}: missing")]
    Missing(String),
}

impl ReleaseManifestV1 {
    /// Structural validation: names are plain file names, sorted, unique and bounded.
    pub fn validate_structure(&self) -> Result<(), ReleaseError> {
        if self.artifacts.is_empty() || self.artifacts.len() > 64 {
            return Err(ReleaseError::Malformed("1..=64 artifacts".into()));
        }
        if self.version.is_empty() || self.version.len() > 64 {
            return Err(ReleaseError::Malformed("version length".into()));
        }
        if self.git_commit.len() > 64 || self.toolchain.len() > 64 || self.notes.len() > 1000 {
            return Err(ReleaseError::Malformed("field too long".into()));
        }
        let mut seen = BTreeSet::new();
        let mut previous = "";
        for a in &self.artifacts {
            if a.name.is_empty()
                || a.name.len() > 64
                || a.name.contains('/')
                || a.name.contains('\\')
                || a.name.starts_with('.')
                || a.name
                    .chars()
                    .any(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
            {
                return Err(ReleaseError::Malformed(format!(
                    "artifact name '{}' is not a plain file name",
                    a.name
                )));
            }
            if !seen.insert(a.name.as_str()) {
                return Err(ReleaseError::Malformed(format!("{} appears twice", a.name)));
            }
            if a.name.as_str() < previous {
                return Err(ReleaseError::Malformed("artifacts must be sorted".into()));
            }
            previous = a.name.as_str();
        }
        Ok(())
    }
}

/// Verify a signed manifest against a trust root the verifier supplies. Trust is a decision of the
/// person running the software: the manifest never carries its own list of acceptable keys.
pub fn verify_manifest(
    signed: &Signed<ReleaseManifestV1>,
    network: &NetworkId,
    trusted: &[PublicKey],
    revoked: &[PublicKey],
) -> Result<(), ReleaseError> {
    signed
        .verify(network, Domain::Release)
        .map_err(|e| ReleaseError::Signature(e.to_string()))?;
    if signed.body.network != *network {
        return Err(ReleaseError::Signature(format!(
            "manifest targets network {}",
            signed.body.network
        )));
    }
    signed.body.validate_structure()?;
    if revoked.contains(&signed.signer) {
        return Err(ReleaseError::Revoked);
    }
    if !trusted.contains(&signed.signer) {
        return Err(ReleaseError::Untrusted);
    }
    Ok(())
}

/// Check the files of `dir` against a manifest. Reading is streamed: a release is not loaded in RAM.
pub fn check_files(manifest: &ReleaseManifestV1, dir: &Path) -> Result<(), ReleaseError> {
    use std::io::Read;
    for a in &manifest.artifacts {
        let path = dir.join(&a.name);
        let mut file =
            std::fs::File::open(&path).map_err(|_| ReleaseError::Missing(a.name.clone()))?;
        let mut hasher = Sha256Digest::hasher();
        let mut buf = vec![0u8; 1 << 20];
        let mut size = 0u64;
        loop {
            let n = file.read(&mut buf).map_err(|e| ReleaseError::Mismatch {
                name: a.name.clone(),
                detail: e.to_string(),
            })?;
            if n == 0 {
                break;
            }
            size += n as u64;
            hasher.update(&buf[..n]);
        }
        let digest = hasher.finalize();
        if size != a.size {
            return Err(ReleaseError::Mismatch {
                name: a.name.clone(),
                detail: format!("{size} bytes on disk, {} in the manifest", a.size),
            });
        }
        if digest != a.sha256 {
            return Err(ReleaseError::Mismatch {
                name: a.name.clone(),
                detail: format!("sha256 {digest} on disk, {} in the manifest", a.sha256),
            });
        }
    }
    Ok(())
}

/// Hash one file for a manifest.
pub fn hash_file(path: &Path) -> Result<(u64, Sha256Digest), ProtocolError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| ProtocolError::Validation(format!("open {}: {e}", path.display())))?;
    let mut hasher = Sha256Digest::hasher();
    let mut buf = vec![0u8; 1 << 20];
    let mut size = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| ProtocolError::Validation(e.to_string()))?;
        if n == 0 {
            break;
        }
        size += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok((size, hasher.finalize()))
}

// --- Verification with standard tools: SHA256SUMS signed in the OpenSSH SSHSIG format -------------
//
// A contributor must be able to check a release **without running any downloaded binary**, `gg`
// included. The offline signer therefore also writes, from the same manifest and with the same
// Ed25519 key:
//
// * `SHA256SUMS`: the digests in the format of `sha256sum -c`, after `#` comment lines (network,
//   version, commit) that GNU coreutils and uutils skip; its last line is the digest of the signed
//   `release.json`, so both files describe the same release;
// * `SHA256SUMS.sshsig`: an OpenSSH signature of that file (PROTOCOL.sshsig), namespace
//   [`SSHSIG_NAMESPACE`], checked with `ssh-keygen -Y verify`, which every Linux distribution ships.
//
// The SSHSIG signed blob starts with the magic `SSHSIG` and carries its own namespace; our tagged
// messages start with [`crate::domain::TAG_PREFIX`]. Neither can be taken for the other, so using
// the release key for both formats opens no cross-protocol signature.

/// Namespace of the OpenSSH signature of a release (`ssh-keygen -Y verify -n gg-release`).
pub const SSHSIG_NAMESPACE: &str = "gg-release";
/// Principal used in the `allowed_signers` file handed to contributors.
pub const SSHSIG_PRINCIPAL: &str = "release@gpu-goblin";
/// File name of the checksum list.
pub const CHECKSUMS_FILE: &str = "SHA256SUMS";
/// File name of its OpenSSH signature.
pub const CHECKSUMS_SIG_FILE: &str = "SHA256SUMS.sshsig";

const SSHSIG_MAGIC: &[u8; 6] = b"SSHSIG";
const SSH_ED25519: &str = "ssh-ed25519";
const ARMOR_BEGIN: &str = "-----BEGIN SSH SIGNATURE-----";
const ARMOR_END: &str = "-----END SSH SIGNATURE-----";

/// `SHA256SUMS` of a signed release: the manifest's files, then the signed manifest itself
/// (`manifest_json`, the exact bytes written as `manifest_file` next to the binaries).
pub fn checksums_text(
    manifest: &ReleaseManifestV1,
    manifest_file: &str,
    manifest_json: &[u8],
) -> Result<String, ReleaseError> {
    manifest.validate_structure()?;
    // Comment lines are signed as well: a control character there could smuggle a checksum line.
    for (field, value) in [
        ("version", manifest.version.as_str()),
        ("git_commit", manifest.git_commit.as_str()),
        ("manifest file name", manifest_file),
    ] {
        if value.chars().any(char::is_control) {
            return Err(ReleaseError::Malformed(format!(
                "{field} contains a control character"
            )));
        }
    }
    if manifest.artifacts.iter().any(|a| a.name == manifest_file) {
        return Err(ReleaseError::Malformed(format!(
            "{manifest_file} is both an artifact and the manifest"
        )));
    }
    let mut out = String::new();
    out.push_str("# GPU Goblin release. Check the signature of this file first, then the files:\n");
    out.push_str(&format!(
        "#   ssh-keygen -Y verify -f allowed_signers -I {SSHSIG_PRINCIPAL} -n {SSHSIG_NAMESPACE} -s {CHECKSUMS_SIG_FILE} < {CHECKSUMS_FILE}\n"
    ));
    out.push_str(&format!("#   sha256sum --strict -c {CHECKSUMS_FILE}\n"));
    out.push_str(&format!("# network: {}\n", manifest.network));
    out.push_str(&format!("# version: {}\n", manifest.version));
    out.push_str(&format!("# commit: {}\n", manifest.git_commit));
    out.push_str(&format!("# built_at_ms: {}\n", manifest.built_at.0));
    for a in &manifest.artifacts {
        out.push_str(&format!("{}  {}\n", a.sha256, a.name));
    }
    out.push_str(&format!(
        "{}  {manifest_file}\n",
        Sha256Digest::of_bytes(manifest_json)
    ));
    Ok(out)
}

/// The OpenSSH public key line of an Ed25519 key (`ssh-ed25519 AAAA...`).
pub fn ssh_public_key(key: &PublicKey) -> String {
    format!("{SSH_ED25519} {}", base64_encode(&ssh_public_key_blob(key)))
}

/// One line of an `allowed_signers` file that accepts this key for release signatures only
/// (`ssh-keygen(1)`, section ALLOWED SIGNERS).
pub fn allowed_signers_line(key: &PublicKey) -> String {
    format!(
        "{SSHSIG_PRINCIPAL} namespaces=\"{SSHSIG_NAMESPACE}\" {}",
        ssh_public_key(key)
    )
}

/// Sign `message` in the armored OpenSSH SSHSIG format, hash `sha512` like `ssh-keygen -Y sign`.
pub fn sshsig_sign(key: &crate::keys::SecretKey, namespace: &str, message: &[u8]) -> String {
    let signature = key.sign_raw(&sshsig_signed_data(namespace, "sha512", message));
    let mut sig_blob = Vec::new();
    put_string(&mut sig_blob, SSH_ED25519.as_bytes());
    put_string(&mut sig_blob, &signature.0);

    let mut blob = Vec::new();
    blob.extend_from_slice(SSHSIG_MAGIC);
    blob.extend_from_slice(&1u32.to_be_bytes());
    put_string(&mut blob, &ssh_public_key_blob(&key.public_key()));
    put_string(&mut blob, namespace.as_bytes());
    put_string(&mut blob, b"");
    put_string(&mut blob, b"sha512");
    put_string(&mut blob, &sig_blob);

    let b64 = base64_encode(&blob);
    let mut out = String::from(ARMOR_BEGIN);
    out.push('\n');
    // ssh-keygen wraps at 70 columns; any width is accepted when reading. Base64 is ASCII, so
    // every 70-byte slice is a valid string.
    let mut rest = b64.as_str();
    while !rest.is_empty() {
        let (line, tail) = rest.split_at(rest.len().min(70));
        out.push_str(line);
        out.push('\n');
        rest = tail;
    }
    out.push_str(ARMOR_END);
    out.push('\n');
    out
}

/// Verify an armored SSHSIG signature of `message` in `namespace` and return the signing key.
/// Whether that key is trusted is the caller's decision, as for [`verify_manifest`].
pub fn sshsig_verify(
    armored: &str,
    namespace: &str,
    message: &[u8],
) -> Result<PublicKey, ReleaseError> {
    let bad = |m: &str| ReleaseError::Signature(format!("SSHSIG: {m}"));
    let inner = armored
        .trim()
        .strip_prefix(ARMOR_BEGIN)
        .and_then(|t| t.strip_suffix(ARMOR_END))
        .ok_or_else(|| bad("missing armor"))?;
    let b64: String = inner.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let blob = base64_decode(&b64).ok_or_else(|| bad("invalid base64"))?;
    let mut r = SshReader(&blob);
    if r.take(6)? != SSHSIG_MAGIC {
        return Err(bad("wrong magic"));
    }
    if r.u32()? != 1 {
        return Err(bad("unsupported version"));
    }
    let key = parse_ssh_public_key_blob(r.string()?)?;
    if r.string()? != namespace.as_bytes() {
        return Err(bad("wrong namespace"));
    }
    let _reserved = r.string()?;
    let hash_alg = match r.string()? {
        b"sha512" => "sha512",
        b"sha256" => "sha256",
        _ => return Err(bad("unsupported hash algorithm")),
    };
    let mut s = SshReader(r.string()?);
    if s.string()? != SSH_ED25519.as_bytes() {
        return Err(bad("not an Ed25519 signature"));
    }
    let raw: [u8; 64] = s
        .string()?
        .try_into()
        .map_err(|_| bad("signature length"))?;
    if !s.0.is_empty() || !r.0.is_empty() {
        return Err(bad("trailing bytes"));
    }
    key.verify_raw(
        &sshsig_signed_data(namespace, hash_alg, message),
        &crate::keys::Signature(raw),
    )
    .map_err(|_| bad("signature does not verify"))?;
    Ok(key)
}

/// The blob actually signed (PROTOCOL.sshsig, "Signed Data, of which the signature goes into the
/// blob above").
fn sshsig_signed_data(namespace: &str, hash_alg: &str, message: &[u8]) -> Vec<u8> {
    use sha2::Digest;
    let digest: Vec<u8> = if hash_alg == "sha256" {
        sha2::Sha256::digest(message).to_vec()
    } else {
        sha2::Sha512::digest(message).to_vec()
    };
    let mut out = Vec::new();
    out.extend_from_slice(SSHSIG_MAGIC);
    put_string(&mut out, namespace.as_bytes());
    put_string(&mut out, b"");
    put_string(&mut out, hash_alg.as_bytes());
    put_string(&mut out, &digest);
    out
}

fn ssh_public_key_blob(key: &PublicKey) -> Vec<u8> {
    let mut blob = Vec::new();
    put_string(&mut blob, SSH_ED25519.as_bytes());
    put_string(&mut blob, &key.0);
    blob
}

fn parse_ssh_public_key_blob(blob: &[u8]) -> Result<PublicKey, ReleaseError> {
    let bad = |m: &str| ReleaseError::Signature(format!("SSHSIG: {m}"));
    let mut r = SshReader(blob);
    if r.string()? != SSH_ED25519.as_bytes() {
        return Err(bad("not an Ed25519 key"));
    }
    let raw: [u8; 32] = r.string()?.try_into().map_err(|_| bad("key length"))?;
    if !r.0.is_empty() {
        return Err(bad("trailing key bytes"));
    }
    Ok(PublicKey(raw))
}

fn put_string(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
}

/// Bounded reader of the SSH wire format (RFC 4251 `uint32` and `string`).
struct SshReader<'a>(&'a [u8]);

impl<'a> SshReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ReleaseError> {
        if self.0.len() < n {
            return Err(ReleaseError::Signature("SSHSIG: truncated".into()));
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, ReleaseError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn string(&mut self) -> Result<&'a [u8], ReleaseError> {
        let n = self.u32()? as usize;
        self.take(n)
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding (RFC 4648 §4), the encoding of OpenSSH keys and signatures.
fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(chunk.get(1).copied().unwrap_or(0)) << 8)
            | u32::from(chunk.get(2).copied().unwrap_or(0));
        for (i, shift) in [18u32, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() {
                out.push(char::from(B64[((n >> shift) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let groups = bytes.len() / 4;
    let mut out = Vec::with_capacity(groups * 3);
    for (i, q) in bytes.chunks(4).enumerate() {
        let pad = q.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 || (pad > 0 && i + 1 != groups) {
            return None;
        }
        let mut n = 0u32;
        for &c in &q[..4 - pad] {
            n = (n << 6) | B64.iter().position(|&x| x == c)? as u32;
        }
        n <<= 6 * pad as u32;
        out.extend_from_slice(&[(n >> 16) as u8, (n >> 8) as u8, n as u8][..3 - pad]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::SecretKey;

    fn net() -> NetworkId {
        NetworkId::new("gg-release-test").unwrap()
    }

    fn manifest(dir: &Path) -> ReleaseManifestV1 {
        let mut artifacts = Vec::new();
        for name in ["gg", "gg-server"] {
            let path = dir.join(name);
            std::fs::write(&path, format!("binary {name}")).unwrap();
            let (size, sha256) = hash_file(&path).unwrap();
            artifacts.push(ReleaseArtifactV1 {
                name: name.to_owned(),
                size,
                sha256,
            });
        }
        ReleaseManifestV1 {
            protocol_version: crate::PROTOCOL_VERSION,
            network: net(),
            version: "0.1.0".into(),
            git_commit: "abc1234".into(),
            toolchain: "1.96.0".into(),
            built_at: UnixMillis(1_760_000_000_000),
            artifacts,
            notes: String::new(),
        }
    }

    #[test]
    fn a_release_is_accepted_only_from_a_key_the_verifier_trusts() {
        let dir = tempfile::tempdir().unwrap();
        let publisher = SecretKey::generate().unwrap();
        let stranger = SecretKey::generate().unwrap();
        let m = manifest(dir.path());
        let signed = Signed::sign(&publisher, &net(), Domain::Release, m.clone()).unwrap();

        // Trusted key, intact files: accepted.
        verify_manifest(&signed, &net(), &[publisher.public_key()], &[]).unwrap();
        check_files(&signed.body, dir.path()).unwrap();

        // A valid signature from a key nobody trusts here is refused.
        assert_eq!(
            verify_manifest(&signed, &net(), &[stranger.public_key()], &[]),
            Err(ReleaseError::Untrusted)
        );
        // A revoked key is refused even though its signature verifies.
        assert_eq!(
            verify_manifest(
                &signed,
                &net(),
                &[publisher.public_key()],
                &[publisher.public_key()]
            ),
            Err(ReleaseError::Revoked)
        );
        // Another network is another release.
        let other = NetworkId::new("gg-other").unwrap();
        assert!(matches!(
            verify_manifest(&signed, &other, &[publisher.public_key()], &[]),
            Err(ReleaseError::Signature(_))
        ));
    }

    #[test]
    fn a_modified_binary_is_caught_before_it_is_used() {
        let dir = tempfile::tempdir().unwrap();
        let publisher = SecretKey::generate().unwrap();
        let m = manifest(dir.path());
        let signed = Signed::sign(&publisher, &net(), Domain::Release, m).unwrap();
        check_files(&signed.body, dir.path()).unwrap();

        // One byte changed, same length.
        std::fs::write(dir.path().join("gg"), "binary gG").unwrap();
        match check_files(&signed.body, dir.path()) {
            Err(ReleaseError::Mismatch { name, detail }) => {
                assert_eq!(name, "gg");
                assert!(detail.contains("sha256"), "{detail}");
            }
            other => panic!("a changed binary must be refused, got {other:?}"),
        }
        // A file that grew is refused on its size, before any hash comparison.
        std::fs::write(dir.path().join("gg"), "binary gg with a tail").unwrap();
        match check_files(&signed.body, dir.path()) {
            Err(ReleaseError::Mismatch { detail, .. }) => {
                assert!(detail.contains("bytes"), "{detail}")
            }
            other => panic!("expected a size mismatch, got {other:?}"),
        }
        // A missing file is named.
        std::fs::remove_file(dir.path().join("gg")).unwrap();
        assert_eq!(
            check_files(&signed.body, dir.path()),
            Err(ReleaseError::Missing("gg".into()))
        );
    }

    #[test]
    fn a_manifest_never_names_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = manifest(dir.path());
        for bad in ["../gg", "/usr/bin/gg", "sub/gg", ".hidden", "gg;rm"] {
            m.artifacts[0].name = bad.to_owned();
            assert!(
                m.validate_structure().is_err(),
                "'{bad}' must not be an artifact name"
            );
        }
        // Sorted and unique.
        let mut m = manifest(dir.path());
        m.artifacts.reverse();
        assert!(m.validate_structure().is_err());
        let mut m = manifest(dir.path());
        m.artifacts[1].name = m.artifacts[0].name.clone();
        assert!(m.validate_structure().is_err());
        let mut m = manifest(dir.path());
        m.artifacts.clear();
        assert!(m.validate_structure().is_err());
    }

    #[test]
    fn base64_matches_rfc_4648() {
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(plain.as_bytes()), encoded);
            assert_eq!(base64_decode(encoded).unwrap(), plain.as_bytes());
        }
        for bad in ["Zg=", "Z===", "Zg==Zg==", "Zm9v!", "Zm9"] {
            assert!(base64_decode(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn an_sshsig_binds_the_message_the_namespace_and_the_key() {
        let key = SecretKey::generate().unwrap();
        let sig = sshsig_sign(&key, SSHSIG_NAMESPACE, b"release bytes");
        assert!(sig.starts_with("-----BEGIN SSH SIGNATURE-----\n"));
        assert_eq!(
            sshsig_verify(&sig, SSHSIG_NAMESPACE, b"release bytes").unwrap(),
            key.public_key()
        );
        assert!(sshsig_verify(&sig, SSHSIG_NAMESPACE, b"release bytez").is_err());
        assert!(sshsig_verify(&sig, "file", b"release bytes").is_err());
        // One flipped character inside the armor.
        let pos = sig.len() / 2;
        let mut bytes = sig.clone().into_bytes();
        bytes[pos] = if bytes[pos] == b'A' { b'B' } else { b'A' };
        let tampered = String::from_utf8(bytes).unwrap();
        assert!(sshsig_verify(&tampered, SSHSIG_NAMESPACE, b"release bytes").is_err());
    }

    #[test]
    fn checksums_carry_the_release_and_refuse_smuggled_lines() {
        let dir = tempfile::tempdir().unwrap();
        let m = manifest(dir.path());
        let text = checksums_text(&m, "release.json", b"{}").unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(lines.len(), m.artifacts.len() + 1);
        assert_eq!(
            lines[0],
            format!("{}  {}", m.artifacts[0].sha256, m.artifacts[0].name)
        );
        assert_eq!(
            *lines.last().unwrap(),
            format!("{}  release.json", Sha256Digest::of_bytes(b"{}"))
        );
        assert!(text.contains("# network: gg-release-test\n"));
        let mut evil = m.clone();
        evil.version = "0.1.0\n0000  gg".into();
        assert!(checksums_text(&evil, "release.json", b"{}").is_err());
        assert!(checksums_text(&m, "gg", b"{}").is_err());
    }

    /// Interoperability with the real OpenSSH, both ways, and the whole chain a contributor runs:
    /// `ssh-keygen -Y verify` on SHA256SUMS, then `sha256sum --strict -c`. This test needs
    /// `ssh-keygen` (openssh-client) and fails when it is missing: skipping would hide a broken
    /// release path.
    #[test]
    fn openssh_and_coreutils_verify_what_we_sign() {
        use std::process::{Command, Stdio};
        let run_with_stdin = |cmd: &mut Command, stdin: &Path| {
            cmd.stdin(Stdio::from(std::fs::File::open(stdin).unwrap()))
                .output()
                .expect("ssh-keygen is required by this test (install openssh-client)")
        };
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let key = SecretKey::generate().unwrap();
        let m = manifest(d);
        let signed = Signed::sign(&key, &net(), Domain::Release, m.clone()).unwrap();
        let json = serde_json::to_vec_pretty(&signed).unwrap();
        std::fs::write(d.join("release.json"), &json).unwrap();
        let sums = checksums_text(&m, "release.json", &json).unwrap();
        std::fs::write(d.join(CHECKSUMS_FILE), &sums).unwrap();
        std::fs::write(
            d.join(CHECKSUMS_SIG_FILE),
            sshsig_sign(&key, SSHSIG_NAMESPACE, sums.as_bytes()),
        )
        .unwrap();
        std::fs::write(
            d.join("allowed_signers"),
            allowed_signers_line(&key.public_key()) + "\n",
        )
        .unwrap();
        let verify = |allowed: &str, namespace: &str, message: &str| {
            let out = run_with_stdin(
                Command::new("ssh-keygen")
                    .current_dir(d)
                    .args(["-Y", "verify", "-f", allowed, "-I", SSHSIG_PRINCIPAL])
                    .args(["-n", namespace, "-s", CHECKSUMS_SIG_FILE]),
                &d.join(message),
            );
            (
                out.status.success(),
                String::from_utf8_lossy(&out.stdout).into_owned()
                    + &String::from_utf8_lossy(&out.stderr),
            )
        };

        // Our signature, OpenSSH's verifier.
        let (ok, log) = verify("allowed_signers", SSHSIG_NAMESPACE, CHECKSUMS_FILE);
        assert!(ok, "ssh-keygen refused a valid signature: {log}");
        // Then coreutils checks the files (comment lines are skipped, --strict refuses junk).
        let sha = Command::new("sha256sum")
            .current_dir(d)
            .args(["--strict", "-c", CHECKSUMS_FILE])
            .output()
            .expect("sha256sum (coreutils) is required by this test");
        assert!(
            sha.status.success(),
            "{}",
            String::from_utf8_lossy(&sha.stderr)
        );

        // A modified checksum file, another namespace, another key: all refused by OpenSSH.
        std::fs::write(d.join("tampered"), sums.replace("  gg\n", "  gh\n")).unwrap();
        assert!(!verify("allowed_signers", SSHSIG_NAMESPACE, "tampered").0);
        assert!(!verify("allowed_signers", "file", CHECKSUMS_FILE).0);
        let other = SecretKey::generate().unwrap();
        std::fs::write(
            d.join("other_signers"),
            allowed_signers_line(&other.public_key()) + "\n",
        )
        .unwrap();
        assert!(!verify("other_signers", SSHSIG_NAMESPACE, CHECKSUMS_FILE).0);
        // A binary modified after signing: coreutils names it.
        std::fs::write(d.join("gg"), "binary gG").unwrap();
        let sha = Command::new("sha256sum")
            .current_dir(d)
            .args(["--strict", "-c", CHECKSUMS_FILE])
            .output()
            .unwrap();
        assert!(!sha.status.success());

        // OpenSSH's signature, our verifier: same wire format both ways.
        let status = Command::new("ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-C", "", "-f"])
            .arg(d.join("id"))
            .status()
            .expect("ssh-keygen is required by this test (install openssh-client)");
        assert!(status.success());
        let status = Command::new("ssh-keygen")
            .args(["-q", "-Y", "sign", "-n", SSHSIG_NAMESPACE, "-f"])
            .arg(d.join("id"))
            .arg(d.join(CHECKSUMS_FILE))
            .status()
            .unwrap();
        assert!(status.success());
        let theirs = std::fs::read_to_string(d.join("SHA256SUMS.sig")).unwrap();
        let signer = sshsig_verify(&theirs, SSHSIG_NAMESPACE, sums.as_bytes()).unwrap();
        let pubkey = std::fs::read_to_string(d.join("id.pub")).unwrap();
        assert_eq!(pubkey.trim(), ssh_public_key(&signer));
        assert!(sshsig_verify(&theirs, SSHSIG_NAMESPACE, b"something else").is_err());
    }
}
