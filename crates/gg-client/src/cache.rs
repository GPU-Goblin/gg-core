//! Local artifact cache keyed by SHA-256 (ADR-0011: checkpoints and datasets are reused across
//! rounds). The cache is never a source of authority: hashes come from the server's signed spec,
//! every hit is re-hashed before use, and a corrupted entry is discarded. Bounded by `max_bytes`
//! with eviction of the least recently used entries.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use gg_core::Sha256Digest;

/// Content-addressed cache directory.
#[derive(Clone, Debug)]
pub struct ArtifactCache {
    root: PathBuf,
    max_bytes: u64,
    /// Artifacts used by an attempt in progress: never evicted while pinned.
    pinned: Arc<Mutex<HashSet<Sha256Digest>>>,
}

/// Outcome of a lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The artifact was materialized at the destination from an intact cached copy.
    Hit,
    /// Not cached (or disabled).
    Miss,
    /// A cached file existed but its size or hash did not match; it was removed.
    Corrupted,
}

impl ArtifactCache {
    /// Cache rooted at `root` (created lazily). `max_bytes = 0` disables caching.
    pub fn new(root: PathBuf, max_bytes: u64) -> Self {
        Self {
            root,
            max_bytes,
            pinned: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Protect an artifact from eviction (an attempt is using it).
    pub fn pin(&self, hash: &Sha256Digest) {
        self.pinned
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(*hash);
    }

    /// Release a pin taken by [`ArtifactCache::pin`].
    pub fn unpin(&self, hash: &Sha256Digest) {
        self.pinned
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(hash);
    }

    /// Whether an artifact is currently pinned.
    pub fn is_pinned(&self, hash: &Sha256Digest) -> bool {
        self.pinned
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(hash)
    }

    /// Path of a cached artifact (it may not exist).
    pub fn path_of(&self, hash: &Sha256Digest) -> PathBuf {
        self.entry(hash)
    }

    /// Read `len` bytes at `start` of a cached artifact, checking the file size first. Returns
    /// `None` when the artifact is absent or its size does not match.
    pub fn read_range(
        &self,
        hash: &Sha256Digest,
        size: u64,
        start: u64,
        len: u64,
    ) -> std::io::Result<Option<Vec<u8>>> {
        use std::io::{Read, Seek, SeekFrom};
        let path = self.entry(hash);
        let Ok(meta) = std::fs::metadata(&path) else {
            return Ok(None);
        };
        if !meta.is_file() || meta.len() != size || start + len > size {
            return Ok(None);
        }
        let mut f = std::fs::File::open(&path)?;
        f.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; len as usize];
        f.read_exact(&mut buf)?;
        Ok(Some(buf))
    }

    /// Root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn entry(&self, hash: &Sha256Digest) -> PathBuf {
        self.root.join(hash.to_hex())
    }

    /// Materialize `hash` at `dest` from the cache after re-hashing the cached bytes.
    pub fn fetch(&self, hash: &Sha256Digest, size: u64, dest: &Path) -> std::io::Result<Lookup> {
        if self.max_bytes == 0 {
            return Ok(Lookup::Miss);
        }
        let entry = self.entry(hash);
        let Ok(meta) = std::fs::metadata(&entry) else {
            return Ok(Lookup::Miss);
        };
        if !meta.is_file() || meta.len() != size || hash_file(&entry)? != *hash {
            let _ = std::fs::remove_file(&entry);
            return Ok(Lookup::Corrupted);
        }
        let _ = std::fs::remove_file(dest);
        if std::fs::hard_link(&entry, dest).is_err() {
            std::fs::copy(&entry, dest)?;
        }
        // Recency for eviction.
        if let Ok(f) = std::fs::File::options().write(true).open(&entry) {
            let _ = f.set_modified(SystemTime::now());
        }
        Ok(Lookup::Hit)
    }

    /// Store the file at `src` (already verified against `hash` by the downloader) and evict the
    /// least recently used entries beyond the budget.
    pub fn store(&self, hash: &Sha256Digest, src: &Path) -> std::io::Result<()> {
        if self.max_bytes == 0 {
            return Ok(());
        }
        let size = std::fs::metadata(src)?.len();
        if size > self.max_bytes {
            return Ok(());
        }
        std::fs::create_dir_all(&self.root)?;
        let entry = self.entry(hash);
        if entry.exists() {
            return Ok(());
        }
        // Unique per call: two stores of the same artifact may run at once (parallel inputs).
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = self.root.join(format!(
            ".tmp-{}-{}-{seq}",
            hash.to_hex(),
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        if std::fs::hard_link(src, &tmp).is_err() {
            std::fs::copy(src, &tmp)?;
        }
        // Another store of the same artifact may have won the race: same bytes, keep either.
        if let Err(e) = std::fs::rename(&tmp, &entry) {
            let _ = std::fs::remove_file(&tmp);
            if !entry.exists() {
                return Err(e);
            }
        }
        self.evict()
    }

    /// Bytes currently cached.
    pub fn used_bytes(&self) -> std::io::Result<u64> {
        Ok(self.entries()?.iter().map(|e| e.1).sum())
    }

    fn entries(&self) -> std::io::Result<Vec<(PathBuf, u64, SystemTime)>> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.root) else {
            return Ok(out);
        };
        for e in rd {
            let e = e?;
            let meta = e.metadata()?;
            if !meta.is_file() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            out.push((
                e.path(),
                meta.len(),
                meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            ));
        }
        Ok(out)
    }

    fn evict(&self) -> std::io::Result<()> {
        let mut entries = self.entries()?;
        let mut used: u64 = entries.iter().map(|e| e.1).sum();
        entries.sort_by_key(|e| e.2);
        let pinned = self
            .pinned
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|h| h.to_hex())
            .collect::<HashSet<String>>();
        for (path, size, _) in entries {
            if used <= self.max_bytes {
                break;
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if pinned.contains(&name) {
                continue; // in use by an attempt in progress
            }
            if std::fs::remove_file(&path).is_ok() {
                used -= size;
            }
        }
        Ok(())
    }
}

/// Streaming SHA-256 of a file.
pub fn hash_file(path: &Path) -> std::io::Result<Sha256Digest> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256Digest::hasher();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize())
}
