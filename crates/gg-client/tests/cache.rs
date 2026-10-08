//! The local artifact cache never trusts its own content: hits are re-hashed, corrupted entries
//! are dropped, the budget is enforced by evicting the least recently used entries.

use std::time::Duration;

use gg_client::cache::{ArtifactCache, Lookup};
use gg_core::Sha256Digest;

fn blob(seed: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
}

#[test]
fn hit_requires_matching_hash_and_size() {
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("cache"), 1 << 20);
    let bytes = blob(1, 4096);
    let hash = Sha256Digest::of_bytes(&bytes);
    let src = dir.path().join("downloaded");
    std::fs::write(&src, &bytes).unwrap();
    let dest = dir.path().join("dest");
    assert_eq!(cache.fetch(&hash, 4096, &dest).unwrap(), Lookup::Miss);
    cache.store(&hash, &src).unwrap();
    assert_eq!(cache.fetch(&hash, 4096, &dest).unwrap(), Lookup::Hit);
    assert_eq!(std::fs::read(&dest).unwrap(), bytes);
    // A size mismatch is refused even for the right hash.
    assert_eq!(
        cache.fetch(&hash, 4095, &dir.path().join("dest2")).unwrap(),
        Lookup::Corrupted
    );
    assert_eq!(cache.fetch(&hash, 4096, &dest).unwrap(), Lookup::Miss);
}

#[test]
fn corrupted_entry_is_detected_and_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("cache"), 1 << 20);
    let bytes = blob(7, 1000);
    let hash = Sha256Digest::of_bytes(&bytes);
    let src = dir.path().join("downloaded");
    std::fs::write(&src, &bytes).unwrap();
    cache.store(&hash, &src).unwrap();
    // Flip one byte in the cached copy (through the cache directory, not the source).
    let entry = cache.root().join(hash.to_hex());
    let mut cached = std::fs::read(&entry).unwrap();
    cached[10] ^= 0xff;
    std::fs::remove_file(&entry).unwrap(); // break the hard link before rewriting
    std::fs::write(&entry, &cached).unwrap();
    let dest = dir.path().join("dest");
    assert_eq!(cache.fetch(&hash, 1000, &dest).unwrap(), Lookup::Corrupted);
    assert!(!entry.exists(), "corrupted entry must be removed");
    assert!(
        !dest.exists(),
        "nothing materialized from a corrupted entry"
    );
}

#[test]
fn budget_evicts_least_recently_used_and_zero_disables() {
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("cache"), 2500);
    let mut hashes = Vec::new();
    for i in 0..3u8 {
        let bytes = blob(i, 1000);
        let hash = Sha256Digest::of_bytes(&bytes);
        let src = dir.path().join(format!("dl{i}"));
        std::fs::write(&src, &bytes).unwrap();
        cache.store(&hash, &src).unwrap();
        hashes.push(hash);
        std::thread::sleep(Duration::from_millis(20));
        if i == 1 {
            // Touch entry 0 so that entry 1 becomes the least recently used.
            assert_eq!(
                cache
                    .fetch(&hashes[0], 1000, &dir.path().join("touch"))
                    .unwrap(),
                Lookup::Hit
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    assert!(cache.used_bytes().unwrap() <= 2500);
    assert!(cache.root().join(hashes[0].to_hex()).exists());
    assert!(
        !cache.root().join(hashes[1].to_hex()).exists(),
        "LRU entry evicted"
    );
    assert!(cache.root().join(hashes[2].to_hex()).exists());

    let off = ArtifactCache::new(dir.path().join("off"), 0);
    off.store(&hashes[2], &dir.path().join("dl2")).unwrap();
    assert_eq!(
        off.fetch(&hashes[2], 1000, &dir.path().join("d")).unwrap(),
        Lookup::Miss
    );
    assert!(!off.root().exists());
}

/// A task's inputs are fetched in parallel, and two of them may carry the same artifact.
/// Simultaneous cache stores of the same hash do not step on each other, and two downloads to
/// `in.0` and `in.1` do not write the same temporary file.
#[test]
fn concurrent_stores_of_one_artifact_and_temporary_names_do_not_collide() {
    let dir = tempfile::tempdir().unwrap();
    let cache = ArtifactCache::new(dir.path().join("cache"), 1 << 30);
    let data = blob(7, 300_000);
    let hash = Sha256Digest::of_bytes(&data);
    let sources: Vec<std::path::PathBuf> = (0..8)
        .map(|i| {
            let p = dir.path().join(format!("in.{i}"));
            std::fs::write(&p, &data).unwrap();
            p
        })
        .collect();
    std::thread::scope(|s| {
        for src in &sources {
            let cache = &cache;
            s.spawn(move || cache.store(&hash, src).unwrap());
        }
    });
    let dest = dir.path().join("out");
    assert_eq!(
        cache.fetch(&hash, data.len() as u64, &dest).unwrap(),
        Lookup::Hit
    );
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    // No temporary file left behind.
    let leftovers = std::fs::read_dir(dir.path().join("cache"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with('.')
        })
        .count();
    assert_eq!(leftovers, 0);

    let a = gg_client::client::part_path(&dir.path().join("in.0"));
    let b = gg_client::client::part_path(&dir.path().join("in.1"));
    assert_ne!(a, b);
    assert!(a.to_string_lossy().ends_with("in.0.part"));
}

/// Stores running at once (parallel downloads, a lab node pulling while it writes a checkpoint)
/// never fail because another store renamed or evicted a file while the cache was being listed.
#[test]
fn concurrent_stores_do_not_fail_on_each_others_files() {
    let dir = tempfile::tempdir().unwrap();
    let cache = gg_client::cache::ArtifactCache::new(dir.path().join("cache"), u64::MAX);
    let threads: Vec<_> = (0..8u32)
        .map(|t| {
            let cache = cache.clone();
            let src = dir.path().join(format!("src-{t}"));
            std::fs::create_dir_all(&src).unwrap();
            std::thread::spawn(move || {
                for i in 0..200u32 {
                    let bytes = format!("{t}-{i}").into_bytes();
                    let path = src.join(i.to_string());
                    std::fs::write(&path, &bytes).unwrap();
                    cache
                        .store(&gg_core::Sha256Digest::of_bytes(&bytes), &path)
                        .unwrap();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert!(cache.used_bytes().unwrap() > 0);
}
