//! Protocol test vectors. The fixtures in `tests/fixtures/protocol/` pin the canonical bytes and the
//! domain-separated hashes of every message type. Regenerate deliberately with
//! `GG_UPDATE_VECTORS=1 cargo test -p gg-protocol --test vectors`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_core::*;
use gg_protocol::PROTOCOL_VERSION;
use gg_protocol::canonical;
use gg_protocol::domain::{Domain, hash_bytes};
use gg_protocol::drbg::Drbg;
use gg_protocol::keys::{PublicKey, SecretKey};
use gg_protocol::messages::*;
use gg_protocol::release::{ReleaseArtifactV1, ReleaseManifestV1};
use gg_protocol::signed::Signed;
use serde::Serialize;
use serde::de::DeserializeOwned;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/protocol")
}

fn net() -> NetworkId {
    NetworkId::new("gg-test-vectors").unwrap()
}

fn digest(n: u8) -> Sha256Digest {
    Sha256Digest([n; 32])
}

fn fixed_key(n: u8) -> SecretKey {
    SecretKey::from_bytes(&[n; 32])
}

#[derive(Serialize, serde::Deserialize)]
struct Vector {
    json: serde_json::Value,
    dagcbor_hex: String,
    domain: String,
    hash_hex: String,
}

fn check<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(
    name: &str,
    domain: Domain,
    value: &T,
) {
    let bytes = canonical::to_bytes(value).expect("encode");
    let hash = hash_bytes(&net(), domain, &bytes);
    let decoded: T = canonical::from_bytes(&bytes).expect("decode");
    assert_eq!(&decoded, value, "{name}: roundtrip");
    let json_value = serde_json::to_value(value).expect("json");
    let from_json: T = serde_json::from_value(json_value.clone()).expect("json roundtrip");
    assert_eq!(&from_json, value, "{name}: json roundtrip");

    let path = fixtures_dir().join(format!("{name}.json"));
    let current = Vector {
        json: json_value,
        dagcbor_hex: hex::encode(&bytes),
        domain: domain.as_str().to_owned(),
        hash_hex: hash.to_hex(),
    };
    if std::env::var_os("GG_UPDATE_VECTORS").is_some() {
        std::fs::create_dir_all(fixtures_dir()).unwrap();
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&current).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {e} (run with GG_UPDATE_VECTORS=1)",
            path.display()
        )
    });
    let expected: Vector = serde_json::from_str(&text).unwrap();
    assert_eq!(
        current.dagcbor_hex, expected.dagcbor_hex,
        "{name}: canonical bytes changed"
    );
    assert_eq!(current.hash_hex, expected.hash_hex, "{name}: hash changed");
    assert_eq!(current.json, expected.json, "{name}: json changed");
    // The pinned bytes must decode to the same value.
    let pinned: T = canonical::from_bytes(&hex::decode(&expected.dagcbor_hex).unwrap()).unwrap();
    assert_eq!(&pinned, value, "{name}: pinned bytes decode differently");
}

fn definition() -> TaskDefinitionV1 {
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "a".to_owned(),
        ArtifactRef {
            hash: digest(0xA1),
            size: 6 + 16 + 4 * 6,
        },
    );
    inputs.insert(
        "b".to_owned(),
        ArtifactRef {
            hash: digest(0xB2),
            size: 6 + 16 + 4 * 12,
        },
    );
    TaskDefinitionV1 {
        catalog_version: 1,
        params: TaskParamsV1::Matmul { m: 2, k: 3, n: 4 },
        inputs,
        numeric_profile: "f32-v1".into(),
        seed_strategy: SeedStrategyV1::None,
        resource_limits: ResourceLimits {
            max_ram_bytes: 1 << 30,
            max_disk_bytes: 1 << 30,
            max_duration_secs: 600,
            max_output_bytes: 1 << 20,
        },
        validation_policy: ValidationPolicyV1 {
            method: "recompute-reference".into(),
            method_version: 1,
            reference_profile: "cpu-f32-flex-v1".into(),
            tolerance: ToleranceV1 {
                abs: 1e-4,
                rel: 1e-4,
            },
        },
        reward_schedule: RewardScheduleV1 {
            primary: Credits(10),
            replica: Credits(8),
            audit: Credits(8),
            verification_cost: Credits(10),
        },
        max_attempts: 3,
    }
}

fn spec() -> TaskSpecV1 {
    TaskSpecV1 {
        spec_version: 1,
        network: net(),
        project_id: ProjectId::from_u128(1),
        experiment_id: ExperimentId::from_u128(2),
        job_id: JobId::from_u128(3),
        task_id: TaskId::from_u128(4),
        index: 0,
        definition: definition(),
    }
}

#[test]
fn task_definition_and_spec_vectors() {
    let d = definition();
    d.validate_structure().unwrap();
    check("task_definition_v1", Domain::TaskDefinition, &d);
    check("task_spec_v1", Domain::TaskSpec, &spec());
}

#[test]
fn grad_params_vector() {
    let d = TaskDefinitionV1 {
        params: TaskParamsV1::MlpGrad(MlpGradParamsV1 {
            d_input: 4,
            d_hidden: 8,
            d_output: 2,
            round_seed: 9,
            example_start: 64,
            example_count: 32,
            total_examples: 1024,
        }),
        seed_strategy: SeedStrategyV1::ProtocolDrbg { seed: 9 },
        ..definition()
    };
    check("task_definition_grad_v1", Domain::TaskDefinition, &d);
}

#[test]
fn assignment_vector() {
    let a = AssignmentV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        task_id: TaskId::from_u128(4),
        spec_hash: digest(0x51),
        attempt_id: AttemptId::from_u128(5),
        worker_id: WorkerId::from_u128(6),
        lease_id: LeaseId::from_u128(7),
        generation: 1,
        role: AttemptRole::Primary,
        assigned_at: UnixMillis(1_700_000_000_000),
        lease_expires_at: UnixMillis(1_700_000_120_000),
        lease_max_until: UnixMillis(1_700_003_600_000),
    };
    check("assignment_v1", Domain::Assignment, &a);
    let signed = Signed::sign(&fixed_key(1), &net(), Domain::Assignment, a).unwrap();
    signed.verify(&net(), Domain::Assignment).unwrap();
    check("assignment_v1_signed", Domain::Assignment, &signed);
}

#[test]
fn result_submission_vector() {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "c".to_owned(),
        ArtifactRef {
            hash: digest(0xC3),
            size: 6 + 16 + 4 * 8,
        },
    );
    let mut metrics = BTreeMap::new();
    metrics.insert("duration_ms".to_owned(), 12.5);
    let r = ResultSubmissionV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        assignment_hash: digest(0x52),
        task_id: TaskId::from_u128(4),
        attempt_id: AttemptId::from_u128(5),
        spec_hash: digest(0x51),
        generation: 1,
        outputs,
        execution_profile: ExecutionProfileV1 {
            engine: "burn-flex".into(),
            engine_version: "0.21.0".into(),
            numeric_profile: "cpu-f32-flex-v1".into(),
            device_name: "cpu".into(),
            duration_ms: 12,
        },
        declared_metrics: metrics,
        submitted_at: UnixMillis(1_700_000_010_000),
    };
    r.validate_structure().unwrap();
    check("result_submission_v1", Domain::Result, &r);
    let signed = Signed::sign(&fixed_key(2), &net(), Domain::Result, r).unwrap();
    check("result_submission_v1_signed", Domain::Result, &signed);
}

#[test]
fn verification_decision_vector() {
    let d = VerificationDecisionV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        decision_id: DecisionId::from_u128(8),
        submission_hash: digest(0x53),
        task_id: TaskId::from_u128(4),
        attempt_id: AttemptId::from_u128(5),
        validator_id: ValidatorId::from_u128(9),
        method: "recompute-reference".into(),
        method_version: 1,
        reference_profile: "cpu-f32-flex-v1".into(),
        checks: vec![
            CheckV1 {
                name: "shape".into(),
                passed: true,
                measured: None,
                tolerance: None,
                detail: "[2,4]".into(),
            },
            CheckV1 {
                name: "max_abs_diff".into(),
                passed: true,
                measured: Some(0.0),
                tolerance: Some(1e-4),
                detail: String::new(),
            },
        ],
        artifacts_checked: vec![ArtifactRef {
            hash: digest(0xC3),
            size: 54,
        }],
        outcome: VerificationOutcome::Accepted,
        decided_at: UnixMillis(1_700_000_020_000),
    };
    d.validate_structure().unwrap();
    check("verification_decision_v1", Domain::Verification, &d);
    let signed = Signed::sign(&fixed_key(3), &net(), Domain::Verification, d).unwrap();
    check(
        "verification_decision_v1_signed",
        Domain::Verification,
        &signed,
    );
}

#[test]
fn receipt_vector() {
    let r = ContributionReceiptV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        receipt_id: ReceiptId::from_u128(10),
        project_id: ProjectId::from_u128(1),
        experiment_id: ExperimentId::from_u128(2),
        job_id: JobId::from_u128(3),
        task_id: TaskId::from_u128(4),
        attempt_id: AttemptId::from_u128(5),
        role: AttemptRole::Primary,
        spec_hash: digest(0x51),
        definition_hash: digest(0x50),
        input_hashes: vec![digest(0xA1), digest(0xB2)],
        submission_hash: digest(0x53),
        output_hashes: vec![digest(0xC3)],
        decision_hash: digest(0x54),
        decision_hashes: vec![digest(0x54), digest(0x55)],
        worker_id: WorkerId::from_u128(6),
        worker_public_key: fixed_key(2).public_key(),
        beneficiary: None,
        amount: Credits(10),
        asset: "TC".into(),
        issued_at: UnixMillis(1_700_000_030_000),
    };
    check("contribution_receipt_v1", Domain::Receipt, &r);
    let signed = Signed::sign(&fixed_key(1), &net(), Domain::Receipt, r).unwrap();
    check("contribution_receipt_v1_signed", Domain::Receipt, &signed);
}

#[test]
fn login_challenge_vector() {
    let c = LoginChallengeV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        action: "worker-login".into(),
        nonce: vec![0x11; 32],
        public_key: fixed_key(2).public_key(),
        issued_at: UnixMillis(1_700_000_000_000),
    };
    c.validate_structure().unwrap();
    check("login_challenge_v1", Domain::Login, &c);
}

#[test]
fn keys_and_drbg_vectors() {
    let pk: PublicKey = fixed_key(2).public_key();
    check("public_key_fixed_2", Domain::Login, &pk);
    let mut d = Drbg::new("vectors", 42);
    let words: Vec<u32> = (0..8).map(|_| d.next_u32()).collect();
    let floats: Vec<f32> = (0..4).map(|_| d.next_sym_f32()).collect();
    let perm = d.permutation(8);
    #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
    struct DrbgVec {
        words: Vec<u32>,
        floats: Vec<f32>,
        perm: Vec<u32>,
    }
    check(
        "drbg_v1",
        Domain::TaskDefinition,
        &DrbgVec {
            words,
            floats,
            perm,
        },
    );
}

#[test]
fn tampered_signed_vector_fails() {
    let path = fixtures_dir().join("result_submission_v1_signed.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    let v: Vector = serde_json::from_str(&text).unwrap();
    let mut bytes = hex::decode(v.dagcbor_hex).unwrap();
    let signed: Signed<ResultSubmissionV1> = canonical::from_bytes(&bytes).unwrap();
    signed.verify(&net(), Domain::Result).unwrap();
    // flip one byte in the middle of the body
    let i = bytes.len() / 2;
    bytes[i] ^= 0x01;
    if let Ok(t) = canonical::from_bytes::<Signed<ResultSubmissionV1>>(&bytes) {
        assert!(t.verify(&net(), Domain::Result).is_err());
    }
}

#[test]
fn chunk_manifest_vector() {
    let m = ChunkManifestV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        artifact_hash: digest(0xC1),
        size: 64 * 1024 + 7,
        chunk_size: 64 * 1024,
        chunks: vec![digest(0x11), digest(0x22)],
        issued_at: UnixMillis(1_760_000_000_000),
    };
    m.validate_structure().unwrap();
    assert_eq!(m.range_of(0), Some((0, 65536)));
    assert_eq!(m.range_of(1), Some((65536, 7)));
    assert_eq!(m.range_of(2), None);
    check("chunk_manifest_v1", Domain::ChunkManifest, &m);
    let signed = Signed::sign(&fixed_key(9), &net(), Domain::ChunkManifest, m.clone()).unwrap();
    check("chunk_manifest_signed_v1", Domain::ChunkManifest, &signed);
    signed.verify(&net(), Domain::ChunkManifest).unwrap();
    // A manifest whose chunk count contradicts the size is refused before any byte is fetched.
    let mut bad = m.clone();
    bad.chunks.pop();
    assert!(bad.validate_structure().is_err());
    let mut small = m;
    small.chunk_size = 1024;
    assert!(small.validate_structure().is_err());
}

#[test]
fn peer_ticket_and_chunk_request_vectors() {
    let t = PeerTicketV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        artifact_hash: digest(0xC1),
        size: 64 * 1024 + 7,
        chunk_size: 64 * 1024,
        reader_id: WorkerId::from_u128(7),
        reader_key: fixed_key(4).public_key(),
        issued_at: UnixMillis(1_760_000_000_000),
        expires_at: UnixMillis(1_760_000_300_000),
    };
    assert_eq!(t.range_of(0), Some((0, 65536)));
    assert_eq!(t.range_of(1), Some((65536, 7)));
    assert_eq!(t.range_of(2), None);
    check("peer_ticket_v1", Domain::PeerTicket, &t);
    let r = ChunkRequestV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        artifact_hash: digest(0xC1),
        chunk_index: 1,
        issued_at: UnixMillis(1_760_000_000_500),
    };
    check("chunk_request_v1", Domain::ChunkRequest, &r);
    // The ticket and the request live in different domains: a signature over one is not valid for
    // the other even with identical bytes.
    let key = fixed_key(4);
    let signed = Signed::sign(&key, &net(), Domain::ChunkRequest, r).unwrap();
    signed.verify(&net(), Domain::ChunkRequest).unwrap();
    assert!(signed.verify(&net(), Domain::PeerTicket).is_err());
}

#[test]
fn release_manifest_vector() {
    let m = ReleaseManifestV1 {
        protocol_version: PROTOCOL_VERSION,
        network: net(),
        version: "0.1.0".into(),
        git_commit: "0123456789abcdef0123456789abcdef01234567".into(),
        toolchain: "1.96.0".into(),
        built_at: UnixMillis(1_760_000_000_000),
        artifacts: vec![
            ReleaseArtifactV1 {
                name: "gg".into(),
                size: 1_234_567,
                sha256: digest(0x71),
            },
            ReleaseArtifactV1 {
                name: "gg-server".into(),
                size: 7_654_321,
                sha256: digest(0x72),
            },
        ],
        notes: "test vector".into(),
    };
    m.validate_structure().unwrap();
    check("release_manifest_v1", Domain::Release, &m);
    let signed = Signed::sign(&fixed_key(3), &net(), Domain::Release, m).unwrap();
    check("release_manifest_signed_v1", Domain::Release, &signed);
    signed.verify(&net(), Domain::Release).unwrap();
    // A release manifest is not readable in any other domain.
    assert!(signed.verify(&net(), Domain::Receipt).is_err());
}
