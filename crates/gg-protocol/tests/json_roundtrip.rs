//! Signed messages must survive the JSON (human-readable) transport: the server re-encodes the
//! body canonically before verifying, so any lossy JSON conversion would break signatures.

use gg_core::*;
use gg_protocol::domain::Domain;
use gg_protocol::keys::SecretKey;
use gg_protocol::messages::*;
use gg_protocol::signed::Signed;

fn decision(values: &[f64]) -> VerificationDecisionV1 {
    VerificationDecisionV1 {
        protocol_version: gg_protocol::PROTOCOL_VERSION,
        network: NetworkId::new("gg-json").unwrap(),
        decision_id: DecisionId::new(),
        submission_hash: Sha256Digest::zero(),
        task_id: TaskId::new(),
        attempt_id: AttemptId::new(),
        validator_id: ValidatorId::new(),
        method: "recompute-reference".into(),
        method_version: 1,
        reference_profile: "cpu-f32-flex-v1".into(),
        checks: values
            .iter()
            .enumerate()
            .map(|(i, v)| CheckV1 {
                name: format!("c{i}"),
                passed: true,
                measured: Some(*v),
                tolerance: Some(1e-4),
                detail: format!("{v:?}"),
            })
            .collect(),
        artifacts_checked: vec![],
        outcome: VerificationOutcome::Accepted,
        decided_at: UnixMillis::now(),
    }
}

#[test]
fn decision_survives_json_text_and_value() {
    let key = SecretKey::generate().unwrap();
    let net = NetworkId::new("gg-json").unwrap();
    let values = [
        0.0,
        -0.0,
        3.0,
        7.248e-5,
        0.5249,
        1.526e-4,
        1e-4,
        f64::MIN_POSITIVE,
        1.7976931348623157e308,
        0.1 + 0.2,
        1.0 / 3.0,
        5e-324,
        0.11082745764249695,
        6.103515625e-5,
    ];
    let signed = Signed::sign(&key, &net, Domain::Verification, decision(&values)).unwrap();
    // via text
    let text = serde_json::to_string(&signed).unwrap();
    let back: Signed<VerificationDecisionV1> = serde_json::from_str(&text).unwrap();
    back.verify(&net, Domain::Verification).unwrap();
    // via Value (what reqwest .json(&Value) does)
    let value = serde_json::to_value(&signed).unwrap();
    let text2 = serde_json::to_string(&value).unwrap();
    let back2: Signed<VerificationDecisionV1> = serde_json::from_str(&text2).unwrap();
    back2.verify(&net, Domain::Verification).unwrap();
}

#[test]
fn non_finite_measurements_cannot_be_signed() {
    let key = SecretKey::generate().unwrap();
    let net = NetworkId::new("gg-json").unwrap();
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            Signed::sign(&key, &net, Domain::Verification, decision(&[v])).is_err(),
            "{v} must be refused"
        );
    }
}

mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Any finite f64 must cross JSON text unchanged (requires serde_json's `float_roundtrip`;
        /// without it 0.11082745764249695 comes back one ulp off and signatures break).
        #[test]
        fn finite_f64_roundtrips_exactly(bits in any::<u64>()) {
            let v = f64::from_bits(bits);
            prop_assume!(v.is_finite());
            let text = serde_json::to_string(&v).unwrap();
            let back: f64 = serde_json::from_str(&text).unwrap();
            prop_assert_eq!(back.to_bits(), v.to_bits(), "{} -> {} -> {}", v, text, back);
        }

        #[test]
        fn signed_decisions_with_random_measurements_verify_after_json(values in proptest::collection::vec(any::<u64>().prop_map(f64::from_bits).prop_filter("finite", |v| v.is_finite()), 1..8)) {
            let key = SecretKey::from_bytes(&[5u8; 32]);
            let net = NetworkId::new("gg-json").unwrap();
            let signed = Signed::sign(&key, &net, Domain::Verification, decision(&values)).unwrap();
            let text = serde_json::to_string(&serde_json::to_value(&signed).unwrap()).unwrap();
            let back: Signed<VerificationDecisionV1> = serde_json::from_str(&text).unwrap();
            prop_assert!(back.verify(&net, Domain::Verification).is_ok());
        }
    }
}
