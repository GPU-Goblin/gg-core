//! Perturbation suite for `tensor.matmul.v1` verification: measures false rejections on conformant
//! outputs and undetected corruptions on non-conformant ones (both must be zero).

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_compute::catalog::{matmul_definition, validate_definition};
use gg_compute::engine::EngineSelection;
use gg_compute::matmul::{execute, matmul_naive_f64, verify};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::ArtifactRef;
use gg_protocol::tensor_file::TensorF32;

struct Case {
    dir: tempfile::TempDir,
    inputs: BTreeMap<String, PathBuf>,
    task: gg_compute::ValidatedTask,
}

fn make_case(m: u32, k: u32, n: u32, seed: u64) -> Case {
    let dir = tempfile::tempdir().unwrap();
    let mut d = Drbg::new("perturbations", seed);
    let a = TensorF32::new(vec![m as u64, k as u64], d.sym_f32_vec((m * k) as usize)).unwrap();
    let b = TensorF32::new(vec![k as u64, n as u64], d.sym_f32_vec((k * n) as usize)).unwrap();
    let mut inputs = BTreeMap::new();
    let mut refs = Vec::new();
    for (name, t) in [("a", &a), ("b", &b)] {
        let bytes = t.encode();
        let p = dir.path().join(format!("{name}.ggt"));
        std::fs::write(&p, &bytes).unwrap();
        inputs.insert(name.to_owned(), p);
        refs.push(ArtifactRef {
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        });
    }
    let def = matmul_definition(m, k, n, refs[0], refs[1]).unwrap();
    let task = validate_definition(&def).unwrap();
    Case { dir, inputs, task }
}

fn write_candidate(
    case: &Case,
    bytes: &[u8],
) -> (BTreeMap<String, PathBuf>, BTreeMap<String, ArtifactRef>) {
    let p = case.dir.path().join("candidate_c.ggt");
    std::fs::write(&p, bytes).unwrap();
    let mut files = BTreeMap::new();
    files.insert("c".to_owned(), p);
    let mut claimed = BTreeMap::new();
    claimed.insert(
        "c".to_owned(),
        ArtifactRef {
            hash: Sha256Digest::of_bytes(bytes),
            size: bytes.len() as u64,
        },
    );
    (files, claimed)
}

fn verdict(case: &Case, bytes: &[u8]) -> VerificationOutcome {
    let (files, claimed) = write_candidate(case, bytes);
    verify(&case.task, &case.inputs, &files, &claimed)
        .unwrap()
        .outcome
}

#[test]
fn execute_then_verify_accepts_and_naive_oracle_agrees() {
    let case = make_case(64, 96, 32, 7);
    let out = case.dir.path().join("out");
    let report = execute(EngineSelection::FlexCpu, &case.task, &case.inputs, &out).unwrap();
    assert_eq!(report.engine.numeric_profile, "cpu-f32-flex-v1");
    let c = &report.outputs["c"];
    let bytes = std::fs::read(&c.path).unwrap();
    assert_eq!(Sha256Digest::of_bytes(&bytes), c.hash);
    let mut files = BTreeMap::new();
    files.insert("c".to_owned(), c.path.clone());
    let mut claimed = BTreeMap::new();
    claimed.insert("c".to_owned(), c.artifact_ref());
    let v = verify(&case.task, &case.inputs, &files, &claimed).unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Accepted, "{:?}", v.checks);
    assert!(v.checks.iter().any(|k| k.name == "reference.recomputed"));
    // independent oracle
    let a = TensorF32::read_from_path(&case.inputs["a"], &Default::default()).unwrap();
    let b = TensorF32::read_from_path(&case.inputs["b"], &Default::default()).unwrap();
    let naive = matmul_naive_f64(&a, &b).encode();
    assert_eq!(verdict(&case, &naive), VerificationOutcome::Accepted);
}

#[test]
fn no_false_rejections_on_conformant_noise() {
    let mut false_rejections = 0;
    for seed in 0..20u64 {
        let case = make_case(24, 40, 16, 100 + seed);
        let a = TensorF32::read_from_path(&case.inputs["a"], &Default::default()).unwrap();
        let b = TensorF32::read_from_path(&case.inputs["b"], &Default::default()).unwrap();
        let mut c = matmul_naive_f64(&a, &b);
        let mut d = Drbg::new("noise", seed);
        for v in c.data.iter_mut() {
            // stay strictly within abs tolerance 1e-4 (rel adds margin)
            *v += 5e-5 * d.next_sym_f32();
        }
        if verdict(&case, &c.encode()) != VerificationOutcome::Accepted {
            false_rejections += 1;
        }
    }
    assert_eq!(false_rejections, 0);
}

#[test]
fn all_corruptions_are_detected() {
    let mut undetected = Vec::new();
    for seed in 0..10u64 {
        let case = make_case(20, 30, 12, 200 + seed);
        let a = TensorF32::read_from_path(&case.inputs["a"], &Default::default()).unwrap();
        let b = TensorF32::read_from_path(&case.inputs["b"], &Default::default()).unwrap();
        let good = matmul_naive_f64(&a, &b);
        let mut d = Drbg::new("corrupt", seed);

        // beyond tolerance everywhere
        let mut c = good.clone();
        for v in c.data.iter_mut() {
            *v += 5e-3 * (1.0 + d.next_unit_f32());
        }
        if verdict(&case, &c.encode()) == VerificationOutcome::Accepted {
            undetected.push(("noise_beyond_tolerance", seed));
        }
        // a single element off
        let mut c = good.clone();
        let idx = d.next_index(c.data.len() as u32) as usize;
        c.data[idx] += 0.01;
        if verdict(&case, &c.encode()) == VerificationOutcome::Accepted {
            undetected.push(("single_element", seed));
        }
        // NaN injected
        let mut c = good.clone();
        c.data[idx] = f32::NAN;
        if verdict(&case, &c.encode()) == VerificationOutcome::Accepted {
            undetected.push(("nan", seed));
        }
        // Inf injected
        let mut c = good.clone();
        c.data[idx] = f32::INFINITY;
        if verdict(&case, &c.encode()) == VerificationOutcome::Accepted {
            undetected.push(("inf", seed));
        }
        // wrong shape (transposed dims, same element count)
        let t = TensorF32::new(vec![12, 20], good.data.clone()).unwrap();
        if verdict(&case, &t.encode()) == VerificationOutcome::Accepted {
            undetected.push(("shape", seed));
        }
        // truncated file
        let bytes = good.encode();
        if verdict(&case, &bytes[..bytes.len() - 4]) == VerificationOutcome::Accepted {
            undetected.push(("truncated", seed));
        }
        // extra bytes
        let mut extra = bytes.clone();
        extra.extend_from_slice(&[0, 0, 0, 0]);
        if verdict(&case, &extra) == VerificationOutcome::Accepted {
            undetected.push(("extra_bytes", seed));
        }
        // all zeros
        let z = TensorF32::new(good.shape.clone(), vec![0.0; good.data.len()]).unwrap();
        if verdict(&case, &z.encode()) == VerificationOutcome::Accepted {
            undetected.push(("zeros", seed));
        }
        // claimed hash mismatch
        let (files, mut claimed) = write_candidate(&case, &bytes);
        claimed.get_mut("c").unwrap().hash = Sha256Digest::zero();
        if verify(&case.task, &case.inputs, &files, &claimed)
            .unwrap()
            .outcome
            == VerificationOutcome::Accepted
        {
            undetected.push(("hash_mismatch", seed));
        }
        // extra claimed output name
        let (files, mut claimed) = write_candidate(&case, &bytes);
        claimed.insert("d".into(), claimed["c"]);
        if verify(&case.task, &case.inputs, &files, &claimed)
            .unwrap()
            .outcome
            == VerificationOutcome::Accepted
        {
            undetected.push(("extra_output", seed));
        }
    }
    assert!(
        undetected.is_empty(),
        "undetected corruptions: {undetected:?}"
    );
}

#[test]
fn rejects_invalid_inputs_before_computing() {
    let case = make_case(4, 4, 4, 1);
    // Replace input a with a NaN-containing file of the right shape.
    let mut a = TensorF32::read_from_path(&case.inputs["a"], &Default::default()).unwrap();
    a.data[0] = f32::NAN;
    std::fs::write(&case.inputs["a"], a.encode()).unwrap();
    let err = execute(
        EngineSelection::FlexCpu,
        &case.task,
        &case.inputs,
        &case.dir.path().join("o"),
    )
    .unwrap_err();
    assert!(
        matches!(err, gg_compute::ComputeError::InputTensor { .. }),
        "{err}"
    );
}
