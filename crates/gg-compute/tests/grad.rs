//! `mlp.grad.v1` execution/verification, perturbations, and the collective-vs-reference equivalence.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_compute::catalog::{grad_definition, validate_definition};
use gg_compute::engine::EngineSelection;
use gg_compute::grad::{
    self, CoordOptim, OptimizerState, aggregate, apply_step, reference_training,
};
use gg_compute::mlp::{MlpTask, Optim, Weights, initial_weights};
use gg_compute::synthetic::SyntheticMlpConfig;
use gg_compute::{execute_task, verify_task};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::messages::{ArtifactRef, MlpGradParamsV1, ToleranceV1};
use gg_protocol::tensor_file::TensorF32;

const D_IN: u32 = 8;
const D_H: u32 = 16;
const D_OUT: u32 = 2;
const N: u32 = 1024;

fn dataset() -> (TensorF32, TensorF32) {
    let ds = SyntheticMlpConfig {
        d_input: D_IN,
        d_output: D_OUT,
        d_teacher: 16,
        n_train: N,
        n_eval: 8,
        seed: 31,
        noise: 0.02,
    }
    .generate()
    .unwrap();
    (ds.train_x, ds.train_y)
}

fn initial() -> Weights {
    initial_weights(&MlpTask {
        d_in: D_IN as usize,
        d_h: D_H as usize,
        d_out: D_OUT as usize,
        optimizer: Optim::Sgd,
        lr: 0.0,
        batch: 1,
        steps: 1,
        seed: 77,
        n_train: N as usize,
        n_eval: 8,
    })
}

struct Case {
    dir: tempfile::TempDir,
    inputs: BTreeMap<String, PathBuf>,
    task: gg_compute::ValidatedTask,
}

fn make_case(
    w: &Weights,
    x: &TensorF32,
    y: &TensorF32,
    round_seed: u64,
    start: u64,
    count: u32,
) -> Case {
    let dir = tempfile::tempdir().unwrap();
    let mut inputs = BTreeMap::new();
    let mut refs = BTreeMap::new();
    for (name, t) in [
        ("w1", &w.w1),
        ("b1", &w.b1),
        ("w2", &w.w2),
        ("b2", &w.b2),
        ("train_x", x),
        ("train_y", y),
    ] {
        let bytes = t.encode();
        let p = dir.path().join(format!("{name}.ggt"));
        std::fs::write(&p, &bytes).unwrap();
        inputs.insert(name.to_owned(), p);
        refs.insert(
            name.to_owned(),
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
    }
    let def = grad_definition(
        MlpGradParamsV1 {
            d_input: D_IN,
            d_hidden: D_H,
            d_output: D_OUT,
            round_seed,
            example_start: start,
            example_count: count,
            total_examples: N as u64,
        },
        refs,
    )
    .unwrap();
    let task = validate_definition(&def).unwrap();
    Case { dir, inputs, task }
}

#[test]
fn execute_verify_and_perturbations() {
    let (x, y) = dataset();
    let w = initial();
    let case = make_case(&w, &x, &y, 5, 64, 32);
    let out = case.dir.path().join("out");
    let report = execute_task(EngineSelection::FlexCpu, &case.task, &case.inputs, &out).unwrap();
    assert_eq!(report.outputs.len(), 5);
    let mut files = BTreeMap::new();
    let mut claimed = BTreeMap::new();
    for (n, f) in &report.outputs {
        files.insert(n.clone(), f.path.clone());
        claimed.insert(n.clone(), f.artifact_ref());
    }
    let v = verify_task(
        &case.task,
        &case.inputs,
        &files,
        &claimed,
        "cpu-f32-flex-v1",
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Accepted, "{:?}", v.checks);
    assert_eq!(
        verify_task(
            &case.task,
            &case.inputs,
            &files,
            &claimed,
            "gpu-f32-wgpu-vulkan-v1"
        )
        .unwrap()
        .outcome,
        VerificationOutcome::Inconclusive
    );
    // gradient tampered
    let mut g =
        TensorF32::read_from_path(&report.outputs["g_w1"].path, &Default::default()).unwrap();
    g.data[0] += 1e-3;
    let bytes = g.encode();
    let p = case.dir.path().join("bad.ggt");
    std::fs::write(&p, &bytes).unwrap();
    let mut f2 = files.clone();
    let mut c2 = claimed.clone();
    f2.insert("g_w1".into(), p);
    c2.insert(
        "g_w1".into(),
        ArtifactRef {
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        },
    );
    assert_eq!(
        verify_task(&case.task, &case.inputs, &f2, &c2, "cpu-f32-flex-v1")
            .unwrap()
            .outcome,
        VerificationOutcome::Rejected
    );
    // gradient of another slice (honest work on the wrong micro-batch)
    let other = make_case(&w, &x, &y, 5, 96, 32);
    let r2 = execute_task(
        EngineSelection::FlexCpu,
        &other.task,
        &other.inputs,
        &other.dir.path().join("out"),
    )
    .unwrap();
    let mut f3 = BTreeMap::new();
    let mut c3 = BTreeMap::new();
    for (n, f) in &r2.outputs {
        f3.insert(n.clone(), f.path.clone());
        c3.insert(n.clone(), f.artifact_ref());
    }
    assert_eq!(
        verify_task(&case.task, &case.inputs, &f3, &c3, "cpu-f32-flex-v1")
            .unwrap()
            .outcome,
        VerificationOutcome::Rejected
    );
    // declared loss lied
    let mut meta: grad::GradMeta =
        serde_json::from_slice(&std::fs::read(&report.outputs["grad_meta"].path).unwrap()).unwrap();
    meta.loss *= 0.5;
    let bytes = serde_json::to_vec(&meta).unwrap();
    let p = case.dir.path().join("meta.json");
    std::fs::write(&p, &bytes).unwrap();
    let mut f4 = files.clone();
    let mut c4 = claimed.clone();
    f4.insert("grad_meta".into(), p);
    c4.insert(
        "grad_meta".into(),
        ArtifactRef {
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        },
    );
    assert_eq!(
        verify_task(&case.task, &case.inputs, &f4, &c4, "cpu-f32-flex-v1")
            .unwrap()
            .outcome,
        VerificationOutcome::Rejected
    );
}

/// Simulates the coordinator in-process: R rounds × K micro-batch tasks executed through the
/// catalog, aggregated with explicit weighting, one optimizer step per round; compared with the
/// single-process reference using the same effective batch, order, precision and rules.
#[test]
fn collective_rounds_match_the_reference_training() {
    let (x, y) = dataset();
    let w0 = initial();
    let batch = 256u32;
    let k = 4u32;
    let rounds = 6u64;
    for optim in [CoordOptim::Sgd, CoordOptim::Adam] {
        let lr = if optim == CoordOptim::Sgd { 0.05 } else { 0.01 };
        let mut w = w0.clone();
        let mut state = OptimizerState::new();
        let mut seeds = Vec::new();
        for step in 0..rounds {
            let seed = 1000 + step;
            seeds.push((seed, batch as u64));
            let per = batch / k;
            let mut parts = Vec::new();
            for i in 0..k {
                let case = make_case(&w, &x, &y, seed, (i * per) as u64, per);
                let out = case.dir.path().join("out");
                let report =
                    execute_task(EngineSelection::FlexCpu, &case.task, &case.inputs, &out).unwrap();
                let rd = |n: &str| {
                    TensorF32::read_from_path(&report.outputs[n].path, &Default::default()).unwrap()
                };
                let meta: grad::GradMeta = serde_json::from_slice(
                    &std::fs::read(&report.outputs["grad_meta"].path).unwrap(),
                )
                .unwrap();
                assert_eq!(meta.n_examples, per);
                parts.push((
                    per,
                    Weights {
                        w1: rd("g_w1"),
                        b1: rd("g_b1"),
                        w2: rd("g_w2"),
                        b2: rd("g_b2"),
                    },
                ));
            }
            let g = aggregate(&parts).unwrap();
            apply_step(optim, lr, &mut w, &g, &mut state);
        }
        let reference = reference_training(&w0, optim, lr, &x, &y, &seeds).unwrap();
        let tol = ToleranceV1 {
            abs: 1e-5,
            rel: 1e-4,
        };
        for ((name, a), b) in ["w1", "b1", "w2", "b2"]
            .iter()
            .zip([&w.w1, &w.b1, &w.w2, &w.b2])
            .zip([&reference.w1, &reference.b1, &reference.w2, &reference.b2])
        {
            let cmp = gg_compute::compare::compare_f32(name, a, b, tol);
            let max_abs = cmp
                .checks
                .iter()
                .find(|c| c.name.ends_with("max_abs_diff"))
                .and_then(|c| c.measured)
                .unwrap_or(0.0);
            eprintln!("{optim:?} {name}: max_abs_diff {max_abs:.3e}");
            assert!(cmp.passed, "{optim:?} {name}: {:?}", cmp.checks);
        }
        assert_ne!(w, w0, "training must move the weights");
    }
}
