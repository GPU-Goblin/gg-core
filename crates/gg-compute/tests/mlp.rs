//! `mlp.train.v1`: real training on CPU, determinism, resume from checkpoint, verification and
//! perturbations.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_compute::catalog::{mlp_definition, validate_definition};
use gg_compute::engine::EngineSelection;
use gg_compute::mlp::{self, MlpTask, train};
use gg_compute::synthetic::SyntheticMlpConfig;
use gg_compute::{execute_task, verify_task};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::messages::{ArtifactRef, MlpTrainParamsV1};
use gg_protocol::tensor_file::TensorF32;

fn params(optimizer: &str, steps: u32, seed: u64) -> MlpTrainParamsV1 {
    MlpTrainParamsV1 {
        d_input: 8,
        d_hidden: 16,
        d_output: 2,
        optimizer: optimizer.into(),
        learning_rate: 0.05,
        batch_size: 32,
        steps,
        seed,
    }
}

struct Case {
    dir: tempfile::TempDir,
    inputs: BTreeMap<String, PathBuf>,
    task: gg_compute::ValidatedTask,
}

fn make_case(p: MlpTrainParamsV1) -> Case {
    let dir = tempfile::tempdir().unwrap();
    let ds = SyntheticMlpConfig {
        d_input: 8,
        d_output: 2,
        d_teacher: 16,
        n_train: 256,
        n_eval: 64,
        seed: 11,
        noise: 0.02,
    }
    .generate()
    .unwrap();
    let mut inputs = BTreeMap::new();
    let mut refs = BTreeMap::new();
    for (name, t) in [
        ("train_x", &ds.train_x),
        ("train_y", &ds.train_y),
        ("eval_x", &ds.eval_x),
        ("eval_y", &ds.eval_y),
    ] {
        let bytes = t.encode();
        let path = dir.path().join(format!("{name}.ggt"));
        std::fs::write(&path, &bytes).unwrap();
        inputs.insert(name.to_owned(), path);
        refs.insert(
            name.to_owned(),
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
    }
    let def = mlp_definition(p, refs).unwrap();
    let task = validate_definition(&def).unwrap();
    Case { dir, inputs, task }
}

fn data(case: &Case) -> mlp::Data {
    let rd = |n: &str| TensorF32::read_from_path(&case.inputs[n], &Default::default()).unwrap();
    mlp::Data {
        train_x: rd("train_x"),
        train_y: rd("train_y"),
        eval_x: rd("eval_x"),
        eval_y: rd("eval_y"),
    }
}

type Cpu = burn::backend::Autodiff<burn::backend::Flex>;

#[test]
fn training_reduces_loss_and_is_deterministic_on_cpu() {
    for opt in ["sgd", "adam"] {
        let case = make_case(params(opt, 200, 1));
        let spec = MlpTask::from_validated(&case.task).unwrap();
        let d = data(&case);
        let device = burn::backend::flex::FlexDevice;
        let (w_a, m_a) = train::<Cpu>(&device, &spec, &d, None, None, "cpu-f32-flex-v1").unwrap();
        let (w_b, m_b) = train::<Cpu>(&device, &spec, &d, None, None, "cpu-f32-flex-v1").unwrap();
        assert_eq!(w_a, w_b, "{opt}: two CPU runs must be bitwise identical");
        assert_eq!(m_a, m_b);
        assert_eq!(m_a.steps_completed, 200);
        let first = m_a.log.first().unwrap().train_loss;
        assert!(
            m_a.eval_loss < first,
            "{opt}: eval {} should be below the first batch loss {first}",
            m_a.eval_loss
        );
        let initial_eval = mlp::mse::<burn::backend::Flex>(
            &device,
            &mlp::initial_weights(&spec),
            &d.eval_x,
            &d.eval_y,
        )
        .unwrap();
        assert!(
            m_a.eval_loss < initial_eval * 0.5,
            "{opt}: eval loss {} vs initial {initial_eval}",
            m_a.eval_loss
        );
    }
}

#[test]
fn resume_from_checkpoint_matches_uninterrupted_run() {
    for opt in ["sgd", "adam"] {
        let case = make_case(params(opt, 160, 2));
        let spec = MlpTask::from_validated(&case.task).unwrap();
        let d = data(&case);
        let device = burn::backend::flex::FlexDevice;
        let (full, full_m) =
            train::<Cpu>(&device, &spec, &d, None, None, "cpu-f32-flex-v1").unwrap();
        let ckpt = case.dir.path().join("ckpt");
        // run 70 steps (checkpoints every 20), stop, then resume to completion
        let (_partial, pm) =
            train::<Cpu>(&device, &spec, &d, Some(&ckpt), Some(70), "cpu-f32-flex-v1").unwrap();
        assert_eq!(pm.steps_completed, 70);
        assert!(ckpt.join("state.json").exists());
        let (resumed, rm) =
            train::<Cpu>(&device, &spec, &d, Some(&ckpt), None, "cpu-f32-flex-v1").unwrap();
        assert_eq!(
            rm.resumed_from_step, 60,
            "{opt}: last checkpoint before step 70"
        );
        assert_eq!(rm.steps_completed, 160);
        assert_eq!(rm.log, full_m.log, "{opt}: loss log must match");
        assert_eq!(
            resumed, full,
            "{opt}: resumed weights must equal the uninterrupted run"
        );
    }
}

/// A process killed while it writes a checkpoint leaves the next one half written. Resuming must
/// ignore it and start again from the last complete checkpoint, never mixing the two.
#[test]
fn a_half_written_checkpoint_is_ignored_on_resume() {
    for opt in ["sgd", "adam"] {
        let case = make_case(params(opt, 160, 2));
        let spec = MlpTask::from_validated(&case.task).unwrap();
        let d = data(&case);
        let device = burn::backend::flex::FlexDevice;
        let (full, _) = train::<Cpu>(&device, &spec, &d, None, None, "cpu-f32-flex-v1").unwrap();
        let ckpt = case.dir.path().join("ckpt");
        train::<Cpu>(&device, &spec, &d, Some(&ckpt), Some(70), "cpu-f32-flex-v1").unwrap();
        // Killed while writing the checkpoint of step 80: the model is there, nothing else.
        let torn = ckpt.join("step-80");
        std::fs::create_dir_all(&torn).unwrap();
        std::fs::copy(
            ckpt.join("step-60").join("model.mpk"),
            torn.join("model.mpk"),
        )
        .unwrap();
        std::fs::write(torn.join("optim.mpk"), b"truncated").unwrap();
        let (resumed, rm) =
            train::<Cpu>(&device, &spec, &d, Some(&ckpt), None, "cpu-f32-flex-v1").unwrap();
        assert_eq!(rm.resumed_from_step, 60, "{opt}: last complete checkpoint");
        assert_eq!(
            resumed, full,
            "{opt}: resumed weights must equal the uninterrupted run"
        );
    }
}

fn write_candidates(
    _case: &Case,
    report: &gg_compute::ExecutionReport,
) -> (BTreeMap<String, PathBuf>, BTreeMap<String, ArtifactRef>) {
    let mut files = BTreeMap::new();
    let mut claimed = BTreeMap::new();
    for (name, f) in &report.outputs {
        files.insert(name.clone(), f.path.clone());
        claimed.insert(name.clone(), f.artifact_ref());
    }
    (files, claimed)
}

#[test]
fn execute_then_verify_accepts_and_perturbations_are_rejected() {
    let case = make_case(params("adam", 120, 3));
    let out = case.dir.path().join("out");
    let report = execute_task(EngineSelection::FlexCpu, &case.task, &case.inputs, &out).unwrap();
    assert_eq!(report.outputs.len(), 5);
    assert!(
        !out.join("checkpoint").exists(),
        "checkpoints are removed after success"
    );
    let (files, claimed) = write_candidates(&case, &report);
    let v = verify_task(
        &case.task,
        &case.inputs,
        &files,
        &claimed,
        "cpu-f32-flex-v1",
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Accepted, "{:?}", v.checks);
    assert!(
        v.checks
            .iter()
            .any(|c| c.name == "metrics.eval_loss" && c.passed)
    );
    assert!(
        v.checks
            .iter()
            .any(|c| c.name == "metrics.log.trajectory" && c.passed)
    );

    // The GPU profile is not admitted for this kind (see profiles::MLP_GPU_ADMITTED): inconclusive
    let v = verify_task(
        &case.task,
        &case.inputs,
        &files,
        &claimed,
        "gpu-f32-wgpu-vulkan-v1",
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Inconclusive);
    // A profile that is not admitted at all is inconclusive before any recomputation
    let v = verify_task(
        &case.task,
        &case.inputs,
        &files,
        &claimed,
        "cpu-f64-unknown",
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Inconclusive);
    assert_eq!(v.checks.len(), 1);

    let metrics_path = &report.outputs["metrics"].path;
    let good_metrics = std::fs::read(metrics_path).unwrap();
    let mut rejected = Vec::new();
    let mut check = |label: &str,
                     files: &BTreeMap<String, PathBuf>,
                     claimed: &BTreeMap<String, ArtifactRef>| {
        let v = verify_task(&case.task, &case.inputs, files, claimed, "cpu-f32-flex-v1").unwrap();
        if v.outcome == VerificationOutcome::Accepted {
            rejected.push(label.to_owned());
        }
    };
    // 1. one weight nudged beyond tolerance
    {
        let mut w1 =
            TensorF32::read_from_path(&report.outputs["w1"].path, &Default::default()).unwrap();
        w1.data[3] += 0.01;
        let bytes = w1.encode();
        let p = case.dir.path().join("w1-bad.ggt");
        std::fs::write(&p, &bytes).unwrap();
        let mut f = files.clone();
        let mut c = claimed.clone();
        f.insert("w1".into(), p);
        c.insert(
            "w1".into(),
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
        check("weight_nudged", &f, &c);
    }
    // 2. declared eval_loss lowered (metrics lie)
    {
        let mut m: mlp::MlpMetrics = serde_json::from_slice(&good_metrics).unwrap();
        m.eval_loss *= 0.5;
        let bytes = serde_json::to_vec(&m).unwrap();
        let p = case.dir.path().join("metrics-lie.json");
        std::fs::write(&p, &bytes).unwrap();
        let mut f = files.clone();
        let mut c = claimed.clone();
        f.insert("metrics".into(), p);
        c.insert(
            "metrics".into(),
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
        check("eval_loss_lie", &f, &c);
    }
    // 3. loss log tampered
    {
        let mut m: mlp::MlpMetrics = serde_json::from_slice(&good_metrics).unwrap();
        m.log[0].train_loss *= 0.1;
        let bytes = serde_json::to_vec(&m).unwrap();
        let p = case.dir.path().join("metrics-log.json");
        std::fs::write(&p, &bytes).unwrap();
        let mut f = files.clone();
        let mut c = claimed.clone();
        f.insert("metrics".into(), p);
        c.insert(
            "metrics".into(),
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
        check("log_tampered", &f, &c);
    }
    // 4. fewer steps than required
    {
        let mut m: mlp::MlpMetrics = serde_json::from_slice(&good_metrics).unwrap();
        m.steps_completed = 100;
        let bytes = serde_json::to_vec(&m).unwrap();
        let p = case.dir.path().join("metrics-steps.json");
        std::fs::write(&p, &bytes).unwrap();
        let mut f = files.clone();
        let mut c = claimed.clone();
        f.insert("metrics".into(), p);
        c.insert(
            "metrics".into(),
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
        check("steps_short", &f, &c);
    }
    // 5. weights from a different seed (honest work on the wrong task)
    {
        let other = make_case(params("adam", 120, 4));
        let out2 = other.dir.path().join("out");
        let r2 = execute_task(EngineSelection::FlexCpu, &other.task, &other.inputs, &out2).unwrap();
        let (f2, c2) = write_candidates(&other, &r2);
        check("wrong_seed", &f2, &c2);
    }
    // 6. missing output
    {
        let mut c = claimed.clone();
        c.remove("b2");
        check("missing_output", &files, &c);
    }
    assert!(rejected.is_empty(), "undetected corruptions: {rejected:?}");
}

#[test]
fn catalog_bounds_for_mlp() {
    let case = make_case(params("sgd", 10, 1));
    let mut refs = BTreeMap::new();
    for (name, path) in &case.inputs {
        let bytes = std::fs::read(path).unwrap();
        refs.insert(
            name.clone(),
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
    }
    let mut p = params("sgd", 10, 1);
    p.optimizer = "lion".into();
    assert!(mlp_definition(p, refs.clone()).is_err());
    let mut p = params("sgd", 10, 1);
    p.learning_rate = 2.0;
    assert!(mlp_definition(p, refs.clone()).is_err());
    let mut p = params("sgd", 10, 1);
    p.batch_size = 1000;
    assert!(
        mlp_definition(p, refs.clone()).is_err(),
        "batch larger than the dataset"
    );
    let mut p = params("sgd", 6000, 1);
    p.steps = 6000;
    assert!(mlp_definition(p, refs.clone()).is_err());
    let mut p = params("sgd", 10, 1);
    p.d_input = 9;
    assert!(
        mlp_definition(p, refs).is_err(),
        "input width must match the dataset"
    );
}
