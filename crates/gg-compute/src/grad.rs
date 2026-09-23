//! `mlp.grad.v1`: mean gradient of one explicit micro-batch from a given checkpoint, plus the
//! coordinator-side aggregation and optimizer step (ADR-0011).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::ElementConversion;
use burn::tensor::backend::AutodiffBackend;
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::{ArtifactRef, CheckV1, MlpGradParamsV1, TaskParamsV1};
use gg_protocol::tensor_file::TensorF32;
use serde::{Deserialize, Serialize};

use crate::catalog::{ValidatedTask, tensor_limits_for};
use crate::compare::compare_f32;
use crate::engine::{EngineInfo, EngineSelection, cpu_info};
use crate::error::ComputeError;
use crate::matmul::{ExecutionReport, OutputFile, Verdict};
use crate::mlp::{Mlp, Weights};

/// Compute kind id.
pub const KIND: &str = "mlp.grad.v1";
/// Gradient output names (same order as the weights).
pub const GRAD_NAMES: [&str; 4] = ["g_w1", "g_b1", "g_w2", "g_b2"];
/// Metadata output name.
pub const META_NAME: &str = "grad_meta";
/// Maximum size of `grad_meta.json`.
pub const MAX_META_BYTES: u64 = 4096;

/// Metadata of a gradient task (declared; recomputed by the validator).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradMeta {
    /// Kind.
    pub kind: String,
    /// Examples in the micro-batch.
    pub n_examples: u32,
    /// Mean squared error of the micro-batch at the checkpoint.
    pub loss: f32,
    /// Numeric profile.
    pub numeric_profile: String,
}

/// Index stream of a round (concatenated permutations of `0..total`).
pub fn round_indices(round_seed: u64, total: u64, start: u64, count: u32) -> Vec<usize> {
    let mut d = Drbg::new("mlp.grad.v1/order", round_seed);
    let mut out = Vec::with_capacity(count as usize);
    let mut pos: u64 = 0;
    let end = start + count as u64;
    loop {
        let perm = d.permutation(total as u32);
        for i in perm {
            if pos >= end {
                return out;
            }
            if pos >= start {
                out.push(i as usize);
            }
            pos += 1;
        }
    }
}

fn gather(t: &TensorF32, idx: &[usize]) -> TensorF32 {
    let width = t.shape[1] as usize;
    let mut data = Vec::with_capacity(idx.len() * width);
    for &i in idx {
        data.extend_from_slice(&t.data[i * width..(i + 1) * width]);
    }
    TensorF32::new(vec![idx.len() as u64, width as u64], data).expect("shape")
}

/// Mean gradient (and mean loss) of the MSE over `(x, y)` at `weights`, on backend `B`.
pub fn gradient<B: AutodiffBackend>(
    device: &B::Device,
    weights: &Weights,
    x: &TensorF32,
    y: &TensorF32,
) -> Result<(Weights, f32), ComputeError> {
    let model = Mlp::<B>::from_weights(device, weights);
    let xt = burn::tensor::Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(x.data.clone(), [x.shape[0] as usize, x.shape[1] as usize]),
        device,
    );
    let yt = burn::tensor::Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(y.data.clone(), [y.shape[0] as usize, y.shape[1] as usize]),
        device,
    );
    let pred = model.forward(xt);
    let loss = burn::nn::loss::MseLoss::new().forward(pred, yt, burn::nn::loss::Reduction::Mean);
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let grads = loss.backward();
    let gp = GradientsParams::from_grads(grads, &model);
    let g = |id, shape: &[u64]| -> Result<TensorF32, ComputeError> {
        let t: burn::tensor::Tensor<B::InnerBackend, 1> = match shape.len() {
            1 => gp
                .get::<B::InnerBackend, 1>(id)
                .ok_or_else(|| ComputeError::EngineUnavailable("missing gradient".into()))?,
            _ => {
                let t2: burn::tensor::Tensor<B::InnerBackend, 2> = gp
                    .get::<B::InnerBackend, 2>(id)
                    .ok_or_else(|| ComputeError::EngineUnavailable("missing gradient".into()))?;
                let data = t2.into_data();
                let values = data
                    .to_vec::<f32>()
                    .map_err(|e| ComputeError::EngineUnavailable(format!("dtype: {e:?}")))?;
                return TensorF32::new(shape.to_vec(), values)
                    .map_err(|e| ComputeError::EngineUnavailable(e.to_string()));
            }
        };
        let data = t.into_data();
        let values = data
            .to_vec::<f32>()
            .map_err(|e| ComputeError::EngineUnavailable(format!("dtype: {e:?}")))?;
        TensorF32::new(shape.to_vec(), values)
            .map_err(|e| ComputeError::EngineUnavailable(e.to_string()))
    };
    let out = Weights {
        w1: g(model.w1.id, &weights.w1.shape)?,
        b1: g(model.b1.id, &weights.b1.shape)?,
        w2: g(model.w2.id, &weights.w2.shape)?,
        b2: g(model.b2.id, &weights.b2.shape)?,
    };
    let _ = model.valid();
    Ok((out, loss_value))
}

/// Resolved task.
#[derive(Clone, Debug)]
pub struct GradTask {
    /// Parameters.
    pub params: MlpGradParamsV1,
}

impl GradTask {
    /// From a validated task.
    pub fn from_validated(task: &ValidatedTask) -> Result<Self, ComputeError> {
        match &task.params {
            TaskParamsV1::MlpGrad(p) => Ok(Self { params: p.clone() }),
            other => Err(ComputeError::UnknownKind(other.compute_kind().to_owned())),
        }
    }
}

fn load_input(
    task: &ValidatedTask,
    name: &str,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<TensorF32, ComputeError> {
    let shape = task
        .input_shapes
        .get(name)
        .ok_or_else(|| ComputeError::Inputs(format!("no expected shape for '{name}'")))?;
    let path = inputs
        .get(name)
        .ok_or_else(|| ComputeError::Inputs(format!("input file '{name}' not provided")))?;
    let t = TensorF32::read_from_path(path, &tensor_limits_for(shape)).map_err(|source| {
        ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        }
    })?;
    t.check_shape(shape)
        .map_err(|source| ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        })?;
    t.check_finite()
        .map_err(|source| ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        })?;
    Ok(t)
}

/// Load the checkpoint weights from the inputs.
pub fn load_weights(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<Weights, ComputeError> {
    Ok(Weights {
        w1: load_input(task, "w1", inputs)?,
        b1: load_input(task, "b1", inputs)?,
        w2: load_input(task, "w2", inputs)?,
        b2: load_input(task, "b2", inputs)?,
    })
}

/// Load the micro-batch of a task.
pub fn load_batch(
    task: &ValidatedTask,
    p: &MlpGradParamsV1,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<(TensorF32, TensorF32), ComputeError> {
    let x = load_input(task, "train_x", inputs)?;
    let y = load_input(task, "train_y", inputs)?;
    let idx = round_indices(
        p.round_seed,
        p.total_examples,
        p.example_start,
        p.example_count,
    );
    Ok((gather(&x, &idx), gather(&y, &idx)))
}

fn write_file(
    out_dir: &Path,
    file_name: &str,
    bytes: &[u8],
    max_bytes: u64,
) -> Result<OutputFile, ComputeError> {
    if bytes.len() as u64 > max_bytes {
        return Err(ComputeError::OutputLimit(format!(
            "output {file_name} is {} bytes > {max_bytes}",
            bytes.len()
        )));
    }
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join(file_name);
    let tmp = out_dir.join(format!(".{file_name}.tmp"));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(OutputFile {
        path,
        hash: Sha256Digest::of_bytes(bytes),
        size: bytes.len() as u64,
    })
}

/// Execute a gradient task.
pub fn execute(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
) -> Result<ExecutionReport, ComputeError> {
    let g = GradTask::from_validated(task)?;
    let weights = load_weights(task, inputs)?;
    let (x, y) = load_batch(task, &g.params, inputs)?;
    let start = Instant::now();
    let (grads, loss, info): (Weights, f32, EngineInfo) = match engine {
        EngineSelection::FlexCpu => {
            let device = burn::backend::flex::FlexDevice;
            let (gr, l) = gradient::<burn::backend::Autodiff<burn::backend::Flex>>(
                &device, &weights, &x, &y,
            )?;
            (gr, l, cpu_info())
        }
        #[cfg(feature = "gpu")]
        EngineSelection::WgpuVulkan { device_index } => {
            let (device, info) = crate::engine::gpu::init_vulkan(device_index)?;
            let (gr, l) = gradient::<burn::backend::Autodiff<burn::backend::Vulkan>>(
                &device, &weights, &x, &y,
            )?;
            (gr, l, info)
        }
        EngineSelection::TorchCuda { .. } => return Err(crate::engine::cuda_models_only()),
        #[cfg(not(feature = "gpu"))]
        EngineSelection::WgpuVulkan { .. } => {
            return Err(ComputeError::EngineUnavailable(
                "this build has no GPU support (feature `gpu`)".into(),
            ));
        }
    };
    let duration_ms = start.elapsed().as_millis() as u64;
    let mut outputs = BTreeMap::new();
    let mut budget = task.limits.max_output_bytes;
    for (name, t) in GRAD_NAMES
        .iter()
        .zip([&grads.w1, &grads.b1, &grads.w2, &grads.b2])
    {
        t.check_finite()
            .map_err(|e| ComputeError::OutputLimit(format!("non-finite gradient: {e}")))?;
        let bytes = t.encode();
        let f = write_file(out_dir, &format!("{name}.ggt"), &bytes, budget)?;
        budget = budget.saturating_sub(f.size);
        outputs.insert((*name).to_owned(), f);
    }
    let meta = GradMeta {
        kind: KIND.into(),
        n_examples: g.params.example_count,
        loss,
        numeric_profile: info.numeric_profile.to_owned(),
    };
    let meta_bytes =
        serde_json::to_vec_pretty(&meta).map_err(|e| ComputeError::Io(std::io::Error::other(e)))?;
    let f = write_file(
        out_dir,
        "grad_meta.json",
        &meta_bytes,
        MAX_META_BYTES.min(budget),
    )?;
    outputs.insert(META_NAME.to_owned(), f);
    let mut metrics = BTreeMap::new();
    metrics.insert("duration_ms".to_owned(), duration_ms as f64);
    metrics.insert("loss".to_owned(), loss as f64);
    Ok(ExecutionReport {
        outputs,
        engine: info,
        duration_ms,
        metrics,
    })
}

fn fail(name: &str, detail: String) -> CheckV1 {
    CheckV1 {
        name: name.to_owned(),
        passed: false,
        measured: None,
        tolerance: None,
        detail: detail.chars().take(200).collect(),
    }
}

fn pass(name: &str, detail: String) -> CheckV1 {
    CheckV1 {
        name: name.to_owned(),
        passed: true,
        measured: None,
        tolerance: None,
        detail: detail.chars().take(200).collect(),
    }
}

/// Verify candidate gradients by recomputing them on the CPU reference profile.
pub fn verify(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
    declared_profile: &str,
) -> Result<Verdict, ComputeError> {
    let g = GradTask::from_validated(task)?;
    let mut checks = Vec::new();
    let mut artifacts_checked = Vec::new();
    let reject = |checks: Vec<CheckV1>, a: Vec<ArtifactRef>| {
        Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Rejected,
            artifacts_checked: a,
        })
    };
    let mut expected: Vec<&str> = GRAD_NAMES.to_vec();
    expected.push(META_NAME);
    expected.sort_unstable();
    let claimed_names: Vec<&str> = claimed.keys().map(String::as_str).collect();
    if expected != claimed_names {
        checks.push(fail(
            "outputs.names",
            format!("expected {expected:?}, got {claimed_names:?}"),
        ));
        return reject(checks, artifacts_checked);
    }
    let mut bytes_by_name: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    for name in expected.iter() {
        let r = claimed[*name];
        let max = if *name == META_NAME {
            MAX_META_BYTES
        } else {
            crate::catalog::expected_rgt_size(&task.output_shapes[*name])
        };
        let Some(path) = candidates.get(*name) else {
            checks.push(fail(
                &format!("outputs.{name}.present"),
                "candidate file missing".into(),
            ));
            return Ok(Verdict {
                checks,
                outcome: VerificationOutcome::Inconclusive,
                artifacts_checked,
            });
        };
        let meta = std::fs::metadata(path)?;
        if meta.len() != r.size || meta.len() > max {
            checks.push(fail(
                &format!("outputs.{name}.size"),
                format!("claimed {}, file {}, max {max}", r.size, meta.len()),
            ));
            return reject(checks, artifacts_checked);
        }
        let bytes = std::fs::read(path)?;
        if Sha256Digest::of_bytes(&bytes) != r.hash {
            checks.push(fail(
                &format!("outputs.{name}.sha256"),
                "hash mismatch".into(),
            ));
            return reject(checks, artifacts_checked);
        }
        checks.push(pass(&format!("outputs.{name}.sha256"), String::new()));
        artifacts_checked.push(r);
        bytes_by_name.insert(name, bytes);
    }
    let mut decoded: BTreeMap<&str, TensorF32> = BTreeMap::new();
    for name in GRAD_NAMES {
        match TensorF32::decode(
            &bytes_by_name[name],
            &tensor_limits_for(&task.output_shapes[name]),
        ) {
            Ok(t) => {
                decoded.insert(name, t);
            }
            Err(e) => {
                checks.push(fail(&format!("outputs.{name}.format"), e.to_string()));
                return reject(checks, artifacts_checked);
            }
        }
    }
    let meta: GradMeta = match serde_json::from_slice(&bytes_by_name[META_NAME]) {
        Ok(m) => m,
        Err(e) => {
            checks.push(fail("outputs.grad_meta.format", e.to_string()));
            return reject(checks, artifacts_checked);
        }
    };
    if meta.kind != KIND || meta.n_examples != g.params.example_count || !meta.loss.is_finite() {
        checks.push(fail(
            "outputs.grad_meta.structure",
            format!("n_examples {}, kind {}", meta.n_examples, meta.kind),
        ));
        return reject(checks, artifacts_checked);
    }
    if !crate::profiles::admitted_for(KIND, declared_profile)
        || meta.numeric_profile != declared_profile
    {
        checks.push(fail(
            "profile.admitted",
            format!(
                "declared profile '{declared_profile}' (meta: '{}') is not admitted for {KIND}",
                meta.numeric_profile
            ),
        ));
        return Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Inconclusive,
            artifacts_checked,
        });
    }
    checks.push(pass("profile.admitted", declared_profile.to_owned()));
    let weights = load_weights(task, inputs)?;
    let (x, y) = load_batch(task, &g.params, inputs)?;
    let device = burn::backend::flex::FlexDevice;
    let (reference, ref_loss) =
        gradient::<burn::backend::Autodiff<burn::backend::Flex>>(&device, &weights, &x, &y)?;
    checks.push(pass(
        "reference.recomputed",
        format!(
            "profile {} over {} examples",
            crate::profiles::REFERENCE_PROFILE,
            g.params.example_count
        ),
    ));
    let mut all_ok = true;
    for (name, r) in
        GRAD_NAMES
            .iter()
            .zip([&reference.w1, &reference.b1, &reference.w2, &reference.b2])
    {
        let cmp = compare_f32(
            &format!("outputs.{name}"),
            &decoded[name],
            r,
            task.tolerance,
        );
        all_ok &= cmp.passed;
        checks.extend(cmp.checks);
    }
    let diff = (meta.loss as f64 - ref_loss as f64).abs();
    let tol = task.tolerance.abs + task.tolerance.rel * (ref_loss as f64).abs();
    let ok = diff <= tol;
    all_ok &= ok;
    checks.push(CheckV1 {
        name: "grad_meta.loss".into(),
        passed: ok,
        measured: Some(diff),
        tolerance: Some(tol),
        detail: format!("declared {}, recomputed {ref_loss}", meta.loss),
    });
    Ok(Verdict {
        checks,
        outcome: if all_ok {
            VerificationOutcome::Accepted
        } else {
            VerificationOutcome::Rejected
        },
        artifacts_checked,
    })
}

/// Optimizer state persisted between rounds (host-side, fully specified; ADR-0011).
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerState {
    /// Adam first moments (empty for SGD).
    pub m: Option<Weights>,
    /// Adam second moments (empty for SGD).
    pub v: Option<Weights>,
    /// Steps applied so far.
    pub t: u32,
}

impl OptimizerState {
    /// Fresh state.
    pub fn new() -> Self {
        Self {
            m: None,
            v: None,
            t: 0,
        }
    }
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimizer choice for the coordinator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordOptim {
    /// Plain SGD.
    Sgd,
    /// Adam (β1 0.9, β2 0.999, ε 1e-8, bias-corrected), `adam-host-v1`.
    Adam,
}

impl CoordOptim {
    /// Parse.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sgd" => Some(Self::Sgd),
            "adam" => Some(Self::Adam),
            _ => None,
        }
    }
}

fn zeros_like(w: &Weights) -> Weights {
    let z =
        |t: &TensorF32| TensorF32::new(t.shape.clone(), vec![0.0; t.data.len()]).expect("shape");
    Weights {
        w1: z(&w.w1),
        b1: z(&w.b1),
        w2: z(&w.w2),
        b2: z(&w.b2),
    }
}

fn for_each4(a: &mut Weights, b: &Weights, mut f: impl FnMut(&mut f32, f32)) {
    for (x, y) in [
        (&mut a.w1, &b.w1),
        (&mut a.b1, &b.b1),
        (&mut a.w2, &b.w2),
        (&mut a.b2, &b.b2),
    ] {
        for (xi, yi) in x.data.iter_mut().zip(y.data.iter()) {
            f(xi, *yi);
        }
    }
}

/// Weighted mean of per-task mean gradients: `Σ n_i·g_i / Σ n_i`. Fails on empty input or shape
/// mismatch. Summation order is the order of `parts` (the coordinator sorts by slice start).
pub fn aggregate(parts: &[(u32, Weights)]) -> Result<Weights, ComputeError> {
    let (first_n, first) = parts
        .first()
        .ok_or_else(|| ComputeError::InvalidParams("no gradients to aggregate".into()))?;
    let _ = first_n;
    let mut acc = zeros_like(first);
    let mut total: f64 = 0.0;
    for (n, g) in parts {
        if g.w1.shape != first.w1.shape
            || g.w2.shape != first.w2.shape
            || g.b1.shape != first.b1.shape
            || g.b2.shape != first.b2.shape
        {
            return Err(ComputeError::InvalidParams("gradient shapes differ".into()));
        }
        let w = *n as f32;
        for_each4(&mut acc, g, |a, gi| *a += w * gi);
        total += *n as f64;
    }
    if total == 0.0 {
        return Err(ComputeError::InvalidParams("zero examples".into()));
    }
    let inv = (1.0 / total) as f32;
    for t in [&mut acc.w1, &mut acc.b1, &mut acc.w2, &mut acc.b2] {
        for v in t.data.iter_mut() {
            *v *= inv;
        }
    }
    Ok(acc)
}

/// Apply one optimizer step to `weights` with gradient `g` (host arithmetic, deterministic).
pub fn apply_step(
    optim: CoordOptim,
    lr: f64,
    weights: &mut Weights,
    g: &Weights,
    state: &mut OptimizerState,
) {
    state.t += 1;
    match optim {
        CoordOptim::Sgd => {
            let lr = lr as f32;
            for_each4(weights, g, |w, gi| *w -= lr * gi);
        }
        CoordOptim::Adam => {
            let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
            let m = state.m.get_or_insert_with(|| zeros_like(weights));
            let v = state.v.get_or_insert_with(|| zeros_like(weights));
            for_each4(m, g, |mi, gi| *mi = b1 * *mi + (1.0 - b1) * gi);
            for_each4(v, g, |vi, gi| *vi = b2 * *vi + (1.0 - b2) * gi * gi);
            let t = state.t as i32;
            let c1 = 1.0 - b1.powi(t);
            let c2 = 1.0 - b2.powi(t);
            let lr = lr as f32;
            for (w, (mi, vi)) in [
                (&mut weights.w1, (&m.w1, &v.w1)),
                (&mut weights.b1, (&m.b1, &v.b1)),
                (&mut weights.w2, (&m.w2, &v.w2)),
                (&mut weights.b2, (&m.b2, &v.b2)),
            ] {
                for ((wi, m_i), v_i) in w.data.iter_mut().zip(mi.data.iter()).zip(vi.data.iter()) {
                    let m_hat = m_i / c1;
                    let v_hat = v_i / c2;
                    *wi -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
    }
}

/// Reference single-process training with the collective rules: for each round, the full batch
/// gradient (one Burn call over all `batch` examples of the round's index stream) then one
/// `apply_step`. Used by the equivalence test and by `gg job collective-check`.
pub fn reference_training(
    initial: &Weights,
    optim: CoordOptim,
    lr: f64,
    train_x: &TensorF32,
    train_y: &TensorF32,
    rounds: &[(u64, u64)], // (round_seed, batch)
) -> Result<Weights, ComputeError> {
    let device = burn::backend::flex::FlexDevice;
    let mut w = initial.clone();
    let mut state = OptimizerState::new();
    let total = train_x.shape[0];
    for (seed, batch) in rounds {
        let idx = round_indices(*seed, total, 0, *batch as u32);
        let x = gather(train_x, &idx);
        let y = gather(train_y, &idx);
        let (g, _) = gradient::<burn::backend::Autodiff<burn::backend::Flex>>(&device, &w, &x, &y)?;
        apply_step(optim, lr, &mut w, &g, &mut state);
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp::{MlpTask, Optim, initial_weights};
    use crate::synthetic::SyntheticMlpConfig;
    use gg_protocol::messages::ToleranceV1;

    fn setup() -> (Weights, TensorF32, TensorF32) {
        let ds = SyntheticMlpConfig {
            d_input: 6,
            d_output: 2,
            d_teacher: 8,
            n_train: 512,
            n_eval: 16,
            seed: 21,
            noise: 0.02,
        }
        .generate()
        .unwrap();
        let task = MlpTask {
            d_in: 6,
            d_h: 12,
            d_out: 2,
            optimizer: Optim::Sgd,
            lr: 0.05,
            batch: 64,
            steps: 1,
            seed: 5,
            n_train: 512,
            n_eval: 16,
        };
        (initial_weights(&task), ds.train_x, ds.train_y)
    }

    #[test]
    fn micro_batch_slices_cover_the_round_exactly_once() {
        let a = round_indices(3, 100, 0, 40);
        let b = round_indices(3, 100, 40, 40);
        let c = round_indices(3, 100, 80, 20);
        let mut all: Vec<usize> = a.iter().chain(&b).chain(&c).copied().collect();
        all.sort_unstable();
        assert_eq!(
            all,
            (0..100).collect::<Vec<_>>(),
            "one epoch covered exactly once"
        );
        let wrap = round_indices(3, 100, 90, 20);
        assert_eq!(wrap.len(), 20, "slices may span into the next permutation");
    }

    #[test]
    fn aggregated_micro_batch_gradients_match_the_full_batch_gradient() {
        let (w, x, y) = setup();
        let device = burn::backend::flex::FlexDevice;
        let batch = 128u64;
        let idx = round_indices(7, 512, 0, batch as u32);
        let (full, _) = gradient::<burn::backend::Autodiff<burn::backend::Flex>>(
            &device,
            &w,
            &gather(&x, &idx),
            &gather(&y, &idx),
        )
        .unwrap();
        let mut parts = Vec::new();
        for (start, n) in [(0u64, 32u32), (32, 48), (80, 48)] {
            let idx = round_indices(7, 512, start, n);
            let (g, _) = gradient::<burn::backend::Autodiff<burn::backend::Flex>>(
                &device,
                &w,
                &gather(&x, &idx),
                &gather(&y, &idx),
            )
            .unwrap();
            parts.push((n, g));
        }
        let agg = aggregate(&parts).unwrap();
        for ((name, a), b) in GRAD_NAMES
            .iter()
            .zip([&agg.w1, &agg.b1, &agg.w2, &agg.b2])
            .zip([&full.w1, &full.b1, &full.w2, &full.b2])
        {
            let cmp = compare_f32(
                name,
                a,
                b,
                ToleranceV1 {
                    abs: 1e-6,
                    rel: 1e-5,
                },
            );
            assert!(cmp.passed, "{name}: {:?}", cmp.checks);
        }
    }

    #[test]
    fn adam_and_sgd_steps_are_deterministic() {
        let (w, x, y) = setup();
        let device = burn::backend::flex::FlexDevice;
        let idx = round_indices(1, 512, 0, 64);
        let (g, _) = gradient::<burn::backend::Autodiff<burn::backend::Flex>>(
            &device,
            &w,
            &gather(&x, &idx),
            &gather(&y, &idx),
        )
        .unwrap();
        for optim in [CoordOptim::Sgd, CoordOptim::Adam] {
            let mut a = w.clone();
            let mut b = w.clone();
            let mut sa = OptimizerState::new();
            let mut sb = OptimizerState::new();
            apply_step(optim, 0.01, &mut a, &g, &mut sa);
            apply_step(optim, 0.01, &mut b, &g, &mut sb);
            assert_eq!(a, b);
            assert_ne!(a, w, "{optim:?} must change the weights");
        }
    }
}
