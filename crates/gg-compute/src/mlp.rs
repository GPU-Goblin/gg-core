//! `mlp.train.v1`: reproducible training of a small dense network (ADR-0008).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::module::{AutodiffModule, Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig};
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Record, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::{ArtifactRef, CheckV1, MlpTrainParamsV1, TaskParamsV1, ToleranceV1};
use gg_protocol::tensor_file::TensorF32;
use serde::{Deserialize, Serialize};

use crate::catalog::{ValidatedTask, tensor_limits_for};
use crate::compare::compare_f32;
use crate::engine::{EngineInfo, EngineSelection, cpu_info};
use crate::error::ComputeError;
use crate::matmul::{ExecutionReport, OutputFile, Verdict};

/// Compute kind id.
pub const KIND: &str = "mlp.train.v1";
/// Maximum size of `metrics.json`.
pub const MAX_METRICS_BYTES: u64 = 64 * 1024;
/// Maximum entries in the loss log.
pub const MAX_LOG_ENTRIES: usize = 64;
/// Tensor output names in order.
pub const WEIGHT_NAMES: [&str; 4] = ["w1", "b1", "w2", "b2"];
/// Metrics output name.
pub const METRICS_NAME: &str = "metrics";

/// Optimizer choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Optim {
    /// Plain SGD.
    Sgd,
    /// Adam with Burn 0.21 defaults (`adam-burn-0.21-v1`).
    Adam,
}

impl Optim {
    /// Parse.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sgd" => Some(Self::Sgd),
            "adam" => Some(Self::Adam),
            _ => None,
        }
    }
}

/// Fully resolved training task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlpTask {
    /// Input width.
    pub d_in: usize,
    /// Hidden width.
    pub d_h: usize,
    /// Output width.
    pub d_out: usize,
    /// Optimizer.
    pub optimizer: Optim,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch: usize,
    /// Steps.
    pub steps: u32,
    /// Seed.
    pub seed: u64,
    /// Training examples.
    pub n_train: usize,
    /// Evaluation examples.
    pub n_eval: usize,
}

impl MlpTask {
    /// Resolve from a validated task.
    pub fn from_validated(task: &ValidatedTask) -> Result<Self, ComputeError> {
        let TaskParamsV1::MlpTrain(p) = &task.params else {
            return Err(ComputeError::UnknownKind(
                task.params.compute_kind().to_owned(),
            ));
        };
        let n_train = task
            .input_shapes
            .get("train_x")
            .and_then(|s| s.first().copied())
            .unwrap_or(0) as usize;
        let n_eval = task
            .input_shapes
            .get("eval_x")
            .and_then(|s| s.first().copied())
            .unwrap_or(0) as usize;
        Self::from_params(p, n_train, n_eval)
    }

    /// Resolve from parameters and dataset sizes.
    pub fn from_params(
        p: &MlpTrainParamsV1,
        n_train: usize,
        n_eval: usize,
    ) -> Result<Self, ComputeError> {
        let optimizer = Optim::parse(&p.optimizer).ok_or_else(|| {
            ComputeError::InvalidParams(format!("unknown optimizer {}", p.optimizer))
        })?;
        Ok(Self {
            d_in: p.d_input as usize,
            d_h: p.d_hidden as usize,
            d_out: p.d_output as usize,
            optimizer,
            lr: p.learning_rate,
            batch: p.batch_size as usize,
            steps: p.steps,
            seed: p.seed,
            n_train,
            n_eval,
        })
    }

    /// Log interval (at most [`MAX_LOG_ENTRIES`] entries).
    pub fn log_every(&self) -> u32 {
        (self.steps as usize).div_ceil(MAX_LOG_ENTRIES).max(1) as u32
    }

    /// Checkpoint interval (about 8 checkpoints per run).
    pub fn checkpoint_every(&self) -> u32 {
        (self.steps / 8).max(1)
    }

    /// Identity of the task for checkpoint compatibility.
    pub fn identity(&self) -> Sha256Digest {
        Sha256Digest::of_bytes(serde_json::to_string(self).unwrap_or_default().as_bytes())
    }
}

/// Model weights on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    /// `[d_in, d_h]`.
    pub w1: TensorF32,
    /// `[d_h]`.
    pub b1: TensorF32,
    /// `[d_h, d_out]`.
    pub w2: TensorF32,
    /// `[d_out]`.
    pub b2: TensorF32,
}

impl Weights {
    /// Named tensors in protocol order.
    pub fn named(&self) -> [(&'static str, &TensorF32); 4] {
        [
            ("w1", &self.w1),
            ("b1", &self.b1),
            ("w2", &self.w2),
            ("b2", &self.b2),
        ]
    }
}

/// Protocol-fixed initial weights (ADR-0008).
pub fn initial_weights(task: &MlpTask) -> Weights {
    let mut d = Drbg::new("mlp.train.v1/init", task.seed);
    let s1 = 1.0 / (task.d_in as f32).sqrt();
    let w1: Vec<f32> = (0..task.d_in * task.d_h)
        .map(|_| d.next_sym_f32() * s1)
        .collect();
    let s2 = 1.0 / (task.d_h as f32).sqrt();
    let w2: Vec<f32> = (0..task.d_h * task.d_out)
        .map(|_| d.next_sym_f32() * s2)
        .collect();
    Weights {
        w1: TensorF32::new(vec![task.d_in as u64, task.d_h as u64], w1).expect("shape"),
        b1: TensorF32::new(vec![task.d_h as u64], vec![0.0; task.d_h]).expect("shape"),
        w2: TensorF32::new(vec![task.d_h as u64, task.d_out as u64], w2).expect("shape"),
        b2: TensorF32::new(vec![task.d_out as u64], vec![0.0; task.d_out]).expect("shape"),
    }
}

/// Deterministic stream of example indices: concatenated per-epoch permutations.
pub struct IndexStream {
    drbg: Drbg,
    n: u32,
    perm: Vec<u32>,
    pos: usize,
}

impl IndexStream {
    /// New stream for a task.
    pub fn new(task: &MlpTask) -> Self {
        let mut drbg = Drbg::new("mlp.train.v1/order", task.seed);
        let perm = drbg.permutation(task.n_train as u32);
        Self {
            drbg,
            n: task.n_train as u32,
            perm,
            pos: 0,
        }
    }

    /// Next index.
    pub fn next_index(&mut self) -> usize {
        if self.pos >= self.perm.len() {
            self.perm = self.drbg.permutation(self.n);
            self.pos = 0;
        }
        let i = self.perm[self.pos];
        self.pos += 1;
        i as usize
    }

    /// Next batch of indices.
    pub fn next_batch(&mut self, batch: usize) -> Vec<usize> {
        (0..batch).map(|_| self.next_index()).collect()
    }

    /// Skip `count` indices (resume).
    pub fn skip(&mut self, count: usize) {
        for _ in 0..count {
            self.next_index();
        }
    }
}

/// The network.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// First layer weights.
    pub w1: Param<Tensor<B, 2>>,
    /// First layer bias.
    pub b1: Param<Tensor<B, 1>>,
    /// Second layer weights.
    pub w2: Param<Tensor<B, 2>>,
    /// Second layer bias.
    pub b2: Param<Tensor<B, 1>>,
}

fn t2<B: Backend>(device: &B::Device, t: &TensorF32) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(t.data.clone(), [t.shape[0] as usize, t.shape[1] as usize]),
        device,
    )
}

fn t1<B: Backend>(device: &B::Device, t: &TensorF32) -> Tensor<B, 1> {
    Tensor::from_data(
        TensorData::new(t.data.clone(), [t.shape[0] as usize]),
        device,
    )
}

fn host<B: Backend, const D: usize>(t: Tensor<B, D>) -> Result<TensorF32, ComputeError> {
    let data = t.into_data();
    let shape: Vec<u64> = data.shape.iter().map(|d| *d as u64).collect();
    let values = data
        .to_vec::<f32>()
        .map_err(|e| ComputeError::EngineUnavailable(format!("dtype: {e:?}")))?;
    TensorF32::new(shape, values).map_err(|e| ComputeError::EngineUnavailable(e.to_string()))
}

impl<B: Backend> Mlp<B> {
    /// Build from host weights.
    pub fn from_weights(device: &B::Device, w: &Weights) -> Self {
        Self {
            w1: Param::from_tensor(t2::<B>(device, &w.w1)),
            b1: Param::from_tensor(t1::<B>(device, &w.b1)),
            w2: Param::from_tensor(t2::<B>(device, &w.w2)),
            b2: Param::from_tensor(t1::<B>(device, &w.b2)),
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h =
            burn::tensor::activation::relu(x.matmul(self.w1.val()).add(self.b1.val().unsqueeze()));
        h.matmul(self.w2.val()).add(self.b2.val().unsqueeze())
    }

    /// Export weights to the host.
    pub fn to_weights(&self) -> Result<Weights, ComputeError> {
        Ok(Weights {
            w1: host(self.w1.val())?,
            b1: host(self.b1.val())?,
            w2: host(self.w2.val())?,
            b2: host(self.b2.val())?,
        })
    }
}

/// Loss log entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    /// Step (1-based, after the update).
    pub step: u32,
    /// Mean squared error of the batch before the update.
    pub train_loss: f32,
}

/// Metrics written to `metrics.json`. Declared by the worker; recomputed by the validator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlpMetrics {
    /// Kind.
    pub kind: String,
    /// Steps completed.
    pub steps_completed: u32,
    /// Step the run resumed from (0 = fresh start).
    pub resumed_from_step: u32,
    /// Bounded loss log.
    pub log: Vec<LogEntry>,
    /// Last batch loss.
    pub final_train_loss: f32,
    /// MSE on the evaluation split with the final weights.
    pub eval_loss: f32,
    /// Numeric profile that produced the weights.
    pub numeric_profile: String,
}

/// Host-side dataset.
pub struct Data {
    /// `[n_train, d_in]`.
    pub train_x: TensorF32,
    /// `[n_train, d_out]`.
    pub train_y: TensorF32,
    /// `[n_eval, d_in]`.
    pub eval_x: TensorF32,
    /// `[n_eval, d_out]`.
    pub eval_y: TensorF32,
}

fn gather(t: &TensorF32, idx: &[usize]) -> (Vec<f32>, usize) {
    let width = t.shape[1] as usize;
    let mut out = Vec::with_capacity(idx.len() * width);
    for &i in idx {
        out.extend_from_slice(&t.data[i * width..(i + 1) * width]);
    }
    (out, width)
}

fn batch_tensor<B: Backend>(device: &B::Device, t: &TensorF32, idx: &[usize]) -> Tensor<B, 2> {
    let (values, width) = gather(t, idx);
    Tensor::from_data(TensorData::new(values, [idx.len(), width]), device)
}

/// Mean squared error of `weights` on `(x, y)`, computed in chunks.
pub fn mse<B: Backend>(
    device: &B::Device,
    weights: &Weights,
    x: &TensorF32,
    y: &TensorF32,
) -> Result<f32, ComputeError> {
    let model = Mlp::<B>::from_weights(device, weights);
    let n = x.shape[0] as usize;
    let mut total = 0.0f64;
    let chunk = 1024usize;
    let mut start = 0;
    while start < n {
        let end = (start + chunk).min(n);
        let idx: Vec<usize> = (start..end).collect();
        let pred = model.forward(batch_tensor::<B>(device, x, &idx));
        let target = batch_tensor::<B>(device, y, &idx);
        let se: f32 = pred.sub(target).powf_scalar(2.0).sum().into_scalar().elem();
        total += se as f64;
        start = end;
    }
    let denom = (n * y.shape[1] as usize) as f64;
    Ok((total / denom) as f32)
}

#[derive(Serialize, Deserialize)]
struct CheckpointState {
    identity: Sha256Digest,
    step: u32,
    optimizer: Optim,
}

/// Optional stop hook for tests (stop after this many steps of the current run).
#[derive(Clone, Copy, Debug, Default)]
pub struct TrainOptions {
    /// Directory for checkpoints (resume if compatible state exists).
    pub checkpoint_dir: Option<&'static str>,
    /// Stop after this many steps in this invocation (tests only).
    pub stop_after: Option<u32>,
}

/// Train according to the protocol on backend `B`. Returns final weights and metrics.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    task: &MlpTask,
    data: &Data,
    checkpoint_dir: Option<&Path>,
    stop_after: Option<u32>,
    profile: &str,
) -> Result<(Weights, MlpMetrics), ComputeError> {
    let model = Mlp::<B>::from_weights(device, &initial_weights(task));
    match task.optimizer {
        Optim::Sgd => train_loop::<B, _>(
            device,
            task,
            data,
            model,
            SgdConfig::new().init(),
            checkpoint_dir,
            stop_after,
            profile,
        ),
        Optim::Adam => train_loop::<B, _>(
            device,
            task,
            data,
            model,
            AdamConfig::new().init(),
            checkpoint_dir,
            stop_after,
            profile,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn train_loop<B: AutodiffBackend, O: Optimizer<Mlp<B>, B>>(
    device: &B::Device,
    task: &MlpTask,
    data: &Data,
    mut model: Mlp<B>,
    mut optim: O,
    checkpoint_dir: Option<&Path>,
    stop_after: Option<u32>,
    profile: &str,
) -> Result<(Weights, MlpMetrics), ComputeError>
where
    O::Record: Record<B>,
{
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let identity = task.identity();
    let mut stream = IndexStream::new(task);
    let mut start_step = 0u32;
    let mut log: Vec<LogEntry> = Vec::new();

    if let Some(dir) = checkpoint_dir {
        let state_path = dir.join("state.json");
        if let Ok(text) = std::fs::read_to_string(&state_path) {
            let state: CheckpointState = serde_json::from_str(&text)
                .map_err(|e| ComputeError::InvalidParams(format!("checkpoint state: {e}")))?;
            if state.identity == identity
                && state.optimizer == task.optimizer
                && state.step > 0
                && state.step < task.steps
            {
                let files = dir.join(step_dir(state.step));
                let model_bytes = std::fs::read(files.join("model.mpk"))?;
                let optim_bytes = std::fs::read(files.join("optim.mpk"))?;
                let log_text = std::fs::read_to_string(files.join("log.json"))?;
                let m: <Mlp<B> as Module<B>>::Record = recorder
                    .load(model_bytes, device)
                    .map_err(|e| ComputeError::InvalidParams(format!("checkpoint model: {e}")))?;
                let o: O::Record = recorder.load(optim_bytes, device).map_err(|e| {
                    ComputeError::InvalidParams(format!("checkpoint optimizer: {e}"))
                })?;
                model = model.load_record(m);
                optim = optim.load_record(o);
                log = serde_json::from_str(&log_text)
                    .map_err(|e| ComputeError::InvalidParams(format!("checkpoint log: {e}")))?;
                start_step = state.step;
                stream.skip(start_step as usize * task.batch);
                tracing::info!(step = start_step, "resuming from checkpoint");
            }
        }
    }

    let log_every = task.log_every();
    let ckpt_every = task.checkpoint_every();
    let mut last_loss = log.last().map(|l| l.train_loss).unwrap_or(f32::NAN);
    let mut step = start_step;
    let mut done_this_run = 0u32;
    while step < task.steps {
        if stop_after.is_some_and(|s| done_this_run >= s) {
            break;
        }
        let idx = stream.next_batch(task.batch);
        let x = batch_tensor::<B>(device, &data.train_x, &idx);
        let y = batch_tensor::<B>(device, &data.train_y, &idx);
        let pred = model.forward(x);
        let loss = burn::nn::loss::MseLoss::new().forward(pred, y, burn::nn::loss::Reduction::Mean);
        last_loss = loss.clone().into_scalar().elem();
        if !last_loss.is_finite() {
            return Err(ComputeError::OutputLimit(format!(
                "training diverged at step {step} (loss {last_loss})"
            )));
        }
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(task.lr, model, grads);
        step += 1;
        done_this_run += 1;
        if step.is_multiple_of(log_every) || step == task.steps {
            log.push(LogEntry {
                step,
                train_loss: last_loss,
            });
        }
        if let Some(dir) = checkpoint_dir
            && step.is_multiple_of(ckpt_every)
            && step < task.steps
        {
            save_checkpoint(
                &recorder,
                dir,
                &identity,
                task.optimizer,
                step,
                &model,
                &optim,
                &log,
            )?;
        }
    }

    let weights = model.valid().to_weights()?;
    let eval_loss = mse::<B::InnerBackend>(device, &weights, &data.eval_x, &data.eval_y)?;
    let metrics = MlpMetrics {
        kind: KIND.into(),
        steps_completed: step,
        resumed_from_step: start_step,
        log,
        final_train_loss: last_loss,
        eval_loss,
        numeric_profile: profile.to_owned(),
    };
    Ok((weights, metrics))
}

/// Directory holding the files of the checkpoint taken at `step`.
fn step_dir(step: u32) -> String {
    format!("step-{step}")
}

/// Write a checkpoint so that an interruption at any point leaves a consistent one behind.
///
/// Model, optimizer and log go into a directory of their own (`step-<n>`), flushed to disk;
/// `state.json`, replaced atomically last, is the only pointer to it. Resuming reads `state.json`
/// and then that directory, so it never mixes the model of one step with the optimizer of
/// another, whatever moment the process was killed at. Older directories are removed afterwards.
#[allow(clippy::too_many_arguments)]
fn save_checkpoint<B: AutodiffBackend, O: Optimizer<Mlp<B>, B>>(
    recorder: &NamedMpkBytesRecorder<FullPrecisionSettings>,
    dir: &Path,
    identity: &Sha256Digest,
    optimizer: Optim,
    step: u32,
    model: &Mlp<B>,
    optim: &O,
    log: &[LogEntry],
) -> Result<(), ComputeError>
where
    O::Record: Record<B>,
{
    let model_bytes: Vec<u8> = recorder
        .record(model.clone().into_record(), ())
        .map_err(|e| ComputeError::Io(std::io::Error::other(e.to_string())))?;
    let optim_bytes: Vec<u8> = recorder
        .record(optim.to_record(), ())
        .map_err(|e| ComputeError::Io(std::io::Error::other(e.to_string())))?;
    let log_bytes = serde_json::to_vec(log)
        .map_err(|e| ComputeError::Io(std::io::Error::other(e.to_string())))?;
    let state_bytes = serde_json::to_vec(&CheckpointState {
        identity: *identity,
        step,
        optimizer,
    })
    .map_err(|e| ComputeError::Io(std::io::Error::other(e.to_string())))?;

    let files = dir.join(step_dir(step));
    if files.exists() {
        std::fs::remove_dir_all(&files)?;
    }
    std::fs::create_dir_all(&files)?;
    let durable = |path: &Path, bytes: &[u8]| -> Result<(), ComputeError> {
        let mut f = std::fs::File::create(path)?;
        std::io::Write::write_all(&mut f, bytes)?;
        f.sync_all()?;
        Ok(())
    };
    durable(&files.join("model.mpk"), &model_bytes)?;
    durable(&files.join("optim.mpk"), &optim_bytes)?;
    durable(&files.join("log.json"), &log_bytes)?;
    std::fs::File::open(&files)?.sync_all()?;

    let tmp = dir.join(".state.json.tmp");
    durable(&tmp, &state_bytes)?;
    std::fs::rename(&tmp, dir.join("state.json"))?;
    std::fs::File::open(dir)?.sync_all()?;

    let current = step_dir(step);
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("step-") && name != current.as_str() {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
    Ok(())
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

fn load_data(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<Data, ComputeError> {
    Ok(Data {
        train_x: load_input(task, "train_x", inputs)?,
        train_y: load_input(task, "train_y", inputs)?,
        eval_x: load_input(task, "eval_x", inputs)?,
        eval_y: load_input(task, "eval_y", inputs)?,
    })
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

/// Execute a validated `mlp.train.v1` task. Checkpoints are written under `out_dir/checkpoint`
/// and used to resume if the executor restarts on the same bundle.
pub fn execute(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
) -> Result<ExecutionReport, ComputeError> {
    let spec = MlpTask::from_validated(task)?;
    let data = load_data(task, inputs)?;
    let ckpt = out_dir.join("checkpoint");
    let start = Instant::now();
    let (weights, metrics, info): (Weights, MlpMetrics, EngineInfo) = match engine {
        EngineSelection::FlexCpu => {
            let info = cpu_info();
            let device = burn::backend::flex::FlexDevice;
            let (w, m) = train::<burn::backend::Autodiff<burn::backend::Flex>>(
                &device,
                &spec,
                &data,
                Some(&ckpt),
                None,
                info.numeric_profile,
            )?;
            (w, m, info)
        }
        #[cfg(feature = "gpu")]
        EngineSelection::WgpuVulkan { device_index } => {
            let (device, info) = crate::engine::gpu::init_vulkan(device_index)?;
            let (w, m) = train::<burn::backend::Autodiff<burn::backend::Vulkan>>(
                &device,
                &spec,
                &data,
                Some(&ckpt),
                None,
                info.numeric_profile,
            )?;
            (w, m, info)
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
    for (name, t) in weights.named() {
        t.check_shape(&task.output_shapes[name])
            .map_err(|e| ComputeError::OutputLimit(e.to_string()))?;
        let bytes = t.encode();
        let f = write_file(out_dir, &format!("{name}.ggt"), &bytes, budget)?;
        budget = budget.saturating_sub(f.size);
        outputs.insert(name.to_owned(), f);
    }
    let metrics_bytes = serde_json::to_vec_pretty(&metrics)
        .map_err(|e| ComputeError::Io(std::io::Error::other(e)))?;
    let f = write_file(
        out_dir,
        "metrics.json",
        &metrics_bytes,
        MAX_METRICS_BYTES.min(budget),
    )?;
    outputs.insert(METRICS_NAME.to_owned(), f);
    let _ = std::fs::remove_dir_all(&ckpt);
    let mut declared = BTreeMap::new();
    declared.insert("duration_ms".to_owned(), duration_ms as f64);
    declared.insert("eval_loss".to_owned(), metrics.eval_loss as f64);
    declared.insert(
        "final_train_loss".to_owned(),
        metrics.final_train_loss as f64,
    );
    declared.insert("steps_completed".to_owned(), metrics.steps_completed as f64);
    Ok(ExecutionReport {
        outputs,
        engine: info,
        duration_ms,
        metrics: declared,
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

fn within(name: &str, measured: f64, reference: f64, tol: ToleranceV1) -> CheckV1 {
    let diff = (measured - reference).abs();
    let ok = diff <= tol.abs + tol.rel * reference.abs();
    CheckV1 {
        name: name.to_owned(),
        passed: ok,
        measured: Some(diff),
        tolerance: Some(tol.abs + tol.rel * reference.abs()),
        detail: format!("declared {measured}, recomputed {reference}"),
    }
}

/// Verify candidate outputs by replaying the training on the CPU reference profile.
pub fn verify(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
    declared_profile: &str,
) -> Result<Verdict, ComputeError> {
    let spec = MlpTask::from_validated(task)?;
    let mut checks = Vec::new();
    let mut artifacts_checked = Vec::new();
    let reject = |checks: Vec<CheckV1>, artifacts_checked: Vec<ArtifactRef>| {
        Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Rejected,
            artifacts_checked,
        })
    };

    let mut expected: Vec<&str> = WEIGHT_NAMES.to_vec();
    expected.push(METRICS_NAME);
    expected.sort_unstable();
    let claimed_names: Vec<&str> = claimed.keys().map(String::as_str).collect();
    if expected != claimed_names {
        checks.push(fail(
            "outputs.names",
            format!("expected {expected:?}, got {claimed_names:?}"),
        ));
        return reject(checks, artifacts_checked);
    }
    // Byte integrity of every claimed output.
    let mut bytes_by_name: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    for name in expected.iter() {
        let r = claimed[*name];
        let max = if *name == METRICS_NAME {
            MAX_METRICS_BYTES
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
    // Decode.
    let mut decoded: BTreeMap<&str, TensorF32> = BTreeMap::new();
    for name in WEIGHT_NAMES {
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
    let metrics: MlpMetrics = match serde_json::from_slice(&bytes_by_name[METRICS_NAME]) {
        Ok(m) => m,
        Err(e) => {
            checks.push(fail("outputs.metrics.format", e.to_string()));
            return reject(checks, artifacts_checked);
        }
    };
    if metrics.log.len() > MAX_LOG_ENTRIES
        || metrics.steps_completed != spec.steps
        || metrics.kind != KIND
        || !metrics.eval_loss.is_finite()
    {
        checks.push(fail(
            "outputs.metrics.structure",
            format!(
                "steps_completed {}, log entries {}",
                metrics.steps_completed,
                metrics.log.len()
            ),
        ));
        return reject(checks, artifacts_checked);
    }
    // Profile admission.
    if !crate::profiles::admitted_for(KIND, declared_profile)
        || metrics.numeric_profile != declared_profile
    {
        checks.push(fail(
            "profile.admitted",
            format!(
                "declared profile '{declared_profile}' (metrics: '{}') is not admitted for {KIND}",
                metrics.numeric_profile
            ),
        ));
        return Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Inconclusive,
            artifacts_checked,
        });
    }
    checks.push(pass("profile.admitted", declared_profile.to_owned()));

    // Replay on the reference profile.
    let data = load_data(task, inputs)?;
    let device = burn::backend::flex::FlexDevice;
    let (reference, ref_metrics) = train::<burn::backend::Autodiff<burn::backend::Flex>>(
        &device,
        &spec,
        &data,
        None,
        None,
        crate::profiles::REFERENCE_PROFILE,
    )?;
    checks.push(pass(
        "reference.recomputed",
        format!(
            "profile {} steps {}",
            crate::profiles::REFERENCE_PROFILE,
            ref_metrics.steps_completed
        ),
    ));
    let candidate = Weights {
        w1: decoded["w1"].clone(),
        b1: decoded["b1"].clone(),
        w2: decoded["w2"].clone(),
        b2: decoded["b2"].clone(),
    };
    let mut all_ok = true;
    for ((name, c), (_, r)) in candidate.named().iter().zip(reference.named().iter()) {
        let cmp = compare_f32(&format!("outputs.{name}"), c, r, task.tolerance);
        all_ok &= cmp.passed;
        checks.extend(cmp.checks);
    }
    // Declared metrics are recomputed from the *candidate* weights, never trusted.
    let recomputed_eval =
        mse::<burn::backend::Flex>(&device, &candidate, &data.eval_x, &data.eval_y)?;
    let c = within(
        "metrics.eval_loss",
        metrics.eval_loss as f64,
        recomputed_eval as f64,
        task.tolerance,
    );
    all_ok &= c.passed;
    checks.push(c);
    // Loss trajectory vs. the replay (same schedule).
    if metrics.log.len() != ref_metrics.log.len()
        || metrics
            .log
            .iter()
            .zip(ref_metrics.log.iter())
            .any(|(a, b)| a.step != b.step)
    {
        checks.push(fail(
            "metrics.log.schedule",
            format!(
                "declared {} entries, expected {}",
                metrics.log.len(),
                ref_metrics.log.len()
            ),
        ));
        all_ok = false;
    } else {
        let mut max_diff = 0.0f64;
        let mut worst_tol = 0.0f64;
        let mut ok = true;
        for (a, b) in metrics.log.iter().zip(ref_metrics.log.iter()) {
            let diff = (a.train_loss as f64 - b.train_loss as f64).abs();
            let tol = task.tolerance.abs + task.tolerance.rel * (b.train_loss as f64).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_tol = tol;
            }
            ok &= diff <= tol;
        }
        checks.push(CheckV1 {
            name: "metrics.log.trajectory".into(),
            passed: ok,
            measured: Some(max_diff),
            tolerance: Some(worst_tol),
            detail: format!("{} entries", metrics.log.len()),
        });
        all_ok &= ok;
    }
    let outcome = if all_ok {
        VerificationOutcome::Accepted
    } else {
        VerificationOutcome::Rejected
    };
    Ok(Verdict {
        checks,
        outcome,
        artifacts_checked,
    })
}
