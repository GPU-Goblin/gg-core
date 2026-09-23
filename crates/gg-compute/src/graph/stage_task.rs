//! `graph.stage.v1`: one stage of a model split across machines, forward or backward pass
//! (gg-server/docs/ROADMAP.md P11 D2, ADR-0024). Input and output names are described in
//! `gg_protocol::messages::GraphStageParamsV1`.
//!
//! A stage task holds only its stage's parameters (the checkpoint shards that carry them) and its
//! inputs: this is what lets a model fit when no single machine can hold it. The validator
//! recomputes it from the same inputs, so a forged activation is caught at the stage that produced
//! it, before a later stage consumes it in an aggregated round.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::messages::{ArtifactRef, CheckV1, GraphStageParamsV1, TaskParamsV1};
use gg_protocol::tensor_file::{BundleF32, TensorF32, TensorLimits};
use serde::{Deserialize, Serialize};

use super::exec::{self, Prepared};
use super::shards::{self, ShardPlan};
use super::stages::{self, StageSpec};
use super::{ModelGraphV1, task};
use crate::catalog::ValidatedTask;
use crate::compare::compare_f32_norm;
use crate::engine::{EngineInfo, EngineSelection, cpu_info};
use crate::error::ComputeError;
use crate::matmul::{ExecutionReport, OutputFile, Verdict};

/// Compute kind id.
pub const KIND: &str = "graph.stage.v1";
/// Metadata output.
pub const META_NAME: &str = "stage_meta";

/// Metadata of a stage task (declared; recomputed by the validator).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageMeta {
    /// Kind.
    pub kind: String,
    /// Stage.
    pub stage: u32,
    /// Backward pass.
    pub backward: bool,
    /// Examples in the micro-batch.
    pub n_examples: u32,
    /// Loss of the micro-batch, for the last stage's backward pass.
    pub loss: Option<f32>,
    /// Numeric profile.
    pub numeric_profile: String,
    /// Peak tensor bytes held (measured).
    pub peak_bytes: u64,
}

/// Name of the k-th non-dataset input of a stage.
pub fn input_name(k: usize) -> String {
    format!("in.{k}")
}
/// Name of the k-th output of a forward pass.
pub fn output_name(k: usize) -> String {
    format!("out.{k}")
}
/// Name of the gradient of output k received from its c-th consumer.
pub fn dout_name(k: usize, c: usize) -> String {
    format!("dout.{k}.{c}")
}
/// Name of the gradient of the k-th non-dataset input, produced by a backward pass.
pub fn din_name(k: usize) -> String {
    format!("din.{k}")
}

/// Everything a stage task derives from its graph and bounds.
pub struct Resolved {
    /// The graph.
    pub graph: ModelGraphV1,
    /// Validated.
    pub prepared: Prepared,
    /// All stages.
    pub stages: Vec<StageSpec>,
    /// Checkpoint shards of the staged run.
    pub plan: ShardPlan,
}

impl Resolved {
    /// This task's stage.
    pub fn stage(&self, p: &GraphStageParamsV1) -> &StageSpec {
        &self.stages[p.stage as usize]
    }
    /// Is this input of the stage a dataset?
    pub fn is_dataset(&self, name: &str) -> bool {
        self.graph.inputs.iter().any(|s| s.name == name) || self.graph.loss.target().name == name
    }
    /// Consumers of an output of stage `s`, later stages in decreasing order.
    pub fn consumers(&self, s: usize, output: &str) -> Vec<usize> {
        (s + 1..self.stages.len())
            .rev()
            .filter(|&c| self.stages[c].inputs.iter().any(|i| i == output))
            .collect()
    }
}

/// Resolve the stage structure of a staged run from its graph and bounds.
pub fn resolve(graph: ModelGraphV1, bounds: &[u32]) -> Result<Resolved, ComputeError> {
    let prepared = Prepared::new(&graph).map_err(|e| invalid(format!("graph refused: {e}")))?;
    let stages = stages::stages(&graph, prepared.report(), bounds).map_err(invalid)?;
    let plan = shards::plan_stages(&graph, &stages).map_err(invalid)?;
    Ok(Resolved {
        graph,
        prepared,
        stages,
        plan,
    })
}

fn invalid(msg: impl Into<String>) -> ComputeError {
    ComputeError::InvalidParams(msg.into())
}

fn params_of(task: &ValidatedTask) -> Result<&GraphStageParamsV1, ComputeError> {
    match &task.params {
        TaskParamsV1::GraphStage(p) => Ok(p),
        other => Err(ComputeError::UnknownKind(other.compute_kind().to_owned())),
    }
}

/// Training FLOPs and activation bytes per example of one stage, from the validation report.
pub fn stage_costs(r: &Resolved, s: usize) -> (u64, u64) {
    let st = &r.stages[s];
    let names = &r.prepared.report().order[st.first..st.end];
    r.prepared
        .report()
        .nodes
        .iter()
        .filter(|c| names.contains(&c.output))
        .fold((0, 0), |(f, a), c| {
            (
                f + c.training_flops_per_example,
                a + c.activation_bytes_per_example,
            )
        })
}

/// Read, hash-check and resolve the graph input; check what the task declares against it.
pub fn load(
    p: &GraphStageParamsV1,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<Resolved, ComputeError> {
    let path = inputs
        .get(task::GRAPH_INPUT)
        .ok_or_else(|| ComputeError::Inputs("input file 'graph' not provided".into()))?;
    let meta = std::fs::metadata(path)?;
    if meta.len() != p.graph_bytes || meta.len() > task::MAX_GRAPH_BYTES {
        return Err(ComputeError::Inputs("graph file size mismatch".into()));
    }
    let bytes = std::fs::read(path)?;
    if Sha256Digest::of_bytes(&bytes).to_hex() != p.graph_hash {
        return Err(ComputeError::Inputs("graph hash mismatch".into()));
    }
    let graph: ModelGraphV1 = serde_json::from_slice(&bytes)
        .map_err(|e| ComputeError::Inputs(format!("graph is not gg-graph/v1: {e}")))?;
    let r = resolve(graph, &p.stage_bounds)?;
    if r.prepared.report().parameters != p.parameters {
        return Err(invalid("parameters do not match the graph"));
    }
    if p.stage as usize >= r.stages.len() {
        return Err(invalid(format!(
            "stage {} of {} stages",
            p.stage,
            r.stages.len()
        )));
    }
    let (flops, act) = stage_costs(&r, p.stage as usize);
    if flops != p.training_flops_per_example || act != p.activation_bytes_per_example {
        return Err(invalid("stage costs do not match the graph"));
    }
    Ok(r)
}

/// Expected inputs of the task, by name, with what they are.
pub enum Expected {
    /// Checkpoint shard `j`.
    Shard(usize),
    /// Dataset rows.
    Dataset(String),
    /// Activation input, tensor name.
    Activation(String),
    /// Gradient of an output from one consumer: tensor name, consumer index.
    OutputGradient(String, usize),
}

/// The inputs a stage task takes (besides `graph`).
pub fn expected_inputs(r: &Resolved, p: &GraphStageParamsV1) -> Vec<(String, Expected)> {
    let st = r.stage(p);
    let mut out = Vec::new();
    for j in shards::stage_shards(&r.plan, st) {
        out.push((r.plan.checkpoint_name(j), Expected::Shard(j)));
    }
    for (k, name) in st.inputs.iter().enumerate() {
        if r.is_dataset(name) {
            out.push((name.clone(), Expected::Dataset(name.clone())));
        } else {
            out.push((input_name(k), Expected::Activation(name.clone())));
        }
    }
    if p.backward && !st.last {
        for (k, name) in st.outputs.iter().enumerate() {
            for c in 0..r.consumers(st.index, name).len() {
                out.push((dout_name(k, c), Expected::OutputGradient(name.clone(), c)));
            }
        }
    }
    out
}

/// The outputs a stage task produces, with their maximum size where known.
pub fn expected_outputs(r: &Resolved, p: &GraphStageParamsV1) -> Vec<String> {
    let st = r.stage(p);
    let mut out = vec![META_NAME.to_owned()];
    if p.backward {
        for j in shards::stage_shards(&r.plan, st) {
            if trained_of(r, st, j).next().is_some() {
                out.push(r.plan.gradient_name(j));
            }
        }
        for (k, name) in st.inputs.iter().enumerate() {
            if !r.is_dataset(name) {
                out.push(din_name(k));
            }
        }
    } else {
        for k in 0..st.outputs.len() {
            out.push(output_name(k));
        }
    }
    out.sort();
    out
}

/// Bytes a stage task writes in all, the executor's output budget (`max_output_bytes`; the control
/// plane also checks each upload against it): a forward pass writes the stage's boundary
/// activations, a backward pass the gradient of each non-dataset input and one gradient bundle
/// per shard it trains, and both a metadata file. Each tensor is counted with 4 KiB for its header,
/// each bundle with 4 KiB more.
///
/// The catalog's estimate counted twice every activation of the stage: at 32 sequences of 64
/// tokens, a SmolLM2 stage exceeded the protocol's 16 GiB bound and the run was refused.
pub fn output_bytes(r: &Resolved, p: &GraphStageParamsV1) -> u64 {
    const HEADER: u64 = 4096;
    let st = r.stage(p);
    let n = p.example_count as u64;
    let act = |name: &str| {
        r.prepared
            .report()
            .nodes
            .iter()
            .find(|c| c.output == name)
            .map_or(0, |c| c.activation_bytes_per_example.saturating_mul(n))
            .saturating_add(HEADER)
    };
    let mut total = task::MAX_META_BYTES;
    if p.backward {
        for name in st.inputs.iter().filter(|name| !r.is_dataset(name)) {
            total = total.saturating_add(act(name));
        }
        for j in shards::stage_shards(&r.plan, st) {
            let bundle: u64 = trained_of(r, st, j)
                .filter_map(|name| r.graph.parameters.iter().find(|q| q.name == name))
                .map(|q| {
                    (q.shape.iter().product::<u64>() * p.gradient_dtype().width())
                        .saturating_add(HEADER)
                })
                .sum();
            total = total.saturating_add(bundle.saturating_add(HEADER));
        }
    } else {
        for name in &st.outputs {
            total = total.saturating_add(act(name));
        }
    }
    total
}

/// Trained parameters of shard `j` used by the stage.
fn trained_of<'a>(r: &'a Resolved, st: &'a StageSpec, j: usize) -> impl Iterator<Item = &'a str> {
    r.plan.shards[j]
        .params
        .iter()
        .filter(move |n| st.params.contains(n))
        .filter(move |n| {
            r.graph
                .parameters
                .iter()
                .any(|q| &&q.name == n && q.trainable)
        })
        .map(String::as_str)
}

fn read_tensor(path: &Path, max_elements: u64, name: &str) -> Result<TensorF32, ComputeError> {
    let t = TensorF32::read_from_path(path, &TensorLimits { max_elements }).map_err(|source| {
        ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        }
    })?;
    t.check_finite()
        .map_err(|source| ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        })?;
    Ok(t)
}

/// Everything the stage computation needs, loaded and checked.
pub struct Loaded {
    /// Parameters of the stage.
    pub params: BTreeMap<String, TensorF32>,
    /// Inputs of the stage, by tensor name.
    pub inputs: BTreeMap<String, TensorF32>,
    /// Received output gradients, by tensor name, in consumer order.
    pub output_gradients: BTreeMap<String, Vec<TensorF32>>,
}

/// Load the task's inputs.
pub fn load_inputs(
    r: &Resolved,
    p: &GraphStageParamsV1,
    files: &BTreeMap<String, PathBuf>,
) -> Result<Loaded, ComputeError> {
    let st = r.stage(p);
    let max = p.parameters.max(1 << 24);
    let mut params = BTreeMap::new();
    let mut inputs = BTreeMap::new();
    let mut output_gradients: BTreeMap<String, Vec<TensorF32>> = BTreeMap::new();
    let get = |n: &str| {
        files
            .get(n)
            .ok_or_else(|| ComputeError::Inputs(format!("input file '{n}' not provided")))
    };
    for (name, what) in expected_inputs(r, p) {
        let path = get(&name)?;
        match what {
            Expected::Shard(j) => {
                let shard = &r.plan.shards[j];
                if std::fs::metadata(path)?.len() != shard.checkpoint_bytes {
                    return Err(ComputeError::Inputs(format!("{name} size mismatch")));
                }
                let b = BundleF32::read_from_path(path, &TensorLimits { max_elements: max })
                    .map_err(|source| ComputeError::InputTensor {
                        name: name.clone(),
                        source,
                    })?;
                task::check_shard(&r.graph, shard, &b, &name, false)?;
                for (k, v) in b.entries {
                    if st.params.contains(&k) {
                        params.insert(k, v);
                    }
                }
            }
            Expected::Dataset(t) | Expected::Activation(t) => {
                let v = read_tensor(path, max, &name)?;
                if v.shape.first() != Some(&(p.example_count as u64)) {
                    return Err(ComputeError::Inputs(format!(
                        "{name} carries {:?}, a micro-batch of {} expected",
                        v.shape.first(),
                        p.example_count
                    )));
                }
                inputs.insert(t, v);
            }
            Expected::OutputGradient(t, _) => {
                output_gradients
                    .entry(t)
                    .or_default()
                    .push(read_tensor(path, max, &name)?);
            }
        }
    }
    Ok(Loaded {
        params,
        inputs,
        output_gradients,
    })
}

/// Result of computing a stage.
pub struct Computed {
    /// Named output tensors (not bundles): `out.k`, `din.k`.
    pub tensors: BTreeMap<String, TensorF32>,
    /// Gradient bundles by name: `gradient.j`.
    pub gradients: BTreeMap<String, BTreeMap<String, TensorF32>>,
    /// Loss.
    pub loss: Option<f32>,
    /// Measured peak.
    pub peak_bytes: u64,
}

/// A stage's backward results as its task outputs: `din.k` per non-dataset input, one gradient
/// bundle per shard it trains.
fn backward_computed(r: &Resolved, st: &StageSpec, b: exec::StageBackward) -> Computed {
    let mut tensors = BTreeMap::new();
    for (k, name) in st.inputs.iter().enumerate() {
        if let Some(g) = b.input_gradients.get(name) {
            tensors.insert(din_name(k), g.clone());
        }
    }
    let mut gradients = BTreeMap::new();
    for j in shards::stage_shards(&r.plan, st) {
        let part: BTreeMap<String, TensorF32> = trained_of(r, st, j)
            .map(|n| (n.to_owned(), b.gradients[n].clone()))
            .collect();
        if !part.is_empty() {
            gradients.insert(r.plan.gradient_name(j), part);
        }
    }
    Computed {
        tensors,
        gradients,
        loss: b.loss,
        peak_bytes: b.peak_bytes,
    }
}

/// Compute the stage on the given engine.
pub fn compute(
    engine: EngineSelection,
    r: &Resolved,
    p: &GraphStageParamsV1,
    l: &Loaded,
) -> Result<(Computed, EngineInfo), ComputeError> {
    let st = r.stage(p);
    let seed = p.round_seed ^ p.example_start.rotate_left(32);
    let to_err = |e: exec::ExecError| invalid(format!("stage execution: {e}"));
    macro_rules! run {
        ($inner:ty, $device:expr, $info:expr) => {{
            let device = $device;
            if p.backward {
                let b = exec::stage_backward::<burn::backend::Autodiff<$inner>>(
                    &device,
                    &r.prepared,
                    st,
                    &l.params,
                    &l.inputs,
                    &l.output_gradients,
                    seed,
                )
                .map_err(to_err)?;
                (backward_computed(r, st, b), $info)
            } else {
                let (outputs, peak) = exec::stage_forward::<$inner>(
                    &device,
                    &r.prepared,
                    st,
                    &l.params,
                    &l.inputs,
                    seed,
                )
                .map_err(to_err)?;
                let tensors = st
                    .outputs
                    .iter()
                    .enumerate()
                    .map(|(k, n)| (output_name(k), outputs[n].clone()))
                    .collect();
                (
                    Computed {
                        tensors,
                        gradients: BTreeMap::new(),
                        loss: None,
                        peak_bytes: peak,
                    },
                    $info,
                )
            }
        }};
    }
    match engine {
        EngineSelection::FlexCpu => Ok(run!(
            burn::backend::Flex,
            burn::backend::flex::FlexDevice,
            cpu_info()
        )),
        #[cfg(feature = "gpu")]
        EngineSelection::WgpuVulkan { device_index } => {
            let (device, info) = crate::engine::gpu::init_vulkan(device_index)?;
            let r = run!(burn::backend::Vulkan, device.clone(), info);
            crate::engine::gpu::release_memory(&device);
            Ok(r)
        }
        #[cfg(not(feature = "gpu"))]
        EngineSelection::WgpuVulkan { .. } => Err(ComputeError::EngineUnavailable(
            "this build has no GPU support (feature `gpu`)".into(),
        )),
        // The forward pass runs the executor's operators on LibTorch; the backward pass uses
        // PyTorch's autograd, as whole-model tasks do (ADR-0033).
        #[cfg(feature = "cuda")]
        EngineSelection::TorchCuda { device_index } => {
            let info = crate::engine::cuda::check(device_index)?;
            if p.backward {
                let b = exec::torch::stage_backward_cuda(
                    device_index,
                    &r.prepared,
                    st,
                    &l.params,
                    &l.inputs,
                    &l.output_gradients,
                    seed,
                )
                .map_err(to_err)?;
                return Ok((backward_computed(r, st, b), info));
            }
            Ok(run!(
                burn::backend::LibTorch<f32>,
                burn::backend::libtorch::LibTorchDevice::Cuda(device_index),
                info
            ))
        }
        #[cfg(not(feature = "cuda"))]
        EngineSelection::TorchCuda { .. } => Err(ComputeError::EngineUnavailable(
            "this build has no native CUDA engine (feature `cuda`)".into(),
        )),
    }
}

fn write_file(
    out_dir: &Path,
    file_name: &str,
    bytes: &[u8],
    budget: &mut u64,
) -> Result<OutputFile, ComputeError> {
    if bytes.len() as u64 > *budget {
        return Err(ComputeError::OutputLimit(format!(
            "output {file_name} of {} bytes exceeds the remaining budget {budget}",
            bytes.len()
        )));
    }
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join(file_name);
    std::fs::write(&path, bytes)?;
    *budget -= bytes.len() as u64;
    Ok(OutputFile {
        path,
        hash: Sha256Digest::of_bytes(bytes),
        size: bytes.len() as u64,
    })
}

/// Execute a `graph.stage.v1` task within `device_memory` bytes when given (C1).
pub fn execute(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
    device_memory: Option<u64>,
) -> Result<ExecutionReport, ComputeError> {
    let p = params_of(task)?;
    let r = load(p, inputs)?;
    if let Some(limit) = device_memory {
        let need = stages::stage_memory(
            &r.graph,
            r.prepared.report(),
            r.stage(p),
            p.example_count as u64,
        );
        if need > limit {
            return Err(ComputeError::MemoryLimit(format!(
                "stage {} needs about {need} bytes, this executor may use {limit}",
                p.stage
            )));
        }
    }
    let loaded = load_inputs(&r, p, inputs)?;
    let start = Instant::now();
    let (c, info) = compute(engine, &r, p, &loaded)?;
    let duration_ms = start.elapsed().as_millis() as u64;
    let mut budget = task.limits.max_output_bytes;
    let mut outputs = BTreeMap::new();
    for (name, t) in &c.tensors {
        let f = write_file(out_dir, &format!("{name}.ggt"), &t.encode(), &mut budget)?;
        outputs.insert(name.clone(), f);
    }
    for (name, part) in c.gradients {
        let b = BundleF32::new(part)
            .map_err(|e| ComputeError::OutputLimit(format!("gradient bundle: {e}")))?;
        let bytes = b
            .encode_as(p.gradient_dtype())
            .map_err(|e| ComputeError::OutputLimit(format!("{name}: {e}")))?;
        let f = write_file(out_dir, &format!("{name}.ggb"), &bytes, &mut budget)?;
        outputs.insert(name, f);
    }
    let meta = StageMeta {
        kind: KIND.into(),
        stage: p.stage,
        backward: p.backward,
        n_examples: p.example_count,
        loss: c.loss,
        numeric_profile: info.numeric_profile.to_owned(),
        peak_bytes: c.peak_bytes,
    };
    let bytes =
        serde_json::to_vec_pretty(&meta).map_err(|e| ComputeError::Io(std::io::Error::other(e)))?;
    outputs.insert(
        META_NAME.to_owned(),
        write_file(out_dir, "stage_meta.json", &bytes, &mut budget)?,
    );
    let mut metrics = BTreeMap::new();
    metrics.insert("duration_ms".to_owned(), duration_ms as f64);
    if let Some(l) = c.loss {
        metrics.insert("loss".to_owned(), l as f64);
    }
    Ok(ExecutionReport {
        outputs,
        engine: info,
        duration_ms,
        metrics,
    })
}

fn check(name: &str, passed: bool, detail: String) -> CheckV1 {
    CheckV1 {
        name: name.to_owned(),
        passed,
        measured: None,
        tolerance: None,
        detail: detail.chars().take(200).collect(),
    }
}

/// Verify a stage task by recomputing it from the same inputs on `reference_engine` (the CPU
/// reference profile, or the GPU for a GPU validator, ADR-0031).
pub fn verify(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
    declared_profile: &str,
    reference_engine: EngineSelection,
) -> Result<Verdict, ComputeError> {
    let p = params_of(task)?;
    let mut checks = Vec::new();
    let mut artifacts_checked = Vec::new();
    let verdict = |checks: Vec<CheckV1>, a: Vec<ArtifactRef>, outcome| {
        Ok(Verdict {
            checks,
            outcome,
            artifacts_checked: a,
        })
    };
    let r = load(p, inputs)?;
    let expected = expected_outputs(&r, p);
    let claimed_names: Vec<&String> = claimed.keys().collect();
    if expected.iter().collect::<Vec<_>>() != claimed_names {
        checks.push(check(
            "outputs.names",
            false,
            format!("expected {expected:?}, got {claimed_names:?}"),
        ));
        return verdict(checks, artifacts_checked, VerificationOutcome::Rejected);
    }
    let mut bytes_of: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for name in &expected {
        let a = claimed[name];
        let Some(path) = candidates.get(name) else {
            checks.push(check(
                &format!("outputs.{name}.present"),
                false,
                "missing".into(),
            ));
            return verdict(checks, artifacts_checked, VerificationOutcome::Inconclusive);
        };
        let bytes = std::fs::read(path)?;
        if bytes.len() as u64 != a.size || Sha256Digest::of_bytes(&bytes) != a.hash {
            checks.push(check(
                &format!("outputs.{name}.sha256"),
                false,
                "hash or size mismatch".into(),
            ));
            return verdict(checks, artifacts_checked, VerificationOutcome::Rejected);
        }
        artifacts_checked.push(a);
        bytes_of.insert(name.clone(), bytes);
    }
    let meta: StageMeta = match serde_json::from_slice(&bytes_of[META_NAME]) {
        Ok(m) => m,
        Err(e) => {
            checks.push(check("outputs.stage_meta.format", false, e.to_string()));
            return verdict(checks, artifacts_checked, VerificationOutcome::Rejected);
        }
    };
    if meta.kind != KIND
        || meta.stage != p.stage
        || meta.backward != p.backward
        || meta.n_examples != p.example_count
    {
        checks.push(check(
            "outputs.stage_meta.structure",
            false,
            format!("{meta:?}"),
        ));
        return verdict(checks, artifacts_checked, VerificationOutcome::Rejected);
    }
    if !crate::profiles::admitted_for(KIND, declared_profile)
        || meta.numeric_profile != declared_profile
    {
        checks.push(check(
            "profile.admitted",
            false,
            declared_profile.to_owned(),
        ));
        return verdict(checks, artifacts_checked, VerificationOutcome::Inconclusive);
    }
    let loaded = load_inputs(&r, p, inputs)?;
    let (reference, _) = compute(reference_engine, &r, p, &loaded)?;
    checks.push(check(
        "reference.recomputed",
        true,
        format!(
            "stage {} ({}) over {} examples, profile {}",
            p.stage,
            if p.backward { "backward" } else { "forward" },
            p.example_count,
            reference_engine.numeric_profile()
        ),
    ));
    let limits = TensorLimits {
        max_elements: p.parameters.max(1 << 24),
    };
    let mut ok = true;
    for (name, want) in &reference.tensors {
        let got = match TensorF32::decode(&bytes_of[name], &limits) {
            Ok(t) => t,
            Err(e) => {
                checks.push(check(
                    &format!("outputs.{name}.format"),
                    false,
                    e.to_string(),
                ));
                return verdict(checks, artifacts_checked, VerificationOutcome::Rejected);
            }
        };
        let cmp = compare_f32_norm(&format!("outputs.{name}"), &got, want, task.tolerance);
        ok &= cmp.passed;
        if !cmp.passed || checks.len() < 64 {
            checks.extend(cmp.checks);
        }
    }
    for (name, want) in &reference.gradients {
        let got = match BundleF32::decode_as(&bytes_of[name], &limits, p.gradient_dtype()) {
            Ok(b) => b.entries,
            Err(e) => {
                checks.push(check(
                    &format!("outputs.{name}.format"),
                    false,
                    e.to_string(),
                ));
                return verdict(checks, artifacts_checked, VerificationOutcome::Rejected);
            }
        };
        if got.keys().collect::<Vec<_>>() != want.keys().collect::<Vec<_>>() {
            checks.push(check(
                &format!("outputs.{name}.layout"),
                false,
                format!("{:?}", got.keys().collect::<Vec<_>>()),
            ));
            return verdict(checks, artifacts_checked, VerificationOutcome::Rejected);
        }
        for (k, w) in want {
            let cmp = compare_f32_norm(&format!("outputs.{name}.{k}"), &got[k], w, task.tolerance);
            ok &= cmp.passed;
            if !cmp.passed || checks.len() < 64 {
                checks.extend(cmp.checks);
            }
        }
    }
    match (meta.loss, reference.loss) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            let diff = (a as f64 - b as f64).abs();
            let tol = task.tolerance.abs + task.tolerance.rel * (b as f64).abs();
            ok &= diff <= tol;
            checks.push(CheckV1 {
                name: "stage_meta.loss".into(),
                passed: diff <= tol,
                measured: Some(diff),
                tolerance: Some(tol),
                detail: format!("declared {a}, recomputed {b}"),
            });
        }
        _ => {
            ok = false;
            checks.push(check(
                "stage_meta.loss",
                false,
                "loss present where not expected, or missing".into(),
            ));
        }
    }
    checks.truncate(gg_protocol::messages::MAX_CHECKS);
    verdict(
        checks,
        artifacts_checked,
        if ok {
            VerificationOutcome::Accepted
        } else {
            VerificationOutcome::Rejected
        },
    )
}
