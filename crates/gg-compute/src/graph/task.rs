//! `graph.grad.v1`: mean gradient of an explicit micro-batch for a model described by a graph,
//! from a given checkpoint (ADR-0016, ADR-0017). This is `mlp.grad.v1` generalized: the same
//! guarantees (disjoint slice drawn from the DRBG, recomputation by the validator, published
//! tolerance), but the model is an input and the parameters are named.
//!
//! Inputs: `graph` (a `gg-graph/v1` description), `checkpoint` (`GGB1` bundle of every parameter)
//! or its shards `checkpoint.0 …` (C2, [`super::shards`]), then one tensor per graph input and one
//! for the loss target.
//! Outputs: `gradient` (`GGB1` bundle of the trained parameters, f32 or bf16 as the task says,
//! ADR-0035) or its shards `gradient.i`, and `grad_meta`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::{ArtifactRef, CheckV1, GraphGradParamsV1, TaskParamsV1};
use gg_protocol::tensor_file::{BundleF32, TensorF32, TensorLimits};
use serde::{Deserialize, Serialize};

use super::exec::{self, ExecOptions, Prepared};
use super::{ModelGraphV1, TensorSpec};
use crate::catalog::{ValidatedTask, tensor_limits_for};
use crate::compare::compare_f32_norm;
use crate::engine::{EngineInfo, EngineSelection, cpu_info};
use crate::error::ComputeError;
use crate::grad::CoordOptim;
use crate::matmul::{ExecutionReport, OutputFile, Verdict};

/// Compute kind id.
pub const KIND: &str = "graph.grad.v1";
/// Name of the graph input.
pub const GRAPH_INPUT: &str = "graph";
/// Name of the checkpoint input.
pub const CHECKPOINT_INPUT: &str = "checkpoint";
/// Name of the gradient output.
pub const GRADIENT_OUTPUT: &str = "gradient";
/// Name of the metadata output.
pub const META_NAME: &str = "grad_meta";
/// Maximum size of `grad_meta.json`.
pub const MAX_META_BYTES: u64 = 4096;
/// Maximum size of a graph description.
pub const MAX_GRAPH_BYTES: u64 = 4 << 20;
/// Maximum size of a checkpoint bundle in this version (about 268 million parameters).
pub const MAX_CHECKPOINT_BYTES: u64 = 1 << 30;
/// Maximum examples in one micro-batch.
pub const MAX_EXAMPLE_COUNT: u32 = 4096;
/// Maximum segments of activation recomputation.
pub const MAX_SEGMENTS: u32 = 4096;
/// Maximum dataset tensors (graph inputs plus the target).
pub const MAX_DATASETS: usize = 16;
/// Maximum training examples.
pub const MAX_TOTAL_EXAMPLES: u64 = 1 << 32;
/// Maximum bytes of one dataset tensor.
pub const MAX_DATASET_BYTES: u64 = 8 << 30;

/// Metadata of a gradient task (declared; recomputed by the validator).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphGradMeta {
    /// Kind.
    pub kind: String,
    /// Examples in the micro-batch.
    pub n_examples: u32,
    /// Mean loss of the micro-batch at the checkpoint.
    pub loss: f32,
    /// Numeric profile.
    pub numeric_profile: String,
    /// Parameters of the model, frozen ones included.
    pub parameters: u64,
    /// Peak activation bytes the executor actually held (measured, not estimated).
    #[serde(default)]
    pub peak_activation_bytes: u64,
    /// Peak parameter bytes held on the device.
    #[serde(default)]
    pub peak_parameter_bytes: u64,
}

/// `GGB1` size of the parameters a graph actually trains: that is a gradient's layout.
pub fn gradient_bundle_size(graph: &ModelGraphV1) -> u64 {
    BundleF32::encoded_size(
        graph
            .parameters
            .iter()
            .filter(|q| q.trainable)
            .map(|q| (q.name.as_str(), q.shape.as_slice())),
    )
}

/// `GGB1` size of a full checkpoint (every parameter, frozen ones included).
pub fn checkpoint_bundle_size(graph: &ModelGraphV1) -> u64 {
    BundleF32::encoded_size(
        graph
            .parameters
            .iter()
            .map(|q| (q.name.as_str(), q.shape.as_slice())),
    )
}

/// Index stream of a round (concatenated permutations of `0..total`), own domain.
pub fn round_indices(round_seed: u64, total: u64, start: u64, count: u32) -> Vec<usize> {
    let mut d = Drbg::new("graph.grad.v1/order", round_seed);
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

/// Seed of the Dropout masks of one micro-batch: each slice of a round has its own.
pub fn dropout_seed(p: &GraphGradParamsV1) -> u64 {
    p.round_seed ^ p.example_start.rotate_left(32)
}

/// Rows `idx` of a tensor whose first dimension indexes examples.
pub fn select_rows(t: &TensorF32, idx: &[usize]) -> TensorF32 {
    let row: usize = t.shape[1..].iter().product::<u64>() as usize;
    let mut data = Vec::with_capacity(idx.len() * row);
    for &i in idx {
        data.extend_from_slice(&t.data[i * row..(i + 1) * row]);
    }
    let mut shape = t.shape.clone();
    shape[0] = idx.len() as u64;
    TensorF32::new(shape, data).expect("consistent shape")
}

fn invalid(msg: impl Into<String>) -> ComputeError {
    ComputeError::InvalidParams(msg.into())
}

/// Graph inputs followed by the target tensor, in the order expected for the datasets.
fn graph_datasets(graph: &ModelGraphV1) -> Vec<&TensorSpec> {
    let mut v: Vec<&TensorSpec> = graph.inputs.iter().collect();
    v.push(graph.loss.target());
    v
}

/// Checks that the description matches the declared task parameters.
pub fn check_declared(
    graph: &ModelGraphV1,
    prepared: &Prepared,
    p: &GraphGradParamsV1,
) -> Result<(), ComputeError> {
    let report = prepared.report();
    if report.parameters != p.parameters {
        return Err(invalid(format!(
            "graph has {} parameters, task declares {}",
            report.parameters, p.parameters
        )));
    }
    if report.training_flops_per_example != p.training_flops_per_example {
        return Err(invalid(format!(
            "graph costs {} FLOPs per example, task declares {}",
            report.training_flops_per_example, p.training_flops_per_example
        )));
    }
    if report.activation_bytes_per_example != p.activation_bytes_per_example {
        return Err(invalid(
            "activation_bytes_per_example does not match the graph",
        ));
    }
    let plan = shard_plan(graph, prepared, p)?;
    let expected_grad = plan.gradient_size(p.gradient_dtype());
    if expected_grad != p.gradient_bytes {
        return Err(invalid(format!(
            "gradient of this graph is {expected_grad} bytes, task declares {}",
            p.gradient_bytes
        )));
    }
    let expected = plan.checkpoint_bytes();
    if expected != p.checkpoint_bytes {
        return Err(invalid(format!(
            "checkpoint of this graph is {expected} bytes, task declares {}",
            p.checkpoint_bytes
        )));
    }
    let specs = graph_datasets(graph);
    if specs.len() != p.datasets.len() {
        return Err(invalid(format!(
            "graph needs {} dataset tensors, task declares {}",
            specs.len(),
            p.datasets.len()
        )));
    }
    for spec in specs {
        let d = p
            .datasets
            .iter()
            .find(|d| d.name == spec.name)
            .ok_or_else(|| invalid(format!("dataset '{}' missing from the task", spec.name)))?;
        if d.dtype != spec.dtype {
            return Err(invalid(format!(
                "dataset '{}' is {} but the graph expects {}",
                d.name, d.dtype, spec.dtype
            )));
        }
        if d.shape.len() != spec.shape.len() || d.shape.first() != Some(&p.total_examples) {
            return Err(invalid(format!(
                "dataset '{}' shape {:?} does not match graph input {:?} over {} examples",
                d.name, d.shape, spec.shape, p.total_examples
            )));
        }
        for (i, (got, want)) in d.shape.iter().zip(spec.shape.iter()).enumerate().skip(1) {
            if *want != 0 && got != want {
                return Err(invalid(format!(
                    "dataset '{}' dimension {i} is {got}, the graph expects {want}",
                    d.name
                )));
            }
        }
    }
    Ok(())
}

/// Read, hash-check and validate the graph input.
pub fn load_graph(
    p: &GraphGradParamsV1,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<(ModelGraphV1, Prepared), ComputeError> {
    let path = inputs
        .get(GRAPH_INPUT)
        .ok_or_else(|| ComputeError::Inputs("input file 'graph' not provided".into()))?;
    let meta = std::fs::metadata(path)?;
    if meta.len() != p.graph_bytes || meta.len() > MAX_GRAPH_BYTES {
        return Err(ComputeError::Inputs(format!(
            "graph file is {} bytes, expected {}",
            meta.len(),
            p.graph_bytes
        )));
    }
    let bytes = std::fs::read(path)?;
    if Sha256Digest::of_bytes(&bytes).to_hex() != p.graph_hash {
        return Err(ComputeError::Inputs("graph hash mismatch".into()));
    }
    let graph: ModelGraphV1 = serde_json::from_slice(&bytes)
        .map_err(|e| ComputeError::Inputs(format!("graph is not gg-graph/v1: {e}")))?;
    let prepared = Prepared::new(&graph).map_err(|e| invalid(format!("graph refused: {e}")))?;
    check_declared(&graph, &prepared, p)?;
    Ok((graph, prepared))
}

/// Checkpoint sharding requested by the task (C2).
pub fn shard_plan(
    graph: &ModelGraphV1,
    prepared: &Prepared,
    p: &GraphGradParamsV1,
) -> Result<super::shards::ShardPlan, ComputeError> {
    super::shards::plan(graph, prepared.report(), p.shard_bytes).map_err(invalid)
}

/// Read and check the checkpoint, shard by shard, against the graph.
pub fn load_checkpoint(
    graph: &ModelGraphV1,
    prepared: &Prepared,
    p: &GraphGradParamsV1,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<BTreeMap<String, TensorF32>, ComputeError> {
    let plan = shard_plan(graph, prepared, p)?;
    let mut out = BTreeMap::new();
    for (i, shard) in plan.shards.iter().enumerate() {
        let name = plan.checkpoint_name(i);
        let path = inputs
            .get(&name)
            .ok_or_else(|| ComputeError::Inputs(format!("input file '{name}' not provided")))?;
        let meta = std::fs::metadata(path)?;
        if meta.len() != shard.checkpoint_bytes {
            return Err(ComputeError::Inputs(format!(
                "{name} is {} bytes, expected {}",
                meta.len(),
                shard.checkpoint_bytes
            )));
        }
        let bundle = BundleF32::read_from_path(
            path,
            &TensorLimits {
                max_elements: p.parameters,
            },
        )
        .map_err(|source| ComputeError::InputTensor {
            name: name.clone(),
            source,
        })?;
        check_shard(graph, shard, &bundle, &name, false)?;
        out.extend(bundle.entries);
    }
    Ok(out)
}

/// A shard holds exactly its parameters (or only its trained ones, for a gradient), with their
/// shapes, and finite values.
pub fn check_shard(
    graph: &ModelGraphV1,
    shard: &super::shards::Shard,
    bundle: &BundleF32,
    what: &str,
    trainable_only: bool,
) -> Result<(), ComputeError> {
    let expected: Vec<&super::ParamSpec> = shard
        .params
        .iter()
        .filter_map(|n| graph.parameters.iter().find(|q| &q.name == n))
        .filter(|q| !trainable_only || q.trainable)
        .collect();
    check_entries(bundle, &expected, what)
}

/// A bundle holds exactly the graph's parameters, with their shapes, and finite values.
pub fn check_bundle_matches(
    graph: &ModelGraphV1,
    bundle: &BundleF32,
    what: &str,
) -> Result<(), ComputeError> {
    check_bundle_subset(graph, bundle, what, false)
}

/// Same check, restricted to the trained parameters: that is a gradient's layout.
pub fn check_gradient_bundle(
    graph: &ModelGraphV1,
    bundle: &BundleF32,
    what: &str,
) -> Result<(), ComputeError> {
    check_bundle_subset(graph, bundle, what, true)
}

fn check_bundle_subset(
    graph: &ModelGraphV1,
    bundle: &BundleF32,
    what: &str,
    trainable_only: bool,
) -> Result<(), ComputeError> {
    let expected: Vec<&super::ParamSpec> = graph
        .parameters
        .iter()
        .filter(|q| !trainable_only || q.trainable)
        .collect();
    check_entries(bundle, &expected, what)
}

fn check_entries(
    bundle: &BundleF32,
    expected: &[&super::ParamSpec],
    what: &str,
) -> Result<(), ComputeError> {
    if bundle.entries.len() != expected.len() {
        return Err(ComputeError::Inputs(format!(
            "{what} has {} tensors, {} expected",
            bundle.entries.len(),
            expected.len()
        )));
    }
    for q in expected {
        let t = bundle
            .entries
            .get(&q.name)
            .ok_or_else(|| ComputeError::Inputs(format!("{what} lacks '{}'", q.name)))?;
        t.check_shape(&q.shape)
            .and_then(|_| t.check_finite())
            .map_err(|source| ComputeError::InputTensor {
                name: format!("{what}/{}", q.name),
                source,
            })?;
    }
    Ok(())
}

/// Load the micro-batch: every dataset tensor, sliced by the round's index stream.
pub fn load_batch(
    task: &ValidatedTask,
    p: &GraphGradParamsV1,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<BTreeMap<String, TensorF32>, ComputeError> {
    let idx = round_indices(
        p.round_seed,
        p.total_examples,
        p.example_start,
        p.example_count,
    );
    let mut out = BTreeMap::new();
    for d in &p.datasets {
        let shape = task
            .input_shapes
            .get(&d.name)
            .ok_or_else(|| ComputeError::Inputs(format!("no expected shape for '{}'", d.name)))?;
        let path = inputs
            .get(&d.name)
            .ok_or_else(|| ComputeError::Inputs(format!("input file '{}' not provided", d.name)))?;
        let t = TensorF32::read_from_path(path, &tensor_limits_for(shape)).map_err(|source| {
            ComputeError::InputTensor {
                name: d.name.clone(),
                source,
            }
        })?;
        t.check_shape(shape)
            .and_then(|_| t.check_finite())
            .map_err(|source| ComputeError::InputTensor {
                name: d.name.clone(),
                source,
            })?;
        // Micro-batch already selected by the coordinator (C4): its rows, in stream order.
        let rows = if p.batch_inputs {
            t
        } else {
            select_rows(&t, &idx)
        };
        out.insert(d.name.clone(), rows);
    }
    Ok(out)
}

fn params_of(task: &ValidatedTask) -> Result<&GraphGradParamsV1, ComputeError> {
    match &task.params {
        TaskParamsV1::GraphGrad(p) => Ok(p),
        other => Err(ComputeError::UnknownKind(other.compute_kind().to_owned())),
    }
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

/// Gradient of the micro-batch on the given engine.
fn compute(
    engine: EngineSelection,
    prepared: &Prepared,
    params: BTreeMap<String, TensorF32>,
    batch: BTreeMap<String, TensorF32>,
    seed: u64,
    options: ExecOptions,
) -> Result<(exec::StepResult, EngineInfo), ComputeError> {
    let to_err = |e: exec::ExecError| ComputeError::InvalidParams(format!("graph execution: {e}"));
    match engine {
        EngineSelection::FlexCpu => {
            let device = burn::backend::flex::FlexDevice;
            let r = exec::gradient_owned_with::<burn::backend::Autodiff<burn::backend::Flex>>(
                &device, prepared, params, batch, seed, options,
            )
            .map_err(to_err)?;
            Ok((r, cpu_info()))
        }
        #[cfg(feature = "gpu")]
        EngineSelection::WgpuVulkan { device_index } => {
            let (device, info) = crate::engine::gpu::init_vulkan(device_index)?;
            let r = exec::gradient_owned_with::<burn::backend::Autodiff<burn::backend::Vulkan>>(
                &device, prepared, params, batch, seed, options,
            );
            crate::engine::gpu::release_memory(&device);
            Ok((r.map_err(to_err)?, info))
        }
        #[cfg(feature = "cuda")]
        EngineSelection::TorchCuda { device_index } => {
            let info = crate::engine::cuda::check(device_index)?;
            let r =
                exec::torch::gradient_cuda(device_index, prepared, params, batch, seed, options)
                    .map_err(to_err)?;
            Ok((r, info))
        }
        #[cfg(not(feature = "cuda"))]
        EngineSelection::TorchCuda { .. } => Err(ComputeError::EngineUnavailable(
            "this build has no native CUDA engine (feature `cuda`)".into(),
        )),
        #[cfg(not(feature = "gpu"))]
        EngineSelection::WgpuVulkan { .. } => Err(ComputeError::EngineUnavailable(
            "this build has no GPU support (feature `gpu`)".into(),
        )),
    }
}

/// Execution options requested by the task.
pub fn exec_options(p: &GraphGradParamsV1) -> ExecOptions {
    ExecOptions {
        segments: p.segments.max(1) as usize,
        offload_parameters: p.offload_parameters,
    }
}

/// Execute a `graph.grad.v1` task.
pub fn execute(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
) -> Result<ExecutionReport, ComputeError> {
    execute_limited(engine, task, inputs, out_dir, None)
}

/// Execute a `graph.grad.v1` task within `device_memory` bytes of accelerator memory (C1): the
/// task is refused before any computation when its estimate exceeds the limit, and the measured
/// peak is checked afterwards. This is what lets a worker behave as the small GPU it declares.
pub fn execute_limited(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
    device_memory: Option<u64>,
) -> Result<ExecutionReport, ComputeError> {
    let p = params_of(task)?;
    let (graph, prepared) = load_graph(p, inputs)?;
    if let Some(limit) = device_memory {
        let est = super::memory::task_memory(
            &graph,
            prepared.report(),
            p.example_count as u64,
            exec_options(p),
        );
        if est.total() > limit {
            return Err(ComputeError::MemoryLimit(format!(
                "the task needs about {} bytes (parameters {}, gradients {}, activations {}, \
batch {}), this executor may use {limit}",
                est.total(),
                est.parameters,
                est.gradients,
                est.activations,
                est.datasets
            )));
        }
    }
    let params = load_checkpoint(&graph, &prepared, p, inputs)?;
    let batch = load_batch(task, p, inputs)?;
    let start = Instant::now();
    let (step, info) = compute(
        engine,
        &prepared,
        params,
        batch,
        dropout_seed(p),
        exec_options(p),
    )?;
    let duration_ms = start.elapsed().as_millis() as u64;
    if let Some(limit) = device_memory {
        let measured = step.peak_activation_bytes + step.peak_parameter_bytes;
        if measured > limit {
            return Err(ComputeError::MemoryLimit(format!(
                "measured peak of {measured} bytes above the limit of {limit}"
            )));
        }
    }

    let plan = shard_plan(&graph, &prepared, p)?;
    let mut gradients = step.gradients;
    let mut outputs = BTreeMap::new();
    let mut budget = task.limits.max_output_bytes;
    for (i, shard) in plan.trained() {
        let part: BTreeMap<String, TensorF32> = shard
            .params
            .iter()
            .filter_map(|n| gradients.remove_entry(n))
            .collect();
        let bundle = BundleF32::new(part)
            .map_err(|e| ComputeError::OutputLimit(format!("gradient bundle: {e}")))?;
        bundle
            .check_finite()
            .map_err(|e| ComputeError::OutputLimit(format!("non-finite gradient: {e}")))?;
        let bytes = bundle
            .encode_as(p.gradient_dtype())
            .map_err(|e| ComputeError::OutputLimit(format!("gradient shard {i}: {e}")))?;
        if bytes.len() as u64 != shard.gradient_size(p.gradient_dtype()) {
            return Err(ComputeError::OutputLimit(format!(
                "gradient shard {i} is {} bytes, the declared layout is {}",
                bytes.len(),
                shard.gradient_size(p.gradient_dtype())
            )));
        }
        let name = plan.gradient_name(i);
        let f = write_file(out_dir, &format!("{name}.ggb"), &bytes, budget)?;
        budget = budget.saturating_sub(f.size);
        outputs.insert(name, f);
    }
    if !gradients.is_empty() {
        return Err(ComputeError::OutputLimit(format!(
            "gradients outside every shard: {:?}",
            gradients.keys().collect::<Vec<_>>()
        )));
    }
    let meta = GraphGradMeta {
        kind: KIND.into(),
        n_examples: p.example_count,
        loss: step.loss,
        numeric_profile: info.numeric_profile.to_owned(),
        parameters: p.parameters,
        peak_activation_bytes: step.peak_activation_bytes,
        peak_parameter_bytes: step.peak_parameter_bytes,
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
    metrics.insert("loss".to_owned(), step.loss as f64);
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
    reference_engine: EngineSelection,
) -> Result<Verdict, ComputeError> {
    let p = params_of(task)?;
    let mut checks = Vec::new();
    let mut artifacts_checked = Vec::new();
    let reject = |checks: Vec<CheckV1>, a: Vec<ArtifactRef>| {
        Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Rejected,
            artifacts_checked: a,
        })
    };
    // The expected outputs depend on the sharding, hence on the description.
    let (graph, prepared) = load_graph(p, inputs)?;
    let plan = shard_plan(&graph, &prepared, p)?;
    let mut expected: Vec<(String, u64)> = plan
        .trained()
        .map(|(i, s)| (plan.gradient_name(i), s.gradient_size(p.gradient_dtype())))
        .collect();
    expected.push((META_NAME.to_owned(), MAX_META_BYTES));
    expected.sort();
    let expected_names: Vec<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    let claimed_names: Vec<&str> = claimed.keys().map(String::as_str).collect();
    if expected_names != claimed_names {
        checks.push(fail(
            "outputs.names",
            format!("expected {expected_names:?}, got {claimed_names:?}"),
        ));
        return reject(checks, artifacts_checked);
    }
    let mut bytes_by_name: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    for (name, max) in &expected {
        let name = name.as_str();
        let max = *max;
        let r = claimed[name];
        let Some(path) = candidates.get(name) else {
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
    let meta: GraphGradMeta = match serde_json::from_slice(&bytes_by_name[META_NAME]) {
        Ok(m) => m,
        Err(e) => {
            checks.push(fail("outputs.grad_meta.format", e.to_string()));
            return reject(checks, artifacts_checked);
        }
    };
    if meta.kind != KIND
        || meta.n_examples != p.example_count
        || meta.parameters != p.parameters
        || !meta.loss.is_finite()
    {
        checks.push(fail(
            "outputs.grad_meta.structure",
            format!(
                "kind {}, n_examples {}, parameters {}",
                meta.kind, meta.n_examples, meta.parameters
            ),
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

    let mut candidate: BTreeMap<String, TensorF32> = BTreeMap::new();
    for (i, shard) in plan.trained() {
        let name = plan.gradient_name(i);
        // Typed: a gradient in the other element type is refused, not converted.
        let bundle = match BundleF32::decode_as(
            &bytes_by_name[name.as_str()],
            &TensorLimits {
                max_elements: p.parameters,
            },
            p.gradient_dtype(),
        ) {
            Ok(b) => b,
            Err(e) => {
                checks.push(fail(&format!("outputs.{name}.format"), e.to_string()));
                return reject(checks, artifacts_checked);
            }
        };
        if let Err(e) = check_shard(&graph, shard, &bundle, &name, true) {
            checks.push(fail(&format!("outputs.{name}.layout"), e.to_string()));
            return reject(checks, artifacts_checked);
        }
        candidate.extend(bundle.entries);
    }
    let params = load_checkpoint(&graph, &prepared, p, inputs)?;
    let batch = load_batch(task, p, inputs)?;
    // The validator recomputes with the same levers as the worker: they do not change the
    // gradient, but recomputing without them would require the memory the worker did not have.
    let (reference, _) = compute(
        reference_engine,
        &prepared,
        params,
        batch,
        dropout_seed(p),
        exec_options(p),
    )?;
    checks.push(pass(
        "reference.recomputed",
        format!(
            "profile {} over {} examples, {} parameters",
            reference_engine.numeric_profile(),
            p.example_count,
            p.parameters
        ),
    ));
    // A bf16 gradient is compared, rounded, with the f32 recomputation: the rounding (≤ 2⁻⁹ of each
    // element) takes about 0.2 % of the norm out of the tolerance (ADR-0035).
    let mut all_ok = true;
    for (name, r) in &reference.gradients {
        let cmp = compare_f32_norm(
            &format!("outputs.gradient.{name}"),
            &candidate[name],
            r,
            task.tolerance,
        );
        all_ok &= cmp.passed;
        // A graph may have hundreds of parameters: keep the failing checks, and a summary for
        // the passing ones, under the protocol bound.
        if !cmp.passed || checks.len() < 64 {
            checks.extend(cmp.checks);
        }
    }
    checks.push(pass(
        "outputs.gradient.compared",
        format!(
            "{} tensors within ‖Δ‖ ≤ {}·√n + {}·‖ref‖",
            reference.gradients.len(),
            task.tolerance.abs,
            task.tolerance.rel
        ),
    ));
    let diff = (meta.loss as f64 - reference.loss as f64).abs();
    let tol = task.tolerance.abs + task.tolerance.rel * (reference.loss as f64).abs();
    let ok = diff <= tol;
    all_ok &= ok;
    checks.push(CheckV1 {
        name: "grad_meta.loss".into(),
        passed: ok,
        measured: Some(diff),
        tolerance: Some(tol),
        detail: format!("declared {}, recomputed {}", meta.loss, reference.loss),
    });
    checks.truncate(gg_protocol::messages::MAX_CHECKS);
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

/// Named tensors, as a checkpoint or a gradient.
pub type Named = BTreeMap<String, TensorF32>;

/// Host-side optimizer state over named parameters (`adam-host-v1`, ADR-0011).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NamedOptimizerState {
    /// Adam first moments (absent for SGD).
    pub m: Option<Named>,
    /// Adam second moments.
    pub v: Option<Named>,
    /// Steps applied so far.
    pub t: u32,
}

fn zeros_like(w: &Named) -> Named {
    w.iter()
        .map(|(k, t)| {
            (
                k.clone(),
                TensorF32::new(t.shape.clone(), vec![0.0; t.data.len()]).expect("shape"),
            )
        })
        .collect()
}

fn same_layout(a: &Named, b: &Named) -> bool {
    a.len() == b.len()
        && a.iter()
            .all(|(k, t)| b.get(k).is_some_and(|u| u.shape == t.shape))
}

/// The gradient covers the trained parameters, hence part of the checkpoint.
fn covers(weights: &Named, g: &Named) -> bool {
    g.iter()
        .all(|(k, t)| weights.get(k).is_some_and(|w| w.shape == t.shape))
}

/// Elements in one unit of work of the server's step: large enough that spreading costs nothing,
/// small enough to balance hundreds of tensors of mixed sizes across threads.
const PAR_CHUNK: usize = 1 << 16;

/// Runs `f` on every item, spread across the machine's threads. Items are independent pieces of
/// element-wise work: each element sees the same operations in the same order as on one thread,
/// so the result is bit for bit the same.
fn par_each<T: Send>(items: Vec<T>, f: impl Fn(T) + Sync) {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(16)
        .min(items.len());
    if threads <= 1 {
        items.into_iter().for_each(f);
        return;
    }
    let mut buckets: Vec<Vec<T>> = (0..threads).map(|_| Vec::new()).collect();
    for (i, item) in items.into_iter().enumerate() {
        buckets[i % threads].push(item);
    }
    let f = &f;
    std::thread::scope(|s| {
        for bucket in buckets {
            s.spawn(move || bucket.into_iter().for_each(f));
        }
    });
}

/// Weighted mean of micro-batch gradients (weights = example counts), one pass.
pub fn aggregate_named(parts: &[(u32, Named)]) -> Result<Named, ComputeError> {
    let (_, first) = parts
        .first()
        .ok_or_else(|| ComputeError::InvalidParams("no gradients to aggregate".into()))?;
    let mut total: f64 = 0.0;
    for (n, g) in parts {
        if !same_layout(first, g) {
            return Err(ComputeError::InvalidParams(
                "gradient layouts differ".into(),
            ));
        }
        total += *n as f64;
    }
    if total == 0.0 {
        return Err(ComputeError::InvalidParams("zero examples".into()));
    }
    let inv = (1.0 / total) as f32;
    let mut acc = zeros_like(first);
    let mut items = Vec::new();
    for (k, t) in acc.iter_mut() {
        for (c, chunk) in t.data.chunks_mut(PAR_CHUNK).enumerate() {
            items.push((k.as_str(), c * PAR_CHUNK, chunk));
        }
    }
    par_each(items, |(k, start, chunk)| {
        for (n, g) in parts {
            let w = *n as f32;
            let src = &g[k].data[start..start + chunk.len()];
            for (a, gi) in chunk.iter_mut().zip(src) {
                *a += w * gi;
            }
        }
        for a in chunk.iter_mut() {
            *a *= inv;
        }
    });
    Ok(acc)
}

/// The moment update of one Adam step, without the weights: `m`, `v` and `t` as
/// [`apply_step_named`] leaves them, bit for bit. The server uses it to bring optimizer state that
/// was not yet durable back to the weights' step from the round's recorded gradients (ADR-0036).
/// Without state (SGD) only `t` advances.
pub fn advance_moments_named(
    optim: CoordOptim,
    g: &Named,
    state: &mut NamedOptimizerState,
) -> Result<(), ComputeError> {
    state.t += 1;
    if optim == CoordOptim::Sgd {
        return Ok(());
    }
    let (b1, b2) = (0.9f32, 0.999f32);
    let NamedOptimizerState { m, v, .. } = state;
    let m = m.get_or_insert_with(|| zeros_like(g));
    let v = v.get_or_insert_with(|| zeros_like(g));
    if !same_layout(m, g) || !same_layout(v, g) {
        return Err(ComputeError::InvalidParams(
            "optimizer state does not match the gradient layout".into(),
        ));
    }
    let mut items = Vec::new();
    for ((k, mk), vk) in m.iter_mut().zip(v.values_mut()) {
        items.extend(
            mk.data
                .chunks_mut(PAR_CHUNK)
                .zip(vk.data.chunks_mut(PAR_CHUNK))
                .zip(g[k].data.chunks(PAR_CHUNK)),
        );
    }
    par_each(items, |((m, v), g)| {
        for ((mi, vi), gi) in m.iter_mut().zip(v).zip(g) {
            *mi = b1 * *mi + (1.0 - b1) * gi;
            *vi = b2 * *vi + (1.0 - b2) * gi * gi;
        }
    });
    Ok(())
}

/// One optimizer step on named parameters. Same arithmetic as `grad::apply_step`.
pub fn apply_step_named(
    optim: CoordOptim,
    lr: f64,
    weights: &mut Named,
    g: &Named,
    state: &mut NamedOptimizerState,
) -> Result<(), ComputeError> {
    // The gradient may cover only part of the checkpoint: frozen parameters keep their value,
    // have no optimizer state and are left untouched (ADR-0019).
    if g.is_empty() || !covers(weights, g) {
        return Err(ComputeError::InvalidParams(
            "gradient does not match the checkpoint".into(),
        ));
    }
    state.t += 1;
    match optim {
        CoordOptim::Sgd => {
            let lr = lr as f32;
            let mut items = Vec::new();
            for (k, w) in weights.iter_mut() {
                if let Some(gk) = g.get(k) {
                    items.extend(w.data.chunks_mut(PAR_CHUNK).zip(gk.data.chunks(PAR_CHUNK)));
                }
            }
            par_each(items, |(w, g)| {
                for (wi, gi) in w.iter_mut().zip(g) {
                    *wi -= lr * gi;
                }
            });
        }
        CoordOptim::Adam => {
            let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
            // The optimizer state follows the gradient layout, hence only the trained
            // parameters: this is where freezing saves the most memory on the coordinator side.
            let NamedOptimizerState { m, v, t } = state;
            let m = m.get_or_insert_with(|| zeros_like(g));
            let v = v.get_or_insert_with(|| zeros_like(g));
            if !same_layout(m, g) || !same_layout(v, g) {
                return Err(ComputeError::InvalidParams(
                    "optimizer state does not match the gradient layout".into(),
                ));
            }
            let t = *t as i32;
            let c1 = 1.0 - b1.powi(t);
            let c2 = 1.0 - b2.powi(t);
            let lr = lr as f32;
            // One pass per element, in the order of the three passes it replaces: moments, then
            // the weight. m, v and g share their keys (checked), in the same sorted order.
            let mut moments = m.iter_mut().zip(v.values_mut());
            let mut items = Vec::new();
            for (k, w) in weights.iter_mut() {
                // A frozen parameter has no gradient and no state: it goes through unchanged.
                let Some(gk) = g.get(k) else {
                    continue;
                };
                let ((mk_name, mk), vk) = moments.next().expect("same layout as the gradient");
                debug_assert_eq!(mk_name, k);
                items.extend(
                    w.data
                        .chunks_mut(PAR_CHUNK)
                        .zip(mk.data.chunks_mut(PAR_CHUNK))
                        .zip(vk.data.chunks_mut(PAR_CHUNK))
                        .zip(gk.data.chunks(PAR_CHUNK)),
                );
            }
            par_each(items, |(((w, m), v), g)| {
                for (((wi, mi), vi), gi) in w.iter_mut().zip(m).zip(v).zip(g) {
                    *mi = b1 * *mi + (1.0 - b1) * gi;
                    *vi = b2 * *vi + (1.0 - b2) * gi * gi;
                    let mhat = *mi / c1;
                    let vhat = *vi / c2;
                    *wi -= lr * mhat / (vhat.sqrt() + eps);
                }
            });
        }
    }
    Ok(())
}
