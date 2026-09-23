//! Catalog of approved compute kinds: schema validation, bounds, resource estimation and pricing.

use std::collections::BTreeMap;

use gg_core::{Credits, ResourceLimits};
use gg_protocol::messages::{
    ArtifactRef, GraphGradParamsV1, GraphStageParamsV1, MlpGradParamsV1, MlpTrainParamsV1,
    RewardScheduleV1, SeedStrategyV1, TaskDefinitionV1, TaskParamsV1, ToleranceV1,
    ValidationPolicyV1,
};
use gg_protocol::tensor_file;

use crate::error::ComputeError;
use crate::profiles;

/// Catalog version. Bumped whenever bounds, pricing or verification rules change.
///
/// Version 2 (ADR-0031): model tasks (`graph.grad.v1`, `graph.stage.v1`) are compared per tensor by
/// the norm of the difference (`recompute-reference` v2), no longer element by element.
pub const CATALOG_VERSION: u32 = 2;

/// Maximum matrix dimension for `tensor.matmul.v1`.
pub const MATMUL_MAX_DIM: u32 = 2048;
/// Maximum absolute tolerance accepted in a validation policy.
pub const MAX_TOLERANCE: f64 = 1e-2;
/// Maximum relative norm tolerance of `recompute-reference` v2 (ADR-0031): the value the catalog
/// sets, measured, and no more.
pub const MAX_NORM_TOLERANCE: f64 = 2e-2;

/// Version of `recompute-reference` a compute kind is verified with: 1 compares element by
/// element (`|a − b| ≤ abs + rel·|b|`), 2 compares each tensor by the norm of the difference
/// (`‖a − b‖ ≤ rel·‖b‖ + abs·√n`, [`crate::compare::compare_f32_norm`]).
pub fn method_version(params: &TaskParamsV1) -> u32 {
    match params {
        TaskParamsV1::GraphGrad(_) | TaskParamsV1::GraphStage(_) => 2,
        _ => 1,
    }
}

/// Compute kinds available in this catalog version.
pub const KINDS: &[&str] = &[
    "tensor.matmul.v1",
    "mlp.train.v1",
    "mlp.grad.v1",
    "graph.grad.v1",
    "graph.stage.v1",
];

/// Bounds of `mlp.train.v1`.
pub mod mlp_bounds {
    /// Maximum input width.
    pub const MAX_D_INPUT: u32 = 256;
    /// Maximum hidden width.
    pub const MAX_D_HIDDEN: u32 = 512;
    /// Maximum output width.
    pub const MAX_D_OUTPUT: u32 = 64;
    /// Maximum batch size.
    pub const MAX_BATCH: u32 = 512;
    /// Maximum optimizer steps.
    pub const MAX_STEPS: u32 = 5000;
    /// Maximum examples per split.
    pub const MAX_EXAMPLES: u64 = 65_536;
}

/// Conservative resource estimate.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResourceEstimate {
    /// Floating point operations.
    pub flops: u64,
    /// Peak RAM in bytes (inputs + outputs + working set).
    pub ram_bytes: u64,
    /// Total output bytes.
    pub output_bytes: u64,
    /// Rough CPU seconds (conservative).
    pub est_cpu_secs: f64,
}

/// A definition validated against the catalog.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedTask {
    /// Parameters.
    pub params: TaskParamsV1,
    /// Expected input shapes by name.
    pub input_shapes: BTreeMap<String, Vec<u64>>,
    /// Expected tensor output shapes by name.
    pub output_shapes: BTreeMap<String, Vec<u64>>,
    /// Non-tensor (bounded JSON) outputs by name.
    pub json_outputs: Vec<String>,
    /// Estimate.
    pub estimate: ResourceEstimate,
    /// Tolerance from the policy.
    pub tolerance: ToleranceV1,
    /// Resource limits from the definition.
    pub limits: ResourceLimits,
}

fn ggt_size(shape: &[u64]) -> u64 {
    6 + 8 * shape.len() as u64 + 4 * shape.iter().product::<u64>()
}

/// Validate a task definition against the catalog. Unknown kinds are rejected explicitly.
pub fn validate_definition(def: &TaskDefinitionV1) -> Result<ValidatedTask, ComputeError> {
    def.validate_structure()?;
    if def.catalog_version != CATALOG_VERSION {
        return Err(ComputeError::InvalidParams(format!(
            "catalog_version {} not supported (current {CATALOG_VERSION})",
            def.catalog_version
        )));
    }
    if def.numeric_profile != profiles::REQUESTED_F32_V1 {
        return Err(ComputeError::ProfileNotAdmitted(
            def.numeric_profile.clone(),
        ));
    }
    let p = &def.validation_policy;
    let version = method_version(&def.params);
    if p.method != "recompute-reference" || p.method_version != version {
        return Err(ComputeError::InvalidParams(format!(
            "validation method must be recompute-reference v{version} for {}",
            def.params.compute_kind()
        )));
    }
    if p.reference_profile != profiles::REFERENCE_PROFILE {
        return Err(ComputeError::ProfileNotAdmitted(
            p.reference_profile.clone(),
        ));
    }
    let max_rel = if version == 2 {
        MAX_NORM_TOLERANCE
    } else {
        MAX_TOLERANCE
    };
    if !(p.tolerance.abs >= 0.0 && p.tolerance.rel >= 0.0)
        || p.tolerance.abs > MAX_TOLERANCE
        || p.tolerance.rel > max_rel
    {
        return Err(ComputeError::InvalidParams("tolerance too large".into()));
    }
    match &def.params {
        TaskParamsV1::Matmul { m, k, n } => validate_matmul(def, *m, *k, *n),
        TaskParamsV1::MlpTrain(p) => validate_mlp(def, p),
        TaskParamsV1::MlpGrad(p) => validate_grad(def, p),
        TaskParamsV1::GraphGrad(p) => validate_graph_grad(def, p),
        TaskParamsV1::GraphStage(p) => validate_graph_stage(def, p),
    }
}

fn validate_graph_stage(
    def: &TaskDefinitionV1,
    p: &GraphStageParamsV1,
) -> Result<ValidatedTask, ComputeError> {
    use crate::graph::task as gt;
    let bad = |m: &str| Err(ComputeError::InvalidParams(m.into()));
    if p.example_count == 0 || p.example_count > gt::MAX_EXAMPLE_COUNT {
        return bad("example_count out of range");
    }
    if p.total_examples == 0 || p.total_examples > gt::MAX_TOTAL_EXAMPLES {
        return bad("total_examples out of range");
    }
    if p.example_start > 16 * gt::MAX_TOTAL_EXAMPLES {
        return bad("example_start too large");
    }
    if p.graph_bytes == 0 || p.graph_bytes > gt::MAX_GRAPH_BYTES {
        return bad("graph_bytes out of range");
    }
    if p.parameters == 0 || p.parameters > crate::graph::bounds::MAX_PARAMETERS {
        return bad("parameters out of range");
    }
    let b = &p.stage_bounds;
    if b.len() < 2
        || b.len() > crate::graph::stages::MAX_STAGES + 1
        || b[0] != 0
        || b.windows(2).any(|w| w[0] >= w[1])
        || p.stage as usize + 1 >= b.len()
    {
        return bad("stage_bounds or stage out of range");
    }
    // A stage may do nothing but move data (Concat of tiles, Reshape): zero FLOPs is legitimate
    // here, and the minimum price of a task remains one credit.
    if p.datasets.is_empty() || p.datasets.len() > gt::MAX_DATASETS {
        return bad("datasets out of range");
    }
    for d in &p.datasets {
        if !matches!(d.dtype.as_str(), "f32" | "i64")
            || d.shape.is_empty()
            || d.shape.len() > crate::graph::bounds::MAX_RANK
            || d.shape[0] != p.total_examples
            || d.shape.contains(&0)
        {
            return Err(ComputeError::InvalidParams(format!(
                "dataset '{}' invalid",
                d.name
            )));
        }
    }
    match def.seed_strategy {
        SeedStrategyV1::ProtocolDrbg { seed } if seed == p.round_seed => {}
        _ => return bad("seed_strategy must be protocol_drbg with round_seed"),
    }
    let g = def
        .inputs
        .get(gt::GRAPH_INPUT)
        .ok_or_else(|| ComputeError::Inputs("missing input 'graph'".into()))?;
    if g.hash.to_hex() != p.graph_hash || g.size != p.graph_bytes {
        return Err(ComputeError::Inputs(
            "graph input does not match graph_hash/graph_bytes".into(),
        ));
    }
    // The other inputs (shards, micro-batch rows, activations, received gradients) depend on the
    // description: their exact list and shapes are checked when it is loaded.
    let estimate = estimate_graph_stage(p);
    if estimate.output_bytes > def.resource_limits.max_output_bytes {
        return Err(ComputeError::OutputLimit(
            "max_output_bytes below the output size".into(),
        ));
    }
    Ok(ValidatedTask {
        params: def.params.clone(),
        input_shapes: BTreeMap::new(),
        output_shapes: BTreeMap::new(),
        json_outputs: vec![crate::graph::stage_task::META_NAME.to_owned()],
        estimate,
        tolerance: def.validation_policy.tolerance,
        limits: def.resource_limits.clone(),
    })
}

/// Estimate for a stage task: a forward pass costs about a third of a training step, a backward
/// pass (which recomputes the stage) the whole of it.
pub fn estimate_graph_stage(p: &GraphStageParamsV1) -> ResourceEstimate {
    let n = p.example_count as u64;
    let step = p.training_flops_per_example.saturating_mul(n);
    let flops = if p.backward { step } else { step / 3 + 1 };
    let activations = p.activation_bytes_per_example.saturating_mul(n);
    // A lower bound, like the host memory below: what a stage writes depends on its structure,
    // and `graph_stage_definition` receives it.
    let output_bytes = crate::graph::task::MAX_META_BYTES;
    ResourceEstimate {
        flops,
        // A lower bound: the stage's own parameters are known only from the graph, and
        // `graph_stage_definition` receives the limit that includes them.
        ram_bytes: activations.saturating_add(EXECUTOR_RAM_BYTES),
        output_bytes,
        est_cpu_secs: flops as f64 / 2e9,
    }
}

fn validate_graph_grad(
    def: &TaskDefinitionV1,
    p: &GraphGradParamsV1,
) -> Result<ValidatedTask, ComputeError> {
    use crate::graph::task as gt;
    if p.example_count == 0 || p.example_count > gt::MAX_EXAMPLE_COUNT {
        return Err(ComputeError::InvalidParams(format!(
            "example_count={} out of range 1..={}",
            p.example_count,
            gt::MAX_EXAMPLE_COUNT
        )));
    }
    if p.total_examples == 0 || p.total_examples > gt::MAX_TOTAL_EXAMPLES {
        return Err(ComputeError::InvalidParams(
            "total_examples out of range".into(),
        ));
    }
    if p.example_start > 16 * gt::MAX_TOTAL_EXAMPLES {
        return Err(ComputeError::InvalidParams(
            "example_start too large".into(),
        ));
    }
    if p.graph_bytes == 0 || p.graph_bytes > gt::MAX_GRAPH_BYTES {
        return Err(ComputeError::InvalidParams(
            "graph_bytes out of range".into(),
        ));
    }
    if p.parameters == 0 || p.parameters > crate::graph::bounds::MAX_PARAMETERS {
        return Err(ComputeError::InvalidParams(
            "parameters out of range".into(),
        ));
    }
    use crate::graph::shards::{MAX_SHARD_BYTES, MAX_SHARDS};
    if p.shard_bytes > MAX_SHARD_BYTES {
        return Err(ComputeError::InvalidParams(format!(
            "shard_bytes out of range 0..={MAX_SHARD_BYTES}"
        )));
    }
    let max_checkpoint = if p.shard_bytes == 0 {
        gt::MAX_CHECKPOINT_BYTES
    } else {
        MAX_SHARDS as u64 * MAX_SHARD_BYTES
    };
    if p.checkpoint_bytes < 8 || p.checkpoint_bytes > max_checkpoint {
        return Err(ComputeError::InvalidParams(format!(
            "checkpoint_bytes out of range 8..={max_checkpoint}"
        )));
    }
    if p.gradient_bytes < 8 || p.gradient_bytes > p.checkpoint_bytes {
        return Err(ComputeError::InvalidParams(format!(
            "gradient_bytes out of range 8..={} (a gradient covers the trained parameters only)",
            p.checkpoint_bytes
        )));
    }
    if p.segments == 0 || p.segments > gt::MAX_SEGMENTS {
        return Err(ComputeError::InvalidParams(format!(
            "segments out of range 1..={}",
            gt::MAX_SEGMENTS
        )));
    }
    if p.training_flops_per_example == 0 {
        return Err(ComputeError::InvalidParams(
            "training_flops_per_example must be > 0".into(),
        ));
    }
    if p.datasets.is_empty() || p.datasets.len() > gt::MAX_DATASETS {
        return Err(ComputeError::InvalidParams(format!(
            "datasets out of range 1..={}",
            gt::MAX_DATASETS
        )));
    }
    match def.seed_strategy {
        SeedStrategyV1::ProtocolDrbg { seed } if seed == p.round_seed => {}
        _ => {
            return Err(ComputeError::InvalidParams(
                "seed_strategy must be protocol_drbg with round_seed".into(),
            ));
        }
    }
    let get = |name: &str| {
        def.inputs
            .get(name)
            .ok_or_else(|| ComputeError::Inputs(format!("missing input '{name}'")))
    };
    let g = get(gt::GRAPH_INPUT)?;
    if g.hash.to_hex() != p.graph_hash || g.size != p.graph_bytes {
        return Err(ComputeError::Inputs(
            "graph input does not match graph_hash/graph_bytes".into(),
        ));
    }
    // Checkpoint: a single `checkpoint` input, or contiguous shards `checkpoint.0 …`. Their exact
    // split depends on the description and is checked when it is loaded; here, only their count
    // and the sum of their sizes.
    let shard_inputs = if p.shard_bytes == 0 {
        if get(gt::CHECKPOINT_INPUT)?.size != p.checkpoint_bytes {
            return Err(ComputeError::Inputs(
                "checkpoint input size does not match checkpoint_bytes".into(),
            ));
        }
        1
    } else {
        let mut n = 0usize;
        let mut total = 0u64;
        while let Some(r) = def.inputs.get(&format!("{}.{n}", gt::CHECKPOINT_INPUT)) {
            if r.size == 0 || r.size > MAX_SHARD_BYTES {
                return Err(ComputeError::Inputs(format!(
                    "checkpoint shard {n} size out of range"
                )));
            }
            total = total.saturating_add(r.size);
            n += 1;
        }
        if n == 0 || n > MAX_SHARDS || total != p.checkpoint_bytes {
            return Err(ComputeError::Inputs(format!(
                "{n} checkpoint shards totalling {total} bytes, expected 1..={MAX_SHARDS} totalling {}",
                p.checkpoint_bytes
            )));
        }
        n
    };
    let mut input_shapes = BTreeMap::new();
    let mut dataset_bytes = 0u64;
    let mut seen = std::collections::BTreeSet::new();
    for d in &p.datasets {
        if d.name == gt::GRAPH_INPUT
            || d.name.starts_with(gt::CHECKPOINT_INPUT)
            || !seen.insert(&d.name)
        {
            return Err(ComputeError::InvalidParams(format!(
                "dataset name '{}' invalid or duplicated",
                d.name
            )));
        }
        if !matches!(d.dtype.as_str(), "f32" | "i64") {
            return Err(ComputeError::InvalidParams(format!(
                "dataset '{}' dtype '{}'",
                d.name, d.dtype
            )));
        }
        if d.shape.is_empty()
            || d.shape.len() > crate::graph::bounds::MAX_RANK
            || d.shape[0] != p.total_examples
            || d.shape.contains(&0)
        {
            return Err(ComputeError::InvalidParams(format!(
                "dataset '{}' shape {:?} must be [total_examples, ...] without zero",
                d.name, d.shape
            )));
        }
        if ggt_size(&d.shape) > gt::MAX_DATASET_BYTES {
            return Err(ComputeError::Inputs(format!(
                "dataset '{}' too large",
                d.name
            )));
        }
        // Micro-batch already selected (C4): the input carries only its rows.
        let input_shape: Vec<u64> = if p.batch_inputs {
            std::iter::once(p.example_count as u64)
                .chain(d.shape[1..].iter().copied())
                .collect()
        } else {
            d.shape.clone()
        };
        let size = ggt_size(&input_shape);
        if get(&d.name)?.size != size {
            return Err(ComputeError::Inputs(format!(
                "input '{}' declared size {} does not match shape {:?} ({size} bytes)",
                d.name,
                get(&d.name)?.size,
                d.shape
            )));
        }
        dataset_bytes = dataset_bytes.saturating_add(size);
        input_shapes.insert(d.name.clone(), input_shape);
    }
    if def.inputs.len() != p.datasets.len() + 1 + shard_inputs {
        return Err(ComputeError::Inputs(
            "graph.grad.v1 takes exactly graph, the checkpoint (or its shards) and one input per dataset"
                .into(),
        ));
    }
    let estimate = estimate_graph_grad(p, dataset_bytes);
    if estimate.output_bytes > def.resource_limits.max_output_bytes {
        return Err(ComputeError::OutputLimit(
            "max_output_bytes below the output size".into(),
        ));
    }
    Ok(ValidatedTask {
        params: def.params.clone(),
        input_shapes,
        output_shapes: BTreeMap::new(),
        json_outputs: vec![gt::META_NAME.to_owned()],
        estimate,
        tolerance: def.validation_policy.tolerance,
        limits: def.resource_limits.clone(),
    })
}

/// Host memory of a `graph.grad.v1` task, which is also its address-space limit on the CPU engine:
/// the datasets up to three times (read, decoded, batched), the checkpoint six times (parameters,
/// optimizer state, gradients, working copies), the activations kept for the backward pass plus a
/// quarter for the transient gradients of the backward pass, and 256 MiB for the executor itself.
///
/// Measured peaks on the CPU engine, one task of 8 examples at 224×224 (virtual ≈ resident):
/// ResNet18 652 MiB for 1 024 (the floor), ViT-tiny 807 for 1 418, ConvNeXt-tiny 1 797 for 3 633.
/// Counting the activations three times had put ConvNeXt at 7 470 MiB, beyond every 4 GiB worker.
pub fn graph_grad_ram_bytes(checkpoint_bytes: u64, activations: u64, dataset_bytes: u64) -> u64 {
    dataset_bytes
        .saturating_mul(3)
        .saturating_add(checkpoint_bytes.saturating_mul(6))
        .saturating_add(activations.saturating_add(activations / 4))
        .saturating_add(EXECUTOR_RAM_BYTES)
        .max(MIN_GRAPH_RAM_BYTES)
}

/// Host memory of the executor itself, outside the tensors of the task.
pub const EXECUTOR_RAM_BYTES: u64 = 256 << 20;

/// Smallest host-memory limit given to a model task.
pub const MIN_GRAPH_RAM_BYTES: u64 = 1 << 30;

/// Host memory of a `graph.stage.v1` task from its stage's memory estimate
/// ([`crate::graph::stages::stage_memory`], which counts only the stage's own parameters), plus
/// the executor. The task parameters name the whole graph's parameter count, not the stage's: a
/// limit derived from them grew with the whole model and made a staged model, split precisely
/// because no machine holds it, unplaceable.
///
/// Measured on the CPU engine: SmolLM2-135M in 2 stages, 8 sequences of 64 tokens, peak
/// 2 684 MiB over the forward and backward waves, for a limit of 4 319 MiB (the former limit,
/// derived from the whole model and four times the activations, was 14 987 MiB).
pub fn graph_stage_ram_bytes(stage_memory_bytes: u64) -> u64 {
    stage_memory_bytes
        .saturating_add(EXECUTOR_RAM_BYTES)
        .max(MIN_GRAPH_RAM_BYTES)
}

/// Estimate for a graph gradient task. `dataset_bytes` is the size of the whole datasets, which the
/// executor reads entirely before slicing.
pub fn estimate_graph_grad(p: &GraphGradParamsV1, dataset_bytes: u64) -> ResourceEstimate {
    let flops = p
        .training_flops_per_example
        .saturating_mul(p.example_count as u64);
    let activations = p
        .activation_bytes_per_example
        .saturating_mul(p.example_count as u64);
    ResourceEstimate {
        flops,
        ram_bytes: graph_grad_ram_bytes(p.checkpoint_bytes, activations, dataset_bytes),
        output_bytes: p.gradient_bytes + crate::graph::task::MAX_META_BYTES,
        est_cpu_secs: flops as f64 / 2e9,
    }
}

fn validate_matmul(
    def: &TaskDefinitionV1,
    m: u32,
    k: u32,
    n: u32,
) -> Result<ValidatedTask, ComputeError> {
    for (name, v) in [("m", m), ("k", k), ("n", n)] {
        if v == 0 || v > MATMUL_MAX_DIM {
            return Err(ComputeError::InvalidParams(format!(
                "{name}={v} out of range 1..={MATMUL_MAX_DIM}"
            )));
        }
    }
    if def.seed_strategy != SeedStrategyV1::None {
        return Err(ComputeError::InvalidParams(
            "tensor.matmul.v1 takes no seed".into(),
        ));
    }
    let a_shape = vec![m as u64, k as u64];
    let b_shape = vec![k as u64, n as u64];
    let c_shape = vec![m as u64, n as u64];
    let mut input_shapes = BTreeMap::new();
    for (name, shape) in [("a", &a_shape), ("b", &b_shape)] {
        let r = def
            .inputs
            .get(name)
            .ok_or_else(|| ComputeError::Inputs(format!("missing input '{name}'")))?;
        let expected = ggt_size(shape);
        if r.size != expected {
            return Err(ComputeError::Inputs(format!(
                "input '{name}' declared size {} does not match shape {shape:?} ({expected} bytes)",
                r.size
            )));
        }
        input_shapes.insert(name.to_owned(), shape.clone());
    }
    if def.inputs.len() != 2 {
        return Err(ComputeError::Inputs(
            "tensor.matmul.v1 takes exactly inputs 'a' and 'b'".into(),
        ));
    }
    let mut output_shapes = BTreeMap::new();
    output_shapes.insert("c".to_owned(), c_shape.clone());
    let estimate = estimate_matmul(m, k, n);
    if estimate.output_bytes > def.resource_limits.max_output_bytes {
        return Err(ComputeError::OutputLimit(
            "max_output_bytes below the output size".into(),
        ));
    }
    if estimate.ram_bytes > def.resource_limits.max_ram_bytes {
        return Err(ComputeError::InvalidParams(
            "max_ram_bytes below the estimated working set".into(),
        ));
    }
    Ok(ValidatedTask {
        params: def.params.clone(),
        input_shapes,
        output_shapes,
        json_outputs: Vec::new(),
        estimate,
        tolerance: def.validation_policy.tolerance,
        limits: def.resource_limits.clone(),
    })
}

fn rows_of(name: &str, r: &ArtifactRef, width: u64) -> Result<u64, ComputeError> {
    let header = 6 + 8 * 2;
    if r.size <= header || !(r.size - header).is_multiple_of(4 * width) {
        return Err(ComputeError::Inputs(format!(
            "input '{name}' size {} is not a [n, {width}] GGT1 tensor",
            r.size
        )));
    }
    let n = (r.size - header) / (4 * width);
    if n == 0 || n > mlp_bounds::MAX_EXAMPLES {
        return Err(ComputeError::Inputs(format!(
            "input '{name}' has {n} rows (max {})",
            mlp_bounds::MAX_EXAMPLES
        )));
    }
    Ok(n)
}

fn validate_mlp(
    def: &TaskDefinitionV1,
    p: &MlpTrainParamsV1,
) -> Result<ValidatedTask, ComputeError> {
    use mlp_bounds::*;
    for (name, v, max) in [
        ("d_input", p.d_input, MAX_D_INPUT),
        ("d_hidden", p.d_hidden, MAX_D_HIDDEN),
        ("d_output", p.d_output, MAX_D_OUTPUT),
        ("batch_size", p.batch_size, MAX_BATCH),
        ("steps", p.steps, MAX_STEPS),
    ] {
        if v == 0 || v > max {
            return Err(ComputeError::InvalidParams(format!(
                "{name}={v} out of range 1..={max}"
            )));
        }
    }
    if crate::mlp::Optim::parse(&p.optimizer).is_none() {
        return Err(ComputeError::InvalidParams(format!(
            "optimizer must be sgd or adam, got '{}'",
            p.optimizer
        )));
    }
    if !(p.learning_rate.is_finite() && p.learning_rate > 0.0 && p.learning_rate <= 1.0) {
        return Err(ComputeError::InvalidParams(
            "learning_rate must be in (0, 1]".into(),
        ));
    }
    match def.seed_strategy {
        SeedStrategyV1::ProtocolDrbg { seed } if seed == p.seed => {}
        _ => {
            return Err(ComputeError::InvalidParams(
                "seed_strategy must be protocol_drbg with the task seed".into(),
            ));
        }
    }
    let get = |name: &str| {
        def.inputs
            .get(name)
            .ok_or_else(|| ComputeError::Inputs(format!("missing input '{name}'")))
    };
    let n_train = rows_of("train_x", get("train_x")?, p.d_input as u64)?;
    if rows_of("train_y", get("train_y")?, p.d_output as u64)? != n_train {
        return Err(ComputeError::Inputs(
            "train_x and train_y row counts differ".into(),
        ));
    }
    let n_eval = rows_of("eval_x", get("eval_x")?, p.d_input as u64)?;
    if rows_of("eval_y", get("eval_y")?, p.d_output as u64)? != n_eval {
        return Err(ComputeError::Inputs(
            "eval_x and eval_y row counts differ".into(),
        ));
    }
    if def.inputs.len() != 4 {
        return Err(ComputeError::Inputs(
            "mlp.train.v1 takes exactly train_x, train_y, eval_x, eval_y".into(),
        ));
    }
    if p.batch_size as u64 > n_train {
        return Err(ComputeError::InvalidParams(
            "batch_size exceeds the number of training examples".into(),
        ));
    }
    let mut input_shapes = BTreeMap::new();
    input_shapes.insert("train_x".into(), vec![n_train, p.d_input as u64]);
    input_shapes.insert("train_y".into(), vec![n_train, p.d_output as u64]);
    input_shapes.insert("eval_x".into(), vec![n_eval, p.d_input as u64]);
    input_shapes.insert("eval_y".into(), vec![n_eval, p.d_output as u64]);
    let mut output_shapes = BTreeMap::new();
    output_shapes.insert("w1".into(), vec![p.d_input as u64, p.d_hidden as u64]);
    output_shapes.insert("b1".into(), vec![p.d_hidden as u64]);
    output_shapes.insert("w2".into(), vec![p.d_hidden as u64, p.d_output as u64]);
    output_shapes.insert("b2".into(), vec![p.d_output as u64]);
    let estimate = estimate_mlp(p, n_train, n_eval);
    if estimate.output_bytes > def.resource_limits.max_output_bytes {
        return Err(ComputeError::OutputLimit(
            "max_output_bytes below the output size".into(),
        ));
    }
    if estimate.ram_bytes > def.resource_limits.max_ram_bytes {
        return Err(ComputeError::InvalidParams(
            "max_ram_bytes below the estimated working set".into(),
        ));
    }
    Ok(ValidatedTask {
        params: def.params.clone(),
        input_shapes,
        output_shapes,
        json_outputs: vec![crate::mlp::METRICS_NAME.to_owned()],
        estimate,
        tolerance: def.validation_policy.tolerance,
        limits: def.resource_limits.clone(),
    })
}

fn validate_grad(
    def: &TaskDefinitionV1,
    p: &MlpGradParamsV1,
) -> Result<ValidatedTask, ComputeError> {
    use mlp_bounds::*;
    for (name, v, max) in [
        ("d_input", p.d_input, MAX_D_INPUT),
        ("d_hidden", p.d_hidden, MAX_D_HIDDEN),
        ("d_output", p.d_output, MAX_D_OUTPUT),
        ("example_count", p.example_count, MAX_BATCH),
    ] {
        if v == 0 || v > max {
            return Err(ComputeError::InvalidParams(format!(
                "{name}={v} out of range 1..={max}"
            )));
        }
    }
    if p.total_examples == 0 || p.total_examples > MAX_EXAMPLES {
        return Err(ComputeError::InvalidParams(
            "total_examples out of range".into(),
        ));
    }
    if p.example_start > 16 * MAX_EXAMPLES {
        return Err(ComputeError::InvalidParams(
            "example_start too large".into(),
        ));
    }
    match def.seed_strategy {
        SeedStrategyV1::ProtocolDrbg { seed } if seed == p.round_seed => {}
        _ => {
            return Err(ComputeError::InvalidParams(
                "seed_strategy must be protocol_drbg with round_seed".into(),
            ));
        }
    }
    let get = |name: &str| {
        def.inputs
            .get(name)
            .ok_or_else(|| ComputeError::Inputs(format!("missing input '{name}'")))
    };
    let n_train = rows_of("train_x", get("train_x")?, p.d_input as u64)?;
    if rows_of("train_y", get("train_y")?, p.d_output as u64)? != n_train
        || n_train != p.total_examples
    {
        return Err(ComputeError::Inputs(
            "train_x/train_y rows must equal total_examples".into(),
        ));
    }
    let shapes: [(&str, Vec<u64>); 4] = [
        ("w1", vec![p.d_input as u64, p.d_hidden as u64]),
        ("b1", vec![p.d_hidden as u64]),
        ("w2", vec![p.d_hidden as u64, p.d_output as u64]),
        ("b2", vec![p.d_output as u64]),
    ];
    let mut input_shapes = BTreeMap::new();
    for (name, shape) in &shapes {
        let r = get(name)?;
        if r.size != ggt_size(shape) {
            return Err(ComputeError::Inputs(format!(
                "checkpoint input '{name}' size {} does not match shape {shape:?}",
                r.size
            )));
        }
        input_shapes.insert((*name).to_owned(), shape.clone());
    }
    if def.inputs.len() != 6 {
        return Err(ComputeError::Inputs(
            "mlp.grad.v1 takes exactly w1, b1, w2, b2, train_x, train_y".into(),
        ));
    }
    input_shapes.insert("train_x".into(), vec![n_train, p.d_input as u64]);
    input_shapes.insert("train_y".into(), vec![n_train, p.d_output as u64]);
    let mut output_shapes = BTreeMap::new();
    for (name, shape) in &shapes {
        output_shapes.insert(format!("g_{name}"), shape.clone());
    }
    let estimate = estimate_grad(p, n_train);
    if estimate.output_bytes > def.resource_limits.max_output_bytes {
        return Err(ComputeError::OutputLimit(
            "max_output_bytes below the output size".into(),
        ));
    }
    Ok(ValidatedTask {
        params: def.params.clone(),
        input_shapes,
        output_shapes,
        json_outputs: vec![crate::grad::META_NAME.to_owned()],
        estimate,
        tolerance: def.validation_policy.tolerance,
        limits: def.resource_limits.clone(),
    })
}

/// Estimate for a gradient task.
pub fn estimate_grad(p: &MlpGradParamsV1, n_train: u64) -> ResourceEstimate {
    let (d_in, d_h, d_out) = (p.d_input as u64, p.d_hidden as u64, p.d_output as u64);
    let per_example = 2 * (d_in * d_h + d_h * d_out);
    let flops = 3 * per_example * p.example_count as u64;
    let weights = 4 * (d_in * d_h + d_h + d_h * d_out + d_out);
    let data_bytes = 4 * n_train * (d_in + d_out);
    ResourceEstimate {
        flops,
        ram_bytes: 3 * data_bytes + 16 * weights + (256 << 20),
        output_bytes: 4 * (6 + 16) + weights + crate::grad::MAX_META_BYTES,
        est_cpu_secs: flops as f64 / 2e9,
    }
}

/// Estimate for a matmul.
pub fn estimate_matmul(m: u32, k: u32, n: u32) -> ResourceEstimate {
    let (m, k, n) = (m as u64, k as u64, n as u64);
    let flops = 2 * m * k * n;
    let inputs = 4 * (m * k + k * n);
    let output = 4 * m * n;
    ResourceEstimate {
        flops,
        ram_bytes: 3 * (inputs + output) + (64 << 20),
        output_bytes: ggt_size(&[m, n]),
        est_cpu_secs: flops as f64 / 5e9,
    }
}

/// Estimate for an MLP training run.
pub fn estimate_mlp(p: &MlpTrainParamsV1, n_train: u64, n_eval: u64) -> ResourceEstimate {
    let (d_in, d_h, d_out) = (p.d_input as u64, p.d_hidden as u64, p.d_output as u64);
    let per_example = 2 * (d_in * d_h + d_h * d_out);
    let flops = 3 * per_example * p.batch_size as u64 * p.steps as u64 + per_example * n_eval;
    let data_bytes = 4 * ((n_train + n_eval) * (d_in + d_out));
    let weights = 4 * (d_in * d_h + d_h + d_h * d_out + d_out);
    let output_bytes = 4 * (6 + 16) + weights + crate::mlp::MAX_METRICS_BYTES;
    ResourceEstimate {
        flops,
        ram_bytes: 3 * data_bytes + 16 * weights + (256 << 20),
        output_bytes,
        est_cpu_secs: flops as f64 / 2e9,
    }
}

/// Reward schedule announced for a kind (fixed in advance, independent of declared metrics).
pub fn price(params: &TaskParamsV1) -> Result<RewardScheduleV1, ComputeError> {
    let flops = match params {
        TaskParamsV1::Matmul { m, k, n } => estimate_matmul(*m, *k, *n).flops,
        // Priced on the optimizer work only (dataset sizes are not part of the parameters).
        TaskParamsV1::MlpTrain(p) => estimate_mlp(p, 0, 0).flops,
        TaskParamsV1::MlpGrad(p) => estimate_grad(p, 0).flops,
        TaskParamsV1::GraphGrad(p) => estimate_graph_grad(p, 0).flops,
        TaskParamsV1::GraphStage(p) => estimate_graph_stage(p).flops,
    };
    // 1 TC per 10^9 flop, minimum 1 TC. Verification costs the same as a primary run.
    let primary = Credits(1 + flops / 1_000_000_000);
    Ok(RewardScheduleV1 {
        primary,
        replica: primary,
        audit: primary,
        verification_cost: primary,
    })
}

/// Default resource limits for a kind.
pub fn default_limits(params: &TaskParamsV1) -> Result<ResourceLimits, ComputeError> {
    match params {
        TaskParamsV1::Matmul { m, k, n } => {
            let est = estimate_matmul(*m, *k, *n);
            Ok(ResourceLimits {
                max_ram_bytes: est.ram_bytes.max(256 << 20),
                max_disk_bytes: (est.output_bytes * 2).max(64 << 20),
                max_duration_secs: 600,
                max_output_bytes: est.output_bytes,
            })
        }
        TaskParamsV1::MlpTrain(p) => {
            let est = estimate_mlp(p, mlp_bounds::MAX_EXAMPLES, mlp_bounds::MAX_EXAMPLES);
            Ok(ResourceLimits {
                max_ram_bytes: est.ram_bytes.max(512 << 20),
                max_disk_bytes: (est.output_bytes * 4).max(64 << 20),
                max_duration_secs: 1800,
                max_output_bytes: est.output_bytes,
            })
        }
        TaskParamsV1::MlpGrad(p) => {
            let est = estimate_grad(p, mlp_bounds::MAX_EXAMPLES);
            Ok(ResourceLimits {
                max_ram_bytes: est.ram_bytes.max(512 << 20),
                max_disk_bytes: (est.output_bytes * 4).max(64 << 20),
                max_duration_secs: 600,
                max_output_bytes: est.output_bytes,
            })
        }
        TaskParamsV1::GraphGrad(p) => {
            let est = estimate_graph_grad(p, 0);
            Ok(ResourceLimits {
                max_ram_bytes: est.ram_bytes.max(1 << 30),
                max_disk_bytes: (p.checkpoint_bytes * 2 + p.gradient_bytes * 2).max(256 << 20),
                max_duration_secs: 3600,
                max_output_bytes: est.output_bytes,
            })
        }
        TaskParamsV1::GraphStage(p) => {
            let est = estimate_graph_stage(p);
            Ok(ResourceLimits {
                max_ram_bytes: est.ram_bytes.max(1 << 30),
                max_disk_bytes: (p.parameters * 4 * 3 + est.output_bytes * 2).max(256 << 20),
                max_duration_secs: 3600,
                max_output_bytes: est.output_bytes,
            })
        }
    }
}

/// Tolerance of model tasks (`graph.grad.v1`, `graph.stage.v1`), compared per tensor by norm:
/// `‖a − b‖ ≤ rel·‖b‖ + abs·√n` (`recompute-reference` v2, ADR-0031).
pub fn graph_tolerance() -> ToleranceV1 {
    // Measured on 2026-09-24, ResNet18 at 224 px and 32 images, the widest gap of the pilot
    // models: GPU against the CPU reference 5.3e-3, the CPU reference itself against float64
    // 4.3e-3, PyTorch FP32 on CUDA against float64 7.1e-3. `rel` is four times the honest gap.
    // `abs` is a root-mean-square per element for gradients that are zero in exact arithmetic (a
    // key bias before a softmax, a bias before a normalization), where both sides are only
    // rounding noise; at the level of float32 rounding, so that it lets no real gradient through
    // (1e-4 did: a gradient of that order could be entirely wrong and pass). The element-wise
    // rule of v1 (1e-4 + 1e-3·|b|) was tighter than the rounding of its own reference once a
    // normalization sums over 32 × 112 × 112 values.
    ToleranceV1 {
        abs: 1e-7,
        rel: 2e-2,
    }
}

/// Default validation policy for a kind (tolerances documented in gg-server/docs/VALIDATION.md).
pub fn default_policy(params: &TaskParamsV1) -> Result<ValidationPolicyV1, ComputeError> {
    let tolerance = match params {
        TaskParamsV1::Matmul { .. } => ToleranceV1 {
            abs: 1e-4,
            rel: 1e-4,
        },
        TaskParamsV1::MlpTrain(_) => ToleranceV1 {
            abs: 1e-4,
            rel: 1e-3,
        },
        TaskParamsV1::MlpGrad(_) => ToleranceV1 {
            abs: 1e-5,
            rel: 1e-4,
        },
        // Per tensor, by norm (method v2, ADR-0031); a stage is a piece of the full gradient (D2).
        TaskParamsV1::GraphGrad(_) | TaskParamsV1::GraphStage(_) => graph_tolerance(),
    };
    Ok(ValidationPolicyV1 {
        method: "recompute-reference".into(),
        method_version: method_version(params),
        reference_profile: profiles::REFERENCE_PROFILE.into(),
        tolerance,
    })
}

/// Build a complete `mlp.train.v1` definition with catalog defaults. `inputs` must contain
/// `train_x`, `train_y`, `eval_x`, `eval_y`.
pub fn mlp_definition(
    p: MlpTrainParamsV1,
    inputs: BTreeMap<String, ArtifactRef>,
) -> Result<TaskDefinitionV1, ComputeError> {
    let seed = p.seed;
    let params = TaskParamsV1::MlpTrain(p);
    let def = TaskDefinitionV1 {
        catalog_version: CATALOG_VERSION,
        reward_schedule: price(&params)?,
        resource_limits: default_limits(&params)?,
        validation_policy: default_policy(&params)?,
        params,
        inputs,
        numeric_profile: profiles::REQUESTED_F32_V1.into(),
        seed_strategy: SeedStrategyV1::ProtocolDrbg { seed },
        max_attempts: 3,
    };
    validate_definition(&def)?;
    Ok(def)
}

/// Build a complete `mlp.grad.v1` definition. `inputs` must contain `w1,b1,w2,b2,train_x,train_y`.
pub fn grad_definition(
    p: MlpGradParamsV1,
    inputs: BTreeMap<String, ArtifactRef>,
) -> Result<TaskDefinitionV1, ComputeError> {
    let seed = p.round_seed;
    let params = TaskParamsV1::MlpGrad(p);
    let def = TaskDefinitionV1 {
        catalog_version: CATALOG_VERSION,
        reward_schedule: price(&params)?,
        resource_limits: default_limits(&params)?,
        validation_policy: default_policy(&params)?,
        params,
        inputs,
        numeric_profile: profiles::REQUESTED_F32_V1.into(),
        seed_strategy: SeedStrategyV1::ProtocolDrbg { seed },
        max_attempts: 3,
    };
    validate_definition(&def)?;
    Ok(def)
}

/// Build a complete `graph.grad.v1` definition with catalog defaults.
pub fn graph_grad_definition(
    p: GraphGradParamsV1,
    inputs: BTreeMap<String, ArtifactRef>,
) -> Result<TaskDefinitionV1, ComputeError> {
    let seed = p.round_seed;
    let params = TaskParamsV1::GraphGrad(p);
    let def = TaskDefinitionV1 {
        catalog_version: CATALOG_VERSION,
        reward_schedule: price(&params)?,
        resource_limits: default_limits(&params)?,
        validation_policy: default_policy(&params)?,
        params,
        inputs,
        numeric_profile: profiles::REQUESTED_F32_V1.into(),
        seed_strategy: SeedStrategyV1::ProtocolDrbg { seed },
        max_attempts: 3,
    };
    validate_definition(&def)?;
    Ok(def)
}

/// Build a complete `graph.stage.v1` definition with catalog defaults.
/// `ram_bytes`: the host-memory limit of the task, [`graph_stage_ram_bytes`] of its stage;
/// `output_bytes`: what it writes in all, [`crate::graph::stage_task::output_bytes`]. Both at
/// least the catalog default.
pub fn graph_stage_definition(
    p: GraphStageParamsV1,
    inputs: BTreeMap<String, ArtifactRef>,
    ram_bytes: u64,
    output_bytes: u64,
) -> Result<TaskDefinitionV1, ComputeError> {
    let seed = p.round_seed;
    let params = TaskParamsV1::GraphStage(p);
    let mut resource_limits = default_limits(&params)?;
    resource_limits.max_ram_bytes = resource_limits.max_ram_bytes.max(ram_bytes);
    resource_limits.max_output_bytes = resource_limits.max_output_bytes.max(output_bytes);
    resource_limits.max_disk_bytes = resource_limits
        .max_disk_bytes
        .saturating_add(output_bytes.saturating_mul(2));
    let def = TaskDefinitionV1 {
        catalog_version: CATALOG_VERSION,
        reward_schedule: price(&params)?,
        resource_limits,
        validation_policy: default_policy(&params)?,
        params,
        inputs,
        numeric_profile: profiles::REQUESTED_F32_V1.into(),
        seed_strategy: SeedStrategyV1::ProtocolDrbg { seed },
        max_attempts: 3,
    };
    validate_definition(&def)?;
    Ok(def)
}

/// Build a complete matmul definition with catalog defaults (used by the CLI and tests).
pub fn matmul_definition(
    m: u32,
    k: u32,
    n: u32,
    a: ArtifactRef,
    b: ArtifactRef,
) -> Result<TaskDefinitionV1, ComputeError> {
    let params = TaskParamsV1::Matmul { m, k, n };
    let mut inputs = BTreeMap::new();
    inputs.insert("a".to_owned(), a);
    inputs.insert("b".to_owned(), b);
    let def = TaskDefinitionV1 {
        catalog_version: CATALOG_VERSION,
        reward_schedule: price(&params)?,
        resource_limits: default_limits(&params)?,
        validation_policy: default_policy(&params)?,
        params,
        inputs,
        numeric_profile: profiles::REQUESTED_F32_V1.into(),
        seed_strategy: SeedStrategyV1::None,
        max_attempts: 3,
    };
    validate_definition(&def)?;
    Ok(def)
}

/// Expected GGT1 size of an f32 tensor of `shape`.
pub fn expected_rgt_size(shape: &[u64]) -> u64 {
    ggt_size(shape)
}

/// Limits for decoding an input/output tensor of a known shape: never more than its element count.
pub fn tensor_limits_for(shape: &[u64]) -> tensor_file::TensorLimits {
    tensor_file::TensorLimits {
        max_elements: shape.iter().product(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gg_core::Sha256Digest;

    fn aref(shape: &[u64]) -> ArtifactRef {
        ArtifactRef {
            hash: Sha256Digest::zero(),
            size: ggt_size(shape),
        }
    }

    #[test]
    fn matmul_bounds_and_inputs() {
        let d = matmul_definition(4, 3, 2, aref(&[4, 3]), aref(&[3, 2])).unwrap();
        let v = validate_definition(&d).unwrap();
        assert_eq!(v.output_shapes["c"], vec![4u64, 2]);
        assert!(matmul_definition(0, 3, 2, aref(&[0, 3]), aref(&[3, 2])).is_err());
        assert!(matmul_definition(4096, 3, 2, aref(&[4096, 3]), aref(&[3, 2])).is_err());
        assert!(matmul_definition(4, 3, 2, aref(&[4, 4]), aref(&[3, 2])).is_err());
        let mut bad = d.clone();
        bad.validation_policy.tolerance.abs = 1.0;
        assert!(validate_definition(&bad).is_err());
        let mut bad = d.clone();
        bad.validation_policy.reference_profile = "gpu-anything".into();
        assert!(matches!(
            validate_definition(&bad),
            Err(ComputeError::ProfileNotAdmitted(_))
        ));
        let mut bad = d.clone();
        bad.inputs.insert("z".into(), aref(&[1]));
        assert!(validate_definition(&bad).is_err());
        // mlp params with matmul inputs are refused
        let mut bad = d;
        bad.params = TaskParamsV1::MlpTrain(MlpTrainParamsV1 {
            d_input: 1,
            d_hidden: 1,
            d_output: 1,
            optimizer: "sgd".into(),
            learning_rate: 0.1,
            batch_size: 1,
            steps: 1,
            seed: 0,
        });
        assert!(validate_definition(&bad).is_err());
    }

    #[test]
    fn pricing_is_deterministic_and_positive() {
        let p = price(&TaskParamsV1::Matmul {
            m: 2048,
            k: 2048,
            n: 2048,
        })
        .unwrap();
        assert_eq!(
            p.primary,
            Credits(1 + 2 * 2048 * 2048 * 2048 / 1_000_000_000)
        );
        assert_eq!(p.verification_cost, p.primary);
        let small = price(&TaskParamsV1::Matmul { m: 2, k: 2, n: 2 }).unwrap();
        assert_eq!(small.primary, Credits(1));
        let mlp = price(&TaskParamsV1::MlpTrain(MlpTrainParamsV1 {
            d_input: 256,
            d_hidden: 512,
            d_output: 64,
            optimizer: "adam".into(),
            learning_rate: 0.001,
            batch_size: 512,
            steps: 5000,
            seed: 1,
        }))
        .unwrap();
        assert!(mlp.primary.get() > 1);
    }
}
