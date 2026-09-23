//! Checkpoint split into shards (gg-server/docs/ROADMAP.md, P11 step C2).
//!
//! A single-piece checkpoint capped a model at 1 GiB of parameters, and the server loaded it whole
//! to aggregate it. Once sharded, it travels and is aggregated one shard at a time: the server never
//! holds the whole model again.
//!
//! The split is not transmitted: it is a **deterministic** function of the description and a target
//! size, which the server, the worker and the validator each compute. Parameters are taken in the
//! order of their first use during execution, which brings a shard close to a stage (ADR-0018), and
//! grouped while the shard stays under the target. A parameter larger than the target forms its own
//! shard; one larger than [`MAX_SHARD_BYTES`] is rejected with the remedy named: tiling the matrix
//! in the description (ADR-0019).
//!
//! Names: without sharding (`shard_bytes = 0`), a single shard named `checkpoint` and `gradient`,
//! exactly as before; with sharding, `checkpoint.0 … checkpoint.k`, and `gradient.i` for each shard
//! that contains at least one trained parameter.

use gg_protocol::tensor_file::{BundleF32, Dtype};

use super::{GraphReport, ModelGraphV1, ParamSpec};

/// Maximum number of shards: a task carries at most 64 input files (graph, shards, datasets).
pub const MAX_SHARDS: usize = 40;
/// Maximum size of a shard, and therefore of a tensor.
pub const MAX_SHARD_BYTES: u64 = 1 << 30;
/// Default target size: a larger checkpoint is sharded.
pub const DEFAULT_SHARD_BYTES: u64 = 256 << 20;

/// A shard: parameters, and the exact size of their containers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shard {
    /// Parameter names, in order of first use.
    pub params: Vec<String>,
    /// Bytes of the `GGB1` container holding all these parameters.
    pub checkpoint_bytes: u64,
    /// Bytes of the f32 container holding only the trained parameters; 0 if there are none. The
    /// layout of the server's optimizer state, and of a gradient sent in f32.
    pub gradient_bytes: u64,
    /// The same container in bf16: a gradient sent in bf16 (ADR-0035).
    pub gradient_bf16_bytes: u64,
}

impl Shard {
    /// Bytes of this shard's gradient sent as `dtype`.
    pub fn gradient_size(&self, dtype: Dtype) -> u64 {
        match dtype {
            Dtype::F32 => self.gradient_bytes,
            Dtype::Bf16 => self.gradient_bf16_bytes,
        }
    }
}

/// How a checkpoint is split into shards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardPlan {
    /// Sharded (`checkpoint.i`) or single-piece (`checkpoint`).
    pub sharded: bool,
    /// The shards.
    pub shards: Vec<Shard>,
}

impl ShardPlan {
    /// Input name of shard `i`.
    pub fn checkpoint_name(&self, i: usize) -> String {
        if self.sharded {
            format!("checkpoint.{i}")
        } else {
            "checkpoint".to_owned()
        }
    }
    /// Gradient output name of shard `i`.
    pub fn gradient_name(&self, i: usize) -> String {
        if self.sharded {
            format!("gradient.{i}")
        } else {
            "gradient".to_owned()
        }
    }
    /// Shards that carry a gradient, with their index.
    pub fn trained(&self) -> impl Iterator<Item = (usize, &Shard)> {
        self.shards
            .iter()
            .enumerate()
            .filter(|(_, s)| s.gradient_bytes > 0)
    }
    /// Bytes of the whole checkpoint.
    pub fn checkpoint_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.checkpoint_bytes).sum()
    }
    /// Bytes of the whole gradient in f32.
    pub fn gradient_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.gradient_bytes).sum()
    }
    /// Bytes of the whole gradient sent as `dtype`.
    pub fn gradient_size(&self, dtype: Dtype) -> u64 {
        self.shards.iter().map(|s| s.gradient_size(dtype)).sum()
    }
    /// Shard holding a parameter.
    pub fn shard_of(&self, name: &str) -> Option<usize> {
        self.shards
            .iter()
            .position(|s| s.params.iter().any(|p| p == name))
    }
}

fn bundle_bytes<'a>(params: impl Iterator<Item = &'a ParamSpec>) -> u64 {
    bundle_bytes_as(params, Dtype::F32)
}

fn bundle_bytes_as<'a>(params: impl Iterator<Item = &'a ParamSpec>, dtype: Dtype) -> u64 {
    BundleF32::encoded_size_as(params.map(|p| (p.name.as_str(), p.shape.as_slice())), dtype)
}

/// Sizes of the gradient container of a group, f32 and bf16; 0 if nothing in it is trained.
fn gradient_sizes(group: &[&ParamSpec]) -> (u64, u64) {
    if !group.iter().any(|p| p.trainable) {
        return (0, 0);
    }
    let trained = || group.iter().copied().filter(|p| p.trainable);
    (
        bundle_bytes_as(trained(), Dtype::F32),
        bundle_bytes_as(trained(), Dtype::Bf16),
    )
}

/// Parameters in order of first use; those no node uses come last, in declaration order.
fn first_use_order<'a>(graph: &'a ModelGraphV1, report: &GraphReport) -> Vec<&'a ParamSpec> {
    let mut out: Vec<&ParamSpec> = Vec::with_capacity(graph.parameters.len());
    for cost in &report.nodes {
        for name in &cost.parameter_names {
            if let Some(p) = graph.parameters.iter().find(|p| &p.name == name)
                && !out.iter().any(|q| q.name == p.name)
            {
                out.push(p);
            }
        }
    }
    for p in &graph.parameters {
        if !out.iter().any(|q| q.name == p.name) {
            out.push(p);
        }
    }
    out
}

/// Splits the checkpoint of `graph` into shards of at most `shard_bytes` bytes (0: a single one).
pub fn plan(
    graph: &ModelGraphV1,
    report: &GraphReport,
    shard_bytes: u64,
) -> Result<ShardPlan, String> {
    let make = |group: &[&ParamSpec]| {
        let (gradient_bytes, gradient_bf16_bytes) = gradient_sizes(group);
        Shard {
            params: group.iter().map(|p| p.name.clone()).collect(),
            checkpoint_bytes: bundle_bytes(group.iter().copied()),
            gradient_bytes,
            gradient_bf16_bytes,
        }
    };
    let order = first_use_order(graph, report);
    if shard_bytes == 0 {
        return Ok(ShardPlan {
            sharded: false,
            shards: vec![make(&order)],
        });
    }
    let mut shards = Vec::new();
    let mut group: Vec<&ParamSpec> = Vec::new();
    for p in order {
        let alone = bundle_bytes(std::iter::once(p));
        if alone > MAX_SHARD_BYTES {
            return Err(format!(
                "parameter '{}' is {alone} bytes, more than a shard ({MAX_SHARD_BYTES}): tile the \
matrix in the description (ADR-0019)",
                p.name
            ));
        }
        group.push(p);
        if group.len() > 1 && bundle_bytes(group.iter().copied()) > shard_bytes {
            group.pop();
            shards.push(make(&group));
            group = vec![p];
        }
    }
    if !group.is_empty() {
        shards.push(make(&group));
    }
    if shards.len() > MAX_SHARDS {
        return Err(format!(
            "{} shards of {shard_bytes} bytes, the maximum is {MAX_SHARDS}: increase the shard \
size, or split the model into stages",
            shards.len()
        ));
    }
    Ok(ShardPlan {
        sharded: true,
        shards,
    })
}

/// Shard size used when the researcher does not specify one: single-piece up to
/// [`DEFAULT_SHARD_BYTES`], sharded beyond, with a raised target if the number of shards would
/// exceed [`MAX_SHARDS`].
pub fn default_shard_bytes(checkpoint_bytes: u64) -> u64 {
    if checkpoint_bytes <= DEFAULT_SHARD_BYTES {
        0
    } else {
        DEFAULT_SHARD_BYTES
            .max(checkpoint_bytes.div_ceil(MAX_SHARDS as u64 / 2))
            .min(MAX_SHARD_BYTES)
    }
}

/// Shards of a staged training run (D2): shard `j` carries the parameters whose first use is in
/// the same stage, in stage order; a stage with no new parameter has no shard. A stage reads the
/// shards of every parameter it uses ([`stage_shards`]).
pub fn plan_stages(
    graph: &ModelGraphV1,
    stages: &[super::stages::StageSpec],
) -> Result<ShardPlan, String> {
    let mut seen: Vec<&str> = Vec::new();
    let mut shards = Vec::new();
    for (k, st) in stages.iter().enumerate() {
        let mut group: Vec<&ParamSpec> = Vec::new();
        let mut names: Vec<&String> = st.params.iter().collect();
        // A parameter no node uses goes with the first stage.
        if k == 0 {
            for p in &graph.parameters {
                if !stages.iter().any(|s| s.params.contains(&p.name)) {
                    names.push(&p.name);
                }
            }
        }
        for n in names {
            if seen.contains(&n.as_str()) {
                continue;
            }
            seen.push(n);
            let p = graph
                .parameters
                .iter()
                .find(|p| &p.name == n)
                .ok_or_else(|| format!("unknown parameter '{n}'"))?;
            group.push(p);
        }
        if group.is_empty() {
            continue;
        }
        let checkpoint_bytes = bundle_bytes(group.iter().copied());
        if checkpoint_bytes > MAX_SHARD_BYTES {
            return Err(format!(
                "stage {k} brings {checkpoint_bytes} bytes of parameters, more than a shard \
({MAX_SHARD_BYTES}): use smaller stages, or tile its matrices"
            ));
        }
        let (gradient_bytes, gradient_bf16_bytes) = gradient_sizes(&group);
        shards.push(Shard {
            params: group.iter().map(|p| p.name.clone()).collect(),
            checkpoint_bytes,
            gradient_bytes,
            gradient_bf16_bytes,
        });
    }
    if shards.len() > MAX_SHARDS {
        return Err(format!(
            "{} shards, the maximum is {MAX_SHARDS}",
            shards.len()
        ));
    }
    Ok(ShardPlan {
        sharded: true,
        shards,
    })
}

/// Shards a stage reads: those carrying one of its parameters, in order.
pub fn stage_shards(plan: &ShardPlan, stage: &super::stages::StageSpec) -> Vec<usize> {
    plan.shards
        .iter()
        .enumerate()
        .filter(|(_, s)| s.params.iter().any(|p| stage.params.contains(p)))
        .map(|(j, _)| j)
        .collect()
}
