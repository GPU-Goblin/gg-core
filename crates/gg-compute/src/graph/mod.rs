//! Models described by a computation graph (ADR-0016).
//!
//! The researcher describes the model; no code is sent. The description is validated **before**
//! any reservation: operators present in the published profile, shapes that agree, an acyclic
//! graph, bounds respected, memory estimated. When something is missing, the message names the
//! offending operator or shape, because a useful refusal is better than a failure halfway through.

pub mod exec;
pub mod formula;
pub mod functions;
pub mod memory;
pub mod ops;
pub mod plan;
pub mod rope;
pub mod shards;
pub mod stage_task;
pub mod stages;
pub mod task;
pub mod templates;
pub mod tiling;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub use ops::{
    Attr, CAUSAL_FILL, Determinism, Dtype, OpSpec, PROFILE, PROFILE_VERSION, Shape, show,
};

/// Format expected at the root of the description.
pub const FORMAT: &str = "gg-graph/v1";

/// Bounds of the graph itself. They replace the architecture bounds of the original catalog.
pub mod bounds {
    /// Maximum number of nodes.
    pub const MAX_NODES: usize = 4096;
    /// Maximum number of named tensors.
    pub const MAX_TENSORS: usize = 8192;
    /// Maximum rank of a tensor. This is also what the executor supports: nothing in the profile
    /// requires more (multi-head attention is rank 4).
    pub const MAX_RANK: usize = 4;
    /// Maximum number of trainable parameters. The description may describe a model that no
    /// single machine can hold: the distribution plan (`plan`) decides feasibility.
    pub const MAX_PARAMETERS: u64 = 1_000_000_000_000;
    /// Maximum length of a name.
    pub const MAX_NAME: usize = 64;
}

/// A graph input tensor. A zero dimension marks the batch, unknown at description time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    /// Name.
    pub name: String,
    /// Shape; `0` means "batch dimension".
    pub shape: Vec<u64>,
    /// Type; only `f32` is executed today, `i64` is used for indices.
    #[serde(default = "default_dtype")]
    pub dtype: String,
}
fn default_dtype() -> String {
    "f32".into()
}

/// Initialization of a parameter. Sampling goes through the protocol DRBG, so it is reproducible.
///
/// An unknown field is refused, here as everywhere in the description: a worker that does not
/// understand a field must not silently ignore it and compute a different function (ADR-0022).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Init {
    /// Zeros.
    Zeros,
    /// Ones.
    Ones,
    /// Symmetric uniform with spread `1/sqrt(fan_in)`.
    XavierUniform,
    /// Normal with the given standard deviation.
    Normal {
        /// Standard deviation.
        std: f64,
    },
    /// All values equal. Used for a model's constants (the 1/√d factor of an attention, for
    /// example) declared as frozen parameters (`"trainable": false`): the description stays
    /// self-contained, without depending on a checkpoint to compute the right function.
    Constant {
        /// Value, finite.
        value: f64,
    },
    /// Original sinusoidal table (Vaswani et al.) over the last two axes `[T, D]`:
    /// `PE[t, 2i] = sin(t·ω_i)`, `PE[t, 2i+1] = cos(t·ω_i)`, with `ω_i = base^(−2i/D)`.
    /// The other axes are 1. ADR-0021.
    Sinusoidal {
        /// Frequency base, 10,000 in the original paper.
        base: f64,
    },
    /// Cosine of a RoPE-style rotation, in the "rotate_half" layout of the Llama models in
    /// Hugging Face: `C[t, j] = A·cos(t·ω_(j mod d/2))`. The frequencies follow the original law
    /// or are given explicitly (see [`Rotary`]). ADR-0021, ADR-0022.
    RotaryCos(Rotary),
    /// Matching sine: `S[t, j] = A·sin(t·ω_(j mod d/2))`.
    RotarySin(Rotary),
}

/// Frequencies of a rotary table. Two mutually exclusive forms:
///
/// - `{"base": b}`: the original law, `ω_i = 1 / b^(2i/d)`, amplitude 1 (10,000 for Llama 2,
///   500,000 for Llama 3, 1,000,000 for Qwen 2);
/// - `{"frequencies": [ω_0, …, ω_(d/2−1)], "amplitude": A}`: the frequencies themselves, in
///   radians per position. This is the form of **every scaling** (linear, dynamic NTK, YaRN,
///   LongRoPE, Llama 3.1), each of which is only a way of computing these frequencies, plus, for
///   YaRN and LongRoPE, an amplitude `A` that multiplies cosine and sine (1 by default).
///
/// The executor therefore has a single formula, `A·cos(t·ω)`, whatever the variant (ADR-0022).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rotary {
    /// Base of the original law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<f64>,
    /// Explicit frequencies, one per pair of dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequencies: Option<Vec<f64>>,
    /// Amplitude, with explicit frequencies only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amplitude: Option<f64>,
}

impl Rotary {
    /// Original law with base `base`.
    pub fn from_base(base: f64) -> Self {
        Self {
            base: Some(base),
            ..Self::default()
        }
    }

    /// Explicit frequencies and amplitude.
    pub fn explicit(frequencies: Vec<f64>, amplitude: f64) -> Self {
        Self {
            frequencies: Some(frequencies),
            amplitude: Some(amplitude),
            ..Self::default()
        }
    }

    /// Checks the form for a table of width `d`; the message names what is missing.
    pub fn check(&self, d: u64) -> Result<(), String> {
        match (self.base, &self.frequencies) {
            (Some(_), Some(_)) => Err("a base or explicit frequencies, not both".into()),
            (None, None) => Err("a base or explicit frequencies are expected".into()),
            (Some(base), None) => {
                if self.amplitude.is_some() {
                    return Err("an amplitude only goes with explicit frequencies".into());
                }
                if !(base.is_finite() && base > 1.0) {
                    return Err(format!(
                        "invalid frequency base {base}, expected finite and > 1"
                    ));
                }
                Ok(())
            }
            (None, Some(freqs)) => {
                if freqs.len() as u64 != d / 2 {
                    return Err(format!(
                        "{} frequencies for a table of width {d}, {} expected (one per pair \
of dimensions)",
                        freqs.len(),
                        d / 2
                    ));
                }
                if let Some((i, w)) = freqs
                    .iter()
                    .enumerate()
                    .find(|(_, w)| !(w.is_finite() && **w >= 0.0))
                {
                    return Err(format!(
                        "invalid frequency {i} ({w}), expected finite and non-negative"
                    ));
                }
                match self.amplitude {
                    Some(a) if !(a.is_finite() && a > 0.0) => Err(format!(
                        "invalid amplitude {a}, expected finite and strictly positive"
                    )),
                    _ => Ok(()),
                }
            }
        }
    }

    /// Frequencies in f32 and amplitude, as the executor uses them, for a width `d` already
    /// checked by [`Rotary::check`].
    pub fn frequencies_f32(&self, d: usize) -> (Vec<f32>, f32) {
        match (&self.frequencies, self.base) {
            (Some(freqs), _) => (
                freqs.iter().map(|w| *w as f32).collect(),
                self.amplitude.unwrap_or(1.0) as f32,
            ),
            (None, Some(base)) => (exec::position_frequencies(base, d), 1.0),
            (None, None) => unreachable!("rotary table already checked"),
        }
    }
}

impl Init {
    /// True for a generated position table (sinusoidal or rotary).
    pub fn is_position_table(&self) -> bool {
        matches!(
            self,
            Init::Sinusoidal { .. } | Init::RotaryCos(_) | Init::RotarySin(_)
        )
    }
}

/// A model parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSpec {
    /// Name.
    pub name: String,
    /// Shape, fully known.
    pub shape: Vec<u64>,
    /// Initialization.
    pub init: Init,
    /// Trained, or frozen. A frozen parameter keeps its value, receives neither a gradient nor
    /// optimizer state, and is not exchanged at synchronization: this is what makes low-rank
    /// adapters over a frozen base practical (ADR-0019).
    #[serde(default = "trainable_default")]
    pub trainable: bool,
}

fn trainable_default() -> bool {
    true
}

/// A graph node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    /// Operator from the profile.
    pub op: String,
    /// Consumed tensors.
    pub inputs: Vec<String>,
    /// Produced tensor (only one today).
    pub output: String,
    /// Attributes.
    #[serde(default)]
    pub attrs: BTreeMap<String, Attr>,
}

/// Loss function.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Loss {
    /// Mean squared error.
    MeanSquaredError {
        /// Compared output.
        prediction: String,
        /// Target.
        target: TensorSpec,
    },
    /// Cross-entropy over logits, target as class indices.
    CrossEntropy {
        /// Logits.
        prediction: String,
        /// Target.
        target: TensorSpec,
    },
}

impl Loss {
    /// Name of the compared tensor.
    pub fn prediction(&self) -> &str {
        match self {
            Loss::MeanSquaredError { prediction, .. } | Loss::CrossEntropy { prediction, .. } => {
                prediction
            }
        }
    }
    /// Expected target.
    pub fn target(&self) -> &TensorSpec {
        match self {
            Loss::MeanSquaredError { target, .. } | Loss::CrossEntropy { target, .. } => target,
        }
    }
}

/// The described model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGraphV1 {
    /// Must be `gg-graph/v1`.
    pub format: String,
    /// Name given by the researcher.
    pub name: String,
    /// Inputs.
    pub inputs: Vec<TensorSpec>,
    /// Trainable parameters.
    pub parameters: Vec<ParamSpec>,
    /// Nodes, in any order: the execution order is inferred.
    pub nodes: Vec<Node>,
    /// Loss.
    pub loss: Loss,
    /// Functions called through `fn:<name>` (ADR-0025). Absent when there are none: earlier
    /// descriptions keep their bytes and their hash.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub functions: Vec<functions::FunctionSpec>,
}

/// Why a description is refused. Each case names what is wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    /// Unknown format.
    Format(String),
    /// Name empty, too long or duplicated.
    Name(String),
    /// Bound exceeded.
    Bound(String),
    /// Operator absent from the published profile.
    UnsupportedOp {
        /// Requested operator.
        op: String,
        /// Node concerned (its output).
        node: String,
        /// Close operators from the profile, if any.
        suggestions: Vec<&'static str>,
    },
    /// Unknown or missing attribute.
    Attribute(String),
    /// Tensor consumed without being produced.
    Unknown(String),
    /// Cycle, or a node that cannot be ordered.
    Cycle(Vec<String>),
    /// Incompatible shapes.
    Shape(String),
    /// Non-executable type.
    Dtype(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Format(m)
            | GraphError::Name(m)
            | GraphError::Bound(m)
            | GraphError::Attribute(m)
            | GraphError::Shape(m)
            | GraphError::Dtype(m) => f.write_str(m),
            GraphError::Unknown(name) => {
                write!(f, "tensor '{name}' is used without being produced")
            }
            GraphError::Cycle(names) => write!(
                f,
                "the graph contains a cycle or unreachable nodes: {}",
                names.join(", ")
            ),
            GraphError::UnsupportedOp {
                op,
                node,
                suggestions,
            } => {
                write!(
                    f,
                    "operator '{op}' (node '{node}') is not part of profile {PROFILE_VERSION}"
                )?;
                if !suggestions.is_empty() {
                    write!(f, "; the profile contains {}", suggestions.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// Cost of a node, for one example. This is the raw material of the distribution plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCost {
    /// Produced tensor.
    pub output: String,
    /// Operator.
    pub op: String,
    /// Consumed tensors (graph inputs, parameters or outputs of other nodes).
    pub inputs: Vec<String>,
    /// FLOPs of one training step for one example (forward + backward).
    pub training_flops_per_example: u64,
    /// Bytes of the output for one example, kept for the backward pass.
    pub activation_bytes_per_example: u64,
    /// Parameters this node is the first to consume.
    pub parameter_names: Vec<String>,
    /// Their bytes.
    pub parameter_bytes: u64,
}

/// What is known about an accepted graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphReport {
    /// Model name.
    pub name: String,
    /// Operator profile used for validation.
    pub profile: String,
    /// Inferred execution order (names of the outputs).
    pub order: Vec<String>,
    /// Operators used, once each.
    pub ops_used: Vec<String>,
    /// Number of parameters, frozen ones included.
    pub parameters: u64,
    /// Bytes of the parameters in f32.
    pub parameter_bytes: u64,
    /// Number of trained parameters (those that receive a gradient).
    pub trainable_parameters: u64,
    /// Their bytes: this is what is exchanged at every synchronization.
    pub trainable_parameter_bytes: u64,
    /// FLOPs of the forward pass, for one example.
    pub forward_flops_per_example: u64,
    /// Estimated FLOPs of one training step (forward + backward ≈ 3×).
    pub training_flops_per_example: u64,
    /// Activation bytes for one example (forward pass, kept for the backward pass).
    pub activation_bytes_per_example: u64,
    /// Inferred shapes, per tensor.
    pub shapes: BTreeMap<String, String>,
    /// Operators whose reproducibility is not bitwise.
    pub tolerance_ops: Vec<String>,
    /// Cost of each node, in execution order.
    pub nodes: Vec<NodeCost>,
    /// Bytes per example of each graph input.
    pub input_bytes_per_example: BTreeMap<String, u64>,
    /// Integer tensors computed by the graph (ADR-0029), with the range of their values. Absent
    /// when there are none: earlier reports keep their bytes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub integer_tensors: BTreeMap<String, IntRange>,
}

/// What validation knows of an integer tensor (ADR-0029).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntRange {
    /// The values are in `[0, bound)`; `None` when they come from the data.
    pub bound: Option<u64>,
    /// Axis along which the values of a line are distinct (the output of `ArgTopK`/`ArgSort`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct_axis: Option<usize>,
}

impl GraphReport {
    /// Estimated memory for a batch, excluding optimizer state: parameters, gradients and
    /// activations.
    pub fn memory_bytes(&self, batch: u64) -> u64 {
        self.parameter_bytes
            .saturating_mul(2)
            .saturating_add(self.activation_bytes_per_example.saturating_mul(batch))
    }
}

fn check_name(kind: &str, name: &str) -> Result<(), GraphError> {
    if name.is_empty() || name.len() > bounds::MAX_NAME {
        return Err(GraphError::Name(format!(
            "{kind} '{name}': the name must be 1 to {} characters long",
            bounds::MAX_NAME
        )));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
    {
        return Err(GraphError::Name(format!(
            "{kind} '{name}': allowed characters are a-z A-Z 0-9 _ - . /"
        )));
    }
    Ok(())
}

/// Validates a description against the published profile and computes what it will cost.
pub fn validate(graph: &ModelGraphV1) -> Result<GraphReport, GraphError> {
    validate_full(graph).map(|(r, _)| r)
}

/// [`validate`], and the upper bound (exclusive) that each i64 input of the micro-batch must
/// respect, from every axis it indexes (ADR-0029): the executor checks it when it loads the data.
pub fn validate_full(
    graph: &ModelGraphV1,
) -> Result<(GraphReport, BTreeMap<String, u64>), GraphError> {
    if graph.format != FORMAT {
        return Err(GraphError::Format(format!(
            "unknown format '{}', expected '{FORMAT}'",
            graph.format
        )));
    }
    check_name("model", &graph.name)?;
    if !graph.functions.is_empty() {
        return Err(GraphError::Format(
            "the description declares functions: it is validated once inlined \
(graph::functions::inline, which the server does at submission)"
                .into(),
        ));
    }
    if graph.nodes.is_empty() {
        return Err(GraphError::Bound("the graph has no nodes".into()));
    }
    if graph.nodes.len() > bounds::MAX_NODES {
        return Err(GraphError::Bound(format!(
            "{} nodes, the maximum is {}",
            graph.nodes.len(),
            bounds::MAX_NODES
        )));
    }

    // Tensors known at the start: inputs, parameters and the loss target.
    let mut shapes: BTreeMap<String, Shape> = BTreeMap::new();
    let mut dtypes: BTreeMap<String, Dtype> = BTreeMap::new();
    // Integer tensors: known range, distinct axis, and the i64 inputs they come from.
    let mut ints: BTreeMap<String, (IntRange, BTreeSet<String>)> = BTreeMap::new();
    let mut index_limits: BTreeMap<String, u64> = BTreeMap::new();
    let mut parameters: u64 = 0;
    let mut param_bytes_by_name: BTreeMap<String, u64> = BTreeMap::new();
    let mut trainable_parameters: u64 = 0;
    let mut input_bytes_per_example: BTreeMap<String, u64> = BTreeMap::new();
    for input in &graph.inputs {
        check_name("input", &input.name)?;
        if input.shape.len() > bounds::MAX_RANK {
            return Err(GraphError::Bound(format!(
                "input '{}' has rank {}, the maximum is {}",
                input.name,
                input.shape.len(),
                bounds::MAX_RANK
            )));
        }
        if !matches!(input.dtype.as_str(), "f32" | "i64") {
            return Err(GraphError::Dtype(format!(
                "input '{}' has type '{}'; only f32 and i64 are executed",
                input.name, input.dtype
            )));
        }
        let shape: Shape = input
            .shape
            .iter()
            .map(|d| if *d == 0 { None } else { Some(*d) })
            .collect();
        let elements: u64 = shape.iter().map(|d| d.unwrap_or(1)).product();
        let width = if input.dtype == "i64" { 8 } else { 4 };
        input_bytes_per_example.insert(input.name.clone(), elements.saturating_mul(width));
        if shapes.insert(input.name.clone(), shape).is_some() {
            return Err(GraphError::Name(format!("'{}' declared twice", input.name)));
        }
        let dt = if input.dtype == "i64" {
            ints.insert(
                input.name.clone(),
                (
                    IntRange {
                        bound: None,
                        distinct_axis: None,
                    },
                    BTreeSet::from([input.name.clone()]),
                ),
            );
            Dtype::I64
        } else {
            Dtype::F32
        };
        dtypes.insert(input.name.clone(), dt);
    }
    for p in &graph.parameters {
        check_name("parameter", &p.name)?;
        match p.init {
            Init::Normal { std } if !(std.is_finite() && std >= 0.0) => {
                return Err(GraphError::Shape(format!(
                    "parameter '{}': invalid standard deviation {std}, expected finite and \
non-negative",
                    p.name
                )));
            }
            Init::Constant { value } if !value.is_finite() => {
                return Err(GraphError::Shape(format!(
                    "parameter '{}': constant {value} is not finite",
                    p.name
                )));
            }
            _ => {}
        }
        if p.init.is_position_table() {
            let r = p.shape.len();
            if r < 2
                || p.shape[..r - 2].iter().any(|d| *d != 1)
                || !p.shape[r - 1].is_multiple_of(2)
            {
                return Err(GraphError::Shape(format!(
                    "parameter '{}': a position table has shape [1, …, 1, T, d] with d even, \
got {:?}",
                    p.name, p.shape
                )));
            }
            let checked = match &p.init {
                Init::Sinusoidal { base } if !(base.is_finite() && *base > 1.0) => Err(format!(
                    "invalid frequency base {base}, expected finite and > 1"
                )),
                Init::RotaryCos(rotary) | Init::RotarySin(rotary) => rotary.check(p.shape[r - 1]),
                _ => Ok(()),
            };
            if let Err(why) = checked {
                return Err(GraphError::Shape(format!("parameter '{}': {why}", p.name)));
            }
        }
        if p.shape.is_empty() || p.shape.contains(&0) {
            return Err(GraphError::Shape(format!(
                "parameter '{}': every dimension must be known and non-zero",
                p.name
            )));
        }
        if p.shape.len() > bounds::MAX_RANK {
            return Err(GraphError::Bound(format!(
                "parameter '{}' has rank {}, the maximum is {}",
                p.name,
                p.shape.len(),
                bounds::MAX_RANK
            )));
        }
        let count: u64 = p.shape.iter().product();
        parameters = parameters.saturating_add(count);
        if p.trainable {
            trainable_parameters = trainable_parameters.saturating_add(count);
        }
        param_bytes_by_name.insert(p.name.clone(), count.saturating_mul(4));
        if shapes
            .insert(p.name.clone(), p.shape.iter().map(|d| Some(*d)).collect())
            .is_some()
        {
            return Err(GraphError::Name(format!("'{}' declared twice", p.name)));
        }
        dtypes.insert(p.name.clone(), Dtype::F32);
    }
    if parameters > bounds::MAX_PARAMETERS {
        return Err(GraphError::Bound(format!(
            "{parameters} parameters, the maximum is {}",
            bounds::MAX_PARAMETERS
        )));
    }

    // Scheduling: a node runs once all of its inputs exist.
    let mut pending: Vec<&Node> = graph.nodes.iter().collect();
    let mut order = Vec::with_capacity(graph.nodes.len());
    let mut ops_used: BTreeSet<String> = BTreeSet::new();
    let mut tolerance_ops: BTreeSet<String> = BTreeSet::new();
    let mut forward_flops: u64 = 0;
    let mut activation_elements: u64 = 0;
    let mut produced: BTreeSet<String> = BTreeSet::new();
    let mut node_costs: Vec<NodeCost> = Vec::with_capacity(graph.nodes.len());
    let mut assigned_params: BTreeSet<String> = BTreeSet::new();

    while !pending.is_empty() {
        let ready = pending
            .iter()
            .position(|n| n.inputs.iter().all(|i| shapes.contains_key(i)));
        let Some(index) = ready else {
            // Either a cycle or an input that exists nowhere: tell the two apart.
            for node in &pending {
                for input in &node.inputs {
                    if !shapes.contains_key(input) && !pending.iter().any(|n| n.output == *input) {
                        return Err(GraphError::Unknown(input.clone()));
                    }
                }
            }
            return Err(GraphError::Cycle(
                pending.iter().map(|n| n.output.clone()).collect(),
            ));
        };
        let node = pending.remove(index);
        check_name("node", &node.output)?;
        let spec = ops::lookup(&node.op).ok_or_else(|| GraphError::UnsupportedOp {
            op: node.op.clone(),
            node: node.output.clone(),
            suggestions: ops::suggestions(&node.op),
        })?;
        if node.inputs.len() < spec.arity.0 || node.inputs.len() > spec.arity.1 {
            return Err(GraphError::Shape(format!(
                "node '{}': {} expects between {} and {} inputs, {} given",
                node.output,
                node.op,
                spec.arity.0,
                spec.arity.1,
                node.inputs.len()
            )));
        }
        for key in node.attrs.keys() {
            if !spec.attributes.contains(&key.as_str()) {
                return Err(GraphError::Attribute(format!(
                    "node '{}': attribute '{key}' does not exist for {} (known: {})",
                    node.output,
                    node.op,
                    if spec.attributes.is_empty() {
                        "none".to_string()
                    } else {
                        spec.attributes.join(", ")
                    }
                )));
            }
        }
        for key in spec.required {
            if !node.attrs.contains_key(*key) {
                return Err(GraphError::Attribute(format!(
                    "node '{}': attribute '{key}' is required by {}",
                    node.output, node.op
                )));
            }
        }
        let input_shapes: Vec<Shape> = node
            .inputs
            .iter()
            .map(|i| shapes[i].clone())
            .collect::<Vec<_>>();
        // Types first: an integer fed to arithmetic is named as such, not as a shape mismatch.
        let input_dtypes: Vec<Dtype> = node.inputs.iter().map(|i| dtypes[i]).collect();
        let out_dtype = ops::infer_dtype(&node.op, &input_dtypes).map_err(|(k, e)| {
            GraphError::Dtype(format!(
                "node '{}' ({}): {e} ('{}')",
                node.output, node.op, node.inputs[k]
            ))
        })?;
        let out = ops::infer(&node.op, &input_shapes, &node.attrs)
            .map_err(|e| GraphError::Shape(format!("node '{}' ({}): {e}", node.output, node.op)))?;
        integer_flow(node, &input_shapes, &mut ints, &mut index_limits)
            .map_err(|e| GraphError::Shape(format!("node '{}' ({}): {e}", node.output, node.op)))?;
        dtypes.insert(node.output.clone(), out_dtype);
        if out.len() > bounds::MAX_RANK {
            return Err(GraphError::Bound(format!(
                "node '{}' produces a tensor of rank {}, the maximum is {}",
                node.output,
                out.len(),
                bounds::MAX_RANK
            )));
        }
        let node_flops = ops::flops(&node.op, &input_shapes, &out, &node.attrs);
        forward_flops = forward_flops.saturating_add(node_flops);
        // An integer counts 8 bytes (two f32 units; i32 on the GPU, an upper bound).
        let out_elements: u64 =
            ops::activation_elements(&node.op, &input_shapes, &out, &node.attrs)
                .saturating_mul(if out_dtype == Dtype::I64 { 2 } else { 1 });
        activation_elements = activation_elements.saturating_add(out_elements);
        let mut parameter_names = Vec::new();
        let mut node_param_bytes = 0u64;
        for input in &node.inputs {
            if let Some(bytes) = param_bytes_by_name.get(input)
                && assigned_params.insert(input.clone())
            {
                parameter_names.push(input.clone());
                node_param_bytes = node_param_bytes.saturating_add(*bytes);
            }
        }
        node_costs.push(NodeCost {
            output: node.output.clone(),
            op: node.op.clone(),
            inputs: node.inputs.clone(),
            training_flops_per_example: node_flops.saturating_mul(3),
            activation_bytes_per_example: out_elements.saturating_mul(4),
            parameter_names,
            parameter_bytes: node_param_bytes,
        });
        ops_used.insert(node.op.clone());
        if spec.determinism != Determinism::Bitwise {
            tolerance_ops.insert(node.op.clone());
        }
        if shapes.insert(node.output.clone(), out).is_some() {
            return Err(GraphError::Name(format!(
                "'{}' is produced twice",
                node.output
            )));
        }
        produced.insert(node.output.clone());
        order.push(node.output.clone());
        if shapes.len() > bounds::MAX_TENSORS {
            return Err(GraphError::Bound(format!(
                "more than {} named tensors",
                bounds::MAX_TENSORS
            )));
        }
    }

    // The loss must apply to a tensor that exists.
    let prediction = graph.loss.prediction();
    let pred_shape = shapes
        .get(prediction)
        .ok_or_else(|| GraphError::Unknown(prediction.to_owned()))?
        .clone();
    let target = graph.loss.target();
    check_name("target", &target.name)?;
    if dtypes.get(prediction) != Some(&Dtype::F32) {
        return Err(GraphError::Dtype(format!(
            "loss: the prediction '{prediction}' is an integer tensor; the loss compares f32"
        )));
    }
    match &graph.loss {
        Loss::MeanSquaredError { .. } => {
            let t: Shape = target
                .shape
                .iter()
                .map(|d| if *d == 0 { None } else { Some(*d) })
                .collect();
            if t.len() != pred_shape.len() {
                return Err(GraphError::Shape(format!(
                    "loss: target {} and prediction {} do not have the same rank",
                    show(&t),
                    show(&pred_shape)
                )));
            }
            for (a, b) in t.iter().zip(pred_shape.iter()) {
                if let (Some(x), Some(y)) = (a, b)
                    && x != y
                {
                    return Err(GraphError::Shape(format!(
                        "loss: target {} does not match prediction {}",
                        show(&t),
                        show(&pred_shape)
                    )));
                }
            }
        }
        Loss::CrossEntropy { .. } => {
            if pred_shape.len() != 2 {
                return Err(GraphError::Shape(format!(
                    "loss: cross-entropy expects logits [N, classes], got {}",
                    show(&pred_shape)
                )));
            }
            if target.dtype != "i64" {
                return Err(GraphError::Dtype(
                    "loss: the cross-entropy target is class indices (i64)".into(),
                ));
            }
            // Each class must be a column of the logits (checked when the data is loaded).
            if let Some(classes) = pred_shape[1] {
                let l = index_limits.entry(target.name.clone()).or_insert(classes);
                *l = (*l).min(classes);
            }
            // One class per example ([N]) or per token ([N, T]); tokens are flattened at
            // execution time, just as the logits [N·T, classes] were.
            if target.shape.is_empty() || target.shape.len() > 2 {
                return Err(GraphError::Shape(format!(
                    "loss: the cross-entropy target has rank 1 or 2, got {}",
                    target.shape.len()
                )));
            }
        }
    }

    if trainable_parameters == 0 {
        return Err(GraphError::Bound(
            "no parameter is trained: nothing to learn".into(),
        ));
    }
    let parameter_bytes = parameters.saturating_mul(4);
    let activation_bytes_per_example = activation_elements.saturating_mul(4);
    Ok((
        GraphReport {
            name: graph.name.clone(),
            profile: PROFILE_VERSION.to_owned(),
            order,
            ops_used: ops_used.into_iter().collect(),
            parameters,
            parameter_bytes,
            trainable_parameters,
            trainable_parameter_bytes: trainable_parameters.saturating_mul(4),
            forward_flops_per_example: forward_flops,
            // Rule of thumb: the backward pass costs about twice the forward pass.
            training_flops_per_example: forward_flops.saturating_mul(3),
            activation_bytes_per_example,
            shapes: shapes.into_iter().map(|(k, v)| (k, show(&v))).collect(),
            tolerance_ops: tolerance_ops.into_iter().collect(),
            nodes: node_costs,
            input_bytes_per_example,
            integer_tensors: ints
                .into_iter()
                .filter(|(k, _)| produced.contains(k))
                .map(|(k, (r, _))| (k, r))
                .collect(),
        },
        index_limits,
    ))
}

/// Integer tensors through one node (ADR-0029): their range and origin, and the checks of the
/// operators that consume them as indices. An index whose range is known must fit the axis it
/// indexes; one that comes from the data registers a limit on that data, checked at loading.
fn integer_flow(
    node: &Node,
    input_shapes: &[Shape],
    ints: &mut BTreeMap<String, (IntRange, BTreeSet<String>)>,
    limits: &mut BTreeMap<String, u64>,
) -> Result<(), String> {
    let axis_attr = |shape: &Shape| -> usize {
        let r = shape.len() as i64;
        let a = node.attrs.get("axis").and_then(Attr::as_int).unwrap_or(0);
        (if a < 0 { a + r } else { a }) as usize
    };
    let op = node.op.as_str();
    let mut use_index = |name: &str, size: Option<u64>| -> Result<(), String> {
        let Some((range, sources)) = ints.get(name) else {
            return Ok(());
        };
        let Some(size) = size else {
            return Err(format!("{op}: the indexed axis must be known"));
        };
        match range.bound {
            Some(b) if b > size => Err(format!(
                "the index '{name}' takes values in [0, {b}), the axis it indexes has {size} \
elements"
            )),
            Some(_) => Ok(()),
            None => {
                for src in sources {
                    let l = limits.entry(src.clone()).or_insert(size);
                    *l = (*l).min(size);
                }
                Ok(())
            }
        }
    };
    match op {
        "Gather" => use_index(&node.inputs[1], input_shapes[0][0])?,
        "GatherElements" | "ScatterElements" => {
            let x = &input_shapes[0];
            let a = axis_attr(x);
            use_index(&node.inputs[1], x[a])?;
            if op == "ScatterElements"
                && matches!(node.attrs.get("reduction"), Some(Attr::Text(r)) if r == "none")
            {
                let distinct = ints.get(&node.inputs[1]).and_then(|(r, _)| r.distinct_axis);
                if distinct != Some(a) {
                    return Err(format!(
                        "an assignment (reduction none) with index '{}' whose values are not known \
to be distinct along axis {a}: repeated positions would depend on the order of the writes. \
Only the output of ArgTopK or ArgSort along that axis is accepted; otherwise use reduction add \
(ADR-0029)",
                        node.inputs[1]
                    ));
                }
            }
        }
        _ => {}
    }
    let produced = match op {
        "ArgMax" | "ArgMin" | "ArgTopK" | "ArgSort" => {
            let x = &input_shapes[0];
            let a = axis_attr(x);
            Some((
                IntRange {
                    bound: x[a],
                    distinct_axis: matches!(op, "ArgTopK" | "ArgSort").then_some(a),
                },
                BTreeSet::new(),
            ))
        }
        "Reshape" | "Transpose" | "Slice" | "Flatten" | "Concat"
            if ints.contains_key(&node.inputs[0]) =>
        {
            let mut bound = Some(0u64);
            let mut sources = BTreeSet::new();
            for i in &node.inputs {
                let (r, s) = &ints[i];
                bound = match (bound, r.bound) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    _ => None,
                };
                sources.extend(s.iter().cloned());
            }
            // A slice along another axis keeps whole lines, hence their distinct values.
            let distinct = match op {
                "Slice" => ints[&node.inputs[0]]
                    .0
                    .distinct_axis
                    .filter(|d| *d != axis_attr(&input_shapes[0])),
                _ => None,
            };
            Some((
                IntRange {
                    bound,
                    distinct_axis: distinct,
                },
                sources,
            ))
        }
        _ => None,
    };
    if let Some(info) = produced {
        ints.insert(node.output.clone(), info);
    }
    Ok(())
}

/// The published profile, in a transmissible form.
pub fn profile() -> Vec<&'static OpSpec> {
    PROFILE.iter().collect()
}
