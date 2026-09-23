//! Execution of a validated graph with Burn: forward pass, loss, backward pass through autodiff.
//!
//! The executor knows only the operators of the profile (ADR-0016) and only ever receives a graph
//! that is already validated: shapes, attributes and ranks were checked beforehand. The checks left
//! here are defensive and name what they reject; none of them is a "to be done later" path.
//!
//! Ranks are dynamic in the description and constant in Burn: `Dyn` carries a tensor of rank 1 to 4
//! and every operator is expanded over those four cases.

use std::collections::BTreeMap;

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::module::{avg_pool2d, max_pool2d};
use burn::tensor::{Bool, ElementConversion, Int, Tensor, TensorData, activation};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

use super::ops::{Attr, CAUSAL_FILL};
use super::{GraphError, GraphReport, Init, Loss, ModelGraphV1, Node, ParamSpec};

/// Why an execution fails. A validated graph should never produce one; if it does, the message
/// says whether validation or the executor is at fault.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecError {
    /// A tensor was expected but not provided.
    Missing(String),
    /// Unexpected shape or rank at execution time.
    Shape(String),
    /// Unexpected type (float versus integer).
    Dtype(String),
    /// Non-finite or non-integer value where one is required.
    Numeric(String),
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecError::Missing(n) => write!(f, "tensor '{n}' not provided"),
            ExecError::Shape(m) | ExecError::Dtype(m) | ExecError::Numeric(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ExecError {}

/// A graph ready for execution: validated, with its nodes indexed by output.
#[derive(Clone, Debug)]
pub struct Prepared {
    graph: ModelGraphV1,
    report: GraphReport,
    by_output: BTreeMap<String, usize>,
    /// Exclusive upper bound of each i64 input used as an index (ADR-0029), checked at loading.
    index_limits: BTreeMap<String, u64>,
}

impl Prepared {
    /// Validates, then indexes.
    pub fn new(graph: &ModelGraphV1) -> Result<Self, GraphError> {
        let (report, index_limits) = super::validate_full(graph)?;
        let by_output = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.output.clone(), i))
            .collect();
        Ok(Self {
            graph: graph.clone(),
            report,
            by_output,
            index_limits,
        })
    }

    /// The validation report.
    pub fn report(&self) -> &GraphReport {
        &self.report
    }

    /// The graph.
    pub fn graph(&self) -> &ModelGraphV1 {
        &self.graph
    }
}

/// Result of a step: mean loss of the micro-batch and gradient of each trained parameter.
#[derive(Clone, Debug)]
pub struct StepResult {
    /// Loss.
    pub loss: f32,
    /// Gradients, by **trained** parameter name, in the shape of the parameter. Frozen parameters
    /// are absent: they receive no gradient and are not transferred.
    pub gradients: BTreeMap<String, TensorF32>,
    /// Peak bytes of activations actually held by the executor, measured, not estimated. This is
    /// the number that tells whether activation recomputation paid off.
    pub peak_activation_bytes: u64,
    /// Peak bytes of parameters held on the device. With parameter offloading, this is the largest
    /// segment instead of the whole model.
    pub peak_parameter_bytes: u64,
}

/// How the executor uses memory (ADR-0019). Default: everything at once, as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecOptions {
    /// Number of recomputation segments. `1` keeps every activation (the original behavior).
    /// Above that, the forward pass keeps only segment boundaries and each segment is recomputed
    /// during the backward pass: activation memory goes down, compute goes up.
    pub segments: usize,
    /// Load onto the device only the parameters of the current segment, instead of all of them.
    /// Values stay in host memory between segments.
    pub offload_parameters: bool,
}

impl Default for ExecOptions {
    fn default() -> Self {
        Self {
            segments: 1,
            offload_parameters: false,
        }
    }
}

/// DRBG domain for initialization.
const INIT_DOMAIN: &str = "gg-graph/init/v1";
/// DRBG domain for Dropout masks.
const DROPOUT_DOMAIN: &str = "gg-graph/dropout/v1";

/// Frequencies `ω_i = base^(−2i/d)` for `i < d/2`, computed in f32 the way reference models do
/// (`1 / base^(2i/d)`), so that the tables match theirs up to rounding.
pub fn position_frequencies(base: f64, d: usize) -> Vec<f32> {
    let b = base as f32;
    (0..d / 2)
        .map(|i| 1.0f32 / b.powf((2 * i) as f32 / d as f32))
        .collect()
}

/// Position table `[1, …, 1, T, d]` of a `Sinusoidal`, `RotaryCos` or `RotarySin` initialization.
/// The angle `t·ω` is computed in f32, as in the reference models; a rotary table is
/// `A·cos(t·ω)` or `A·sin(t·ω)` whatever the law of its frequencies (ADR-0022).
pub fn position_table(init: &Init, shape: &[u64]) -> Vec<f32> {
    let r = shape.len();
    let (t_len, d) = (shape[r - 2] as usize, shape[r - 1] as usize);
    let half = d / 2;
    let mut out = Vec::with_capacity(t_len * d);
    match init {
        Init::RotaryCos(rotary) | Init::RotarySin(rotary) => {
            let (freqs, amplitude) = rotary.frequencies_f32(d);
            let cosine = matches!(init, Init::RotaryCos(_));
            for t in 0..t_len {
                let tf = t as f32;
                for j in 0..d {
                    let angle = tf * freqs[j % half];
                    let v = if cosine { angle.cos() } else { angle.sin() };
                    out.push(v * amplitude);
                }
            }
        }
        Init::Sinusoidal { base } => {
            let freqs = position_frequencies(*base, d);
            for t in 0..t_len {
                let tf = t as f32;
                for j in 0..d {
                    let angle = tf * freqs[j / 2];
                    out.push(if j.is_multiple_of(2) {
                        angle.sin()
                    } else {
                        angle.cos()
                    });
                }
            }
        }
        _ => unreachable!("position table"),
    }
    out
}

/// Initializes the parameters of a graph from a seed: reproducible on any machine.
pub fn init_parameters(graph: &ModelGraphV1, seed: u64) -> BTreeMap<String, TensorF32> {
    graph
        .parameters
        .iter()
        .map(|p| (p.name.clone(), init_parameter(p, seed)))
        .collect()
}

/// Initializes a single parameter: each one has its own DRBG stream, so a checkpoint shard is
/// initialized without computing the others (C2), identically.
pub fn init_parameter(p: &ParamSpec, seed: u64) -> TensorF32 {
    {
        let count: usize = p.shape.iter().product::<u64>() as usize;
        let mut d = Drbg::new(&format!("{INIT_DOMAIN}/{}", p.name), seed);
        let data: Vec<f32> = match &p.init {
            Init::Zeros => vec![0.0; count],
            Init::Ones => vec![1.0; count],
            Init::XavierUniform => {
                let fan_in: u64 = if p.shape.len() >= 2 {
                    p.shape[1..].iter().product()
                } else {
                    p.shape[0]
                };
                let bound = 1.0 / (fan_in.max(1) as f32).sqrt();
                (0..count).map(|_| d.next_sym_f32() * bound).collect()
            }
            Init::Constant { value } => vec![*value as f32; count],
            Init::Sinusoidal { .. } | Init::RotaryCos(_) | Init::RotarySin(_) => {
                position_table(&p.init, &p.shape)
            }
            Init::Normal { std } => {
                let std = *std as f32;
                (0..count)
                    .map(|_| {
                        // Box-Muller; u1 is bounded away from zero to stay finite.
                        let u1 = d.next_unit_f32().max(1e-7);
                        let u2 = d.next_unit_f32();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
                    })
                    .collect()
            }
        };
        TensorF32::new(p.shape.clone(), data).expect("validated shape")
    }
}

/// Tensor of dynamic rank.
pub enum Dyn<B: Backend> {
    /// Float, rank 1.
    F1(Tensor<B, 1>),
    /// Float, rank 2.
    F2(Tensor<B, 2>),
    /// Float, rank 3.
    F3(Tensor<B, 3>),
    /// Float, rank 4.
    F4(Tensor<B, 4>),
    /// Integer, rank 1.
    I1(Tensor<B, 1, Int>),
    /// Integer, rank 2.
    I2(Tensor<B, 2, Int>),
    /// Integer, rank 3 (ADR-0029).
    I3(Tensor<B, 3, Int>),
    /// Integer, rank 4.
    I4(Tensor<B, 4, Int>),
}

impl<B: Backend> Clone for Dyn<B> {
    fn clone(&self) -> Self {
        match self {
            Dyn::F1(t) => Dyn::F1(t.clone()),
            Dyn::F2(t) => Dyn::F2(t.clone()),
            Dyn::F3(t) => Dyn::F3(t.clone()),
            Dyn::F4(t) => Dyn::F4(t.clone()),
            Dyn::I1(t) => Dyn::I1(t.clone()),
            Dyn::I2(t) => Dyn::I2(t.clone()),
            Dyn::I3(t) => Dyn::I3(t.clone()),
            Dyn::I4(t) => Dyn::I4(t.clone()),
        }
    }
}

impl<B: Backend> Dyn<B> {
    /// Dimensions.
    pub fn dims(&self) -> Vec<usize> {
        match self {
            Dyn::F1(t) => t.dims().to_vec(),
            Dyn::F2(t) => t.dims().to_vec(),
            Dyn::F3(t) => t.dims().to_vec(),
            Dyn::F4(t) => t.dims().to_vec(),
            Dyn::I1(t) => t.dims().to_vec(),
            Dyn::I2(t) => t.dims().to_vec(),
            Dyn::I3(t) => t.dims().to_vec(),
            Dyn::I4(t) => t.dims().to_vec(),
        }
    }

    fn is_float(&self) -> bool {
        matches!(self, Dyn::F1(_) | Dyn::F2(_) | Dyn::F3(_) | Dyn::F4(_))
    }
}

fn shape_err(msg: impl Into<String>) -> ExecError {
    ExecError::Shape(msg.into())
}

fn dtype_err(msg: impl Into<String>) -> ExecError {
    ExecError::Dtype(msg.into())
}

/// Applies a function to a float tensor, rank by rank.
macro_rules! map_float {
    ($t:expr, $x:ident => $e:expr, $what:expr) => {
        match $t {
            Dyn::F1($x) => Dyn::F1($e),
            Dyn::F2($x) => Dyn::F2($e),
            Dyn::F3($x) => Dyn::F3($e),
            Dyn::F4($x) => Dyn::F4($e),
            _ => return Err(dtype_err(format!("{} expects a float tensor", $what))),
        }
    };
}

/// Applies a function to two float tensors of the same rank.
macro_rules! zip_float {
    ($a:expr, $b:expr, $x:ident, $y:ident => $e:expr, $what:expr) => {
        match ($a, $b) {
            (Dyn::F1($x), Dyn::F1($y)) => Dyn::F1($e),
            (Dyn::F2($x), Dyn::F2($y)) => Dyn::F2($e),
            (Dyn::F3($x), Dyn::F3($y)) => Dyn::F3($e),
            (Dyn::F4($x), Dyn::F4($y)) => Dyn::F4($e),
            _ => {
                return Err(shape_err(format!(
                    "{} expects two float tensors of the same rank",
                    $what
                )));
            }
        }
    };
}

fn reshape_to<B: Backend, const D: usize>(
    t: Tensor<B, D>,
    dims: &[usize],
) -> Result<Dyn<B>, ExecError> {
    Ok(match dims.len() {
        1 => Dyn::F1(t.reshape([dims[0]])),
        2 => Dyn::F2(t.reshape([dims[0], dims[1]])),
        3 => Dyn::F3(t.reshape([dims[0], dims[1], dims[2]])),
        4 => Dyn::F4(t.reshape([dims[0], dims[1], dims[2], dims[3]])),
        n => return Err(shape_err(format!("rank {n} outside 1..=4"))),
    })
}

fn reshape_int<B: Backend, const D: usize>(
    t: Tensor<B, D, Int>,
    dims: &[usize],
) -> Result<Dyn<B>, ExecError> {
    Ok(match dims.len() {
        1 => Dyn::I1(t.reshape([dims[0]])),
        2 => Dyn::I2(t.reshape([dims[0], dims[1]])),
        3 => Dyn::I3(t.reshape([dims[0], dims[1], dims[2]])),
        4 => Dyn::I4(t.reshape([dims[0], dims[1], dims[2], dims[3]])),
        n => return Err(shape_err(format!("rank {n} outside 1..=4"))),
    })
}

/// Applies a function to an integer tensor, rank by rank (ADR-0029).
macro_rules! map_int {
    ($t:expr, $x:ident => $e:expr, $what:expr) => {
        match $t {
            Dyn::I1($x) => Dyn::I1($e),
            Dyn::I2($x) => Dyn::I2($e),
            Dyn::I3($x) => Dyn::I3($e),
            Dyn::I4($x) => Dyn::I4($e),
            _ => return Err(dtype_err(format!("{} expects an integer tensor", $what))),
        }
    };
}

/// Applies a shape function to a tensor of either type, rank by rank.
macro_rules! map_any {
    ($t:expr, $x:ident => $e:expr, $what:expr) => {
        match $t {
            t @ (Dyn::F1(_) | Dyn::F2(_) | Dyn::F3(_) | Dyn::F4(_)) => {
                map_float!(t, $x => $e, $what)
            }
            t => map_int!(t, $x => $e, $what),
        }
    };
}

/// Changes the shape of a tensor (float or integer); the number of elements must be preserved.
fn reshape_dyn<B: Backend>(t: Dyn<B>, dims: &[usize]) -> Result<Dyn<B>, ExecError> {
    let before: usize = t.dims().iter().product();
    let after: usize = dims.iter().product();
    if before != after {
        return Err(shape_err(format!(
            "Reshape: {before} elements to {dims:?} ({after} elements)"
        )));
    }
    match t {
        Dyn::F1(x) => reshape_to(x, dims),
        Dyn::F2(x) => reshape_to(x, dims),
        Dyn::F3(x) => reshape_to(x, dims),
        Dyn::F4(x) => reshape_to(x, dims),
        Dyn::I1(x) => reshape_int(x, dims),
        Dyn::I2(x) => reshape_int(x, dims),
        Dyn::I3(x) => reshape_int(x, dims),
        Dyn::I4(x) => reshape_int(x, dims),
    }
}

/// Prepends size-1 dimensions up to the requested rank (NumPy-style broadcasting).
fn align<B: Backend>(t: Dyn<B>, rank: usize) -> Result<Dyn<B>, ExecError> {
    let dims = t.dims();
    if dims.len() >= rank {
        return Ok(t);
    }
    let mut target = vec![1usize; rank - dims.len()];
    target.extend(dims);
    reshape_dyn(t, &target)
}

fn norm_axis(axis: i64, rank: usize) -> Result<usize, ExecError> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        return Err(shape_err(format!("axis {axis} outside rank {rank}")));
    }
    Ok(a as usize)
}

fn attr_int(attrs: &BTreeMap<String, Attr>, key: &str, default: i64) -> i64 {
    attrs.get(key).and_then(Attr::as_int).unwrap_or(default)
}

fn attr_float(attrs: &BTreeMap<String, Attr>, key: &str, default: f64) -> f64 {
    attrs.get(key).and_then(Attr::as_float).unwrap_or(default)
}

fn attr_pair(
    attrs: &BTreeMap<String, Attr>,
    key: &str,
    default: [usize; 2],
) -> Result<[usize; 2], ExecError> {
    match attrs.get(key) {
        None => Ok(default),
        Some(Attr::Int(v)) => {
            let v = usize::try_from(*v).map_err(|_| shape_err(format!("negative {key}")))?;
            Ok([v, v])
        }
        Some(Attr::Ints(v)) => match v.as_slice() {
            [a] => {
                let a = usize::try_from(*a).map_err(|_| shape_err(format!("negative {key}")))?;
                Ok([a, a])
            }
            [a, b] => Ok([
                usize::try_from(*a).map_err(|_| shape_err(format!("negative {key}")))?,
                usize::try_from(*b).map_err(|_| shape_err(format!("negative {key}")))?,
            ]),
            _ => Err(shape_err(format!("{key} expects one or two integers"))),
        },
        Some(_) => Err(shape_err(format!("{key} expects integers"))),
    }
}

/// Mean over an axis, keeping the dimension.
fn mean_axis<B: Backend>(t: Dyn<B>, axis: usize) -> Result<Dyn<B>, ExecError> {
    Ok(map_float!(t, x => x.mean_dim(axis), "mean"))
}

fn binary<B: Backend>(op: &str, a: Dyn<B>, b: Dyn<B>) -> Result<Dyn<B>, ExecError> {
    let rank = a.dims().len().max(b.dims().len());
    let a = align(a, rank)?;
    let b = align(b, rank)?;
    Ok(match op {
        "Add" => zip_float!(a, b, x, y => x.add(y), op),
        "Sub" => zip_float!(a, b, x, y => x.sub(y), op),
        "Mul" => zip_float!(a, b, x, y => x.mul(y), op),
        "Div" => zip_float!(a, b, x, y => x.div(y), op),
        other => return Err(shape_err(format!("unknown binary operator {other}"))),
    })
}

fn matmul<B: Backend>(a: Dyn<B>, b: Dyn<B>) -> Result<Dyn<B>, ExecError> {
    let (ra, rb) = (a.dims().len(), b.dims().len());
    if !a.is_float() || !b.is_float() {
        return Err(dtype_err("MatMul expects float tensors"));
    }
    if ra > 2 && rb == 2 {
        // [.., m, k] × [k, n]: flatten the batch dimensions, rank-2 product, reshape back.
        let dims = a.dims();
        let k = dims[ra - 1];
        let rows: usize = dims[..ra - 1].iter().product();
        let a2 = match reshape_dyn(a, &[rows, k])? {
            Dyn::F2(t) => t,
            _ => unreachable!("reshape to rank 2"),
        };
        let b2 = match b {
            Dyn::F2(t) => t,
            _ => unreachable!("rank 2 checked above"),
        };
        let n = b2.dims()[1];
        let out = a2.matmul(b2);
        let mut target = dims[..ra - 1].to_vec();
        target.push(n);
        return reshape_dyn(Dyn::F2(out), &target);
    }
    let rank = ra.max(rb);
    let a = align(a, rank)?;
    let b = align(b, rank)?;
    Ok(zip_float!(a, b, x, y => x.matmul(y), "MatMul"))
}

/// Mask of future positions: true at `(i, j)` when `j > i`, that is, when key `j` comes after
/// query `i`. Built on the host so that the semantics depend on no backend convention (the one of
/// `triu_mask` in Burn is the opposite of intuition).
fn future_mask<B: Backend>(device: &B::Device, t: usize) -> Tensor<B, 2, Bool> {
    let mut bits = Vec::with_capacity(t * t);
    for i in 0..t {
        for j in 0..t {
            bits.push(j > i);
        }
    }
    Tensor::<B, 2, Bool>::from_data(TensorData::new(bits, [t, t]), device)
}

/// Replaces the scores of future positions with `CAUSAL_FILL`, over the last two axes.
fn causal_mask<B: Backend>(device: &B::Device, t: Dyn<B>) -> Result<Dyn<B>, ExecError> {
    let dims = t.dims();
    let rank = dims.len();
    if rank < 2 || dims[rank - 1] != dims[rank - 2] {
        return Err(shape_err(format!(
            "CausalMask expects square scores [.., T, T], got {dims:?}"
        )));
    }
    let n = dims[rank - 1];
    let mask = future_mask::<B>(device, n);
    Ok(match t {
        Dyn::F2(x) => Dyn::F2(x.mask_fill(mask, CAUSAL_FILL)),
        Dyn::F3(x) => {
            let [a, _, _] = x.dims();
            let m = mask.reshape([1, n, n]).expand([a, n, n]);
            Dyn::F3(x.mask_fill(m, CAUSAL_FILL))
        }
        Dyn::F4(x) => {
            let [a, b, _, _] = x.dims();
            let m = mask.reshape([1, 1, n, n]).expand([a, b, n, n]);
            Dyn::F4(x.mask_fill(m, CAUSAL_FILL))
        }
        _ => {
            return Err(dtype_err("CausalMask expects float scores of rank 2 to 4"));
        }
    })
}

fn softmax_axis<B: Backend>(t: Dyn<B>, axis: i64) -> Result<Dyn<B>, ExecError> {
    let axis = norm_axis(axis, t.dims().len())?;
    Ok(map_float!(t, x => activation::softmax(x, axis), "Softmax"))
}

/// Layer normalization: (x − mean) / sqrt(variance + ε) over the axis, then gain and bias.
fn layer_norm<B: Backend>(
    x: Dyn<B>,
    gain: Option<Dyn<B>>,
    bias: Option<Dyn<B>>,
    axis: i64,
    eps: f64,
) -> Result<Dyn<B>, ExecError> {
    let rank = x.dims().len();
    let axis = norm_axis(axis, rank)?;
    let mean = mean_axis(x.clone(), axis)?;
    let centered = binary("Sub", x, mean)?;
    let var = mean_axis(
        map_float!(centered.clone(), c => c.clone().mul(c), "LayerNorm"),
        axis,
    )?;
    let denom = map_float!(var, v => v.add_scalar(eps as f32).sqrt(), "LayerNorm");
    let mut out = binary("Div", centered, denom)?;
    if let Some(g) = gain {
        out = binary("Mul", out, g)?;
    }
    if let Some(b) = bias {
        out = binary("Add", out, b)?;
    }
    Ok(out)
}

/// Root-mean-square normalization: x / sqrt(mean(x²) + ε), then gain.
fn rms_norm<B: Backend>(
    x: Dyn<B>,
    gain: Option<Dyn<B>>,
    axis: i64,
    eps: f64,
) -> Result<Dyn<B>, ExecError> {
    let rank = x.dims().len();
    let axis = norm_axis(axis, rank)?;
    let sq = map_float!(x.clone(), c => c.clone().mul(c), "RMSNorm");
    let ms = mean_axis(sq, axis)?;
    let denom = map_float!(ms, v => v.add_scalar(eps as f32).sqrt(), "RMSNorm");
    let mut out = binary("Div", x, denom)?;
    if let Some(g) = gain {
        out = binary("Mul", out, g)?;
    }
    Ok(out)
}

fn as_f4<B: Backend>(t: Dyn<B>, what: &str) -> Result<Tensor<B, 4>, ExecError> {
    match t {
        Dyn::F4(x) => Ok(x),
        other => Err(shape_err(format!(
            "{what} expects [N, C, H, W], got rank {}",
            other.dims().len()
        ))),
    }
}

fn as_f1<B: Backend>(t: Dyn<B>, what: &str) -> Result<Tensor<B, 1>, ExecError> {
    match t {
        Dyn::F1(x) => Ok(x),
        other => Err(shape_err(format!(
            "{what} expects a vector, got rank {}",
            other.dims().len()
        ))),
    }
}

/// Batch normalization in training mode: per-channel statistics of the micro-batch.
/// The running means and variances (inputs 4 and 5) serve inference and take no part in the
/// gradient; they are accepted and ignored here, as docs/MODELS.md states.
fn batch_norm<B: Backend>(
    x: Dyn<B>,
    gamma: Dyn<B>,
    beta: Dyn<B>,
    eps: f64,
) -> Result<Dyn<B>, ExecError> {
    let x = as_f4(x, "BatchNorm2d")?;
    let c = x.dims()[1];
    let mean = x.clone().mean_dim(0).mean_dim(2).mean_dim(3);
    let centered = x.sub(mean);
    let var = centered
        .clone()
        .mul(centered.clone())
        .mean_dim(0)
        .mean_dim(2)
        .mean_dim(3);
    let normed = centered.div(var.add_scalar(eps as f32).sqrt());
    let gamma = as_f1(gamma, "BatchNorm2d (gamma)")?.reshape([1, c, 1, 1]);
    let beta = as_f1(beta, "BatchNorm2d (beta)")?.reshape([1, c, 1, 1]);
    Ok(Dyn::F4(normed.mul(gamma).add(beta)))
}

/// Target shape of a Reshape from the actual dimensions of the input.
fn reshape_target(input: &[usize], spec: &[i64]) -> Result<Vec<usize>, ExecError> {
    let total: usize = input.iter().product();
    let mut out = Vec::with_capacity(spec.len());
    let mut infer = None;
    let mut known = 1usize;
    for (i, d) in spec.iter().enumerate() {
        match *d {
            0 => {
                let v = *input.get(i).ok_or_else(|| {
                    shape_err(format!("Reshape: copied dimension {i} is out of rank"))
                })?;
                out.push(v);
                known *= v;
            }
            -1 => {
                if infer.is_some() {
                    return Err(shape_err("Reshape: only one -1 dimension allowed"));
                }
                infer = Some(i);
                out.push(0);
            }
            v if v > 0 => {
                out.push(v as usize);
                known *= v as usize;
            }
            v => return Err(shape_err(format!("Reshape: invalid dimension {v}"))),
        }
    }
    if let Some(i) = infer {
        if known == 0 || !total.is_multiple_of(known) {
            return Err(shape_err(format!(
                "Reshape: {total} elements do not fit into {spec:?}"
            )));
        }
        out[i] = total / known;
    }
    Ok(out)
}

fn permute_dyn<B: Backend>(t: Dyn<B>, perm: &[i64]) -> Result<Dyn<B>, ExecError> {
    let rank = t.dims().len();
    if perm.len() != rank {
        return Err(shape_err(format!(
            "Transpose: perm of length {} for rank {rank}",
            perm.len()
        )));
    }
    let p: Vec<usize> = perm
        .iter()
        .map(|v| usize::try_from(*v).map_err(|_| shape_err("Transpose: negative index")))
        .collect::<Result<_, _>>()?;
    Ok(match t {
        Dyn::F1(x) => Dyn::F1(x.permute([p[0]])),
        Dyn::F2(x) => Dyn::F2(x.permute([p[0], p[1]])),
        Dyn::F3(x) => Dyn::F3(x.permute([p[0], p[1], p[2]])),
        Dyn::F4(x) => Dyn::F4(x.permute([p[0], p[1], p[2], p[3]])),
        Dyn::I1(x) => Dyn::I1(x.permute([p[0]])),
        Dyn::I2(x) => Dyn::I2(x.permute([p[0], p[1]])),
        Dyn::I3(x) => Dyn::I3(x.permute([p[0], p[1], p[2]])),
        Dyn::I4(x) => Dyn::I4(x.permute([p[0], p[1], p[2], p[3]])),
    })
}

fn concat<B: Backend>(parts: Vec<Dyn<B>>, axis: usize) -> Result<Dyn<B>, ExecError> {
    let Some(first) = parts.first() else {
        return Err(shape_err("Concat without input"));
    };
    let rank = first.dims().len();
    macro_rules! collect {
        ($variant:ident) => {{
            let mut v = Vec::with_capacity(parts.len());
            for p in parts {
                match p {
                    Dyn::$variant(t) => v.push(t),
                    _ => return Err(shape_err("Concat: mismatched ranks or types")),
                }
            }
            Dyn::$variant(Tensor::cat(v, axis))
        }};
    }
    Ok(match (rank, first.is_float()) {
        (1, true) => collect!(F1),
        (2, true) => collect!(F2),
        (3, true) => collect!(F3),
        (4, true) => collect!(F4),
        (1, false) => collect!(I1),
        (2, false) => collect!(I2),
        (3, false) => collect!(I3),
        (4, false) => collect!(I4),
        (n, _) => return Err(shape_err(format!("Concat: unsupported rank {n}"))),
    })
}

/// Selects rows (axis 0) of a table of rank 2 to 4 with integer indices of any rank: the indices
/// are flattened, the rows selected, and the result takes the shape `indices ++ table[1..]`.
fn gather_rows<B: Backend>(table: Dyn<B>, idx: Dyn<B>) -> Result<Dyn<B>, ExecError> {
    if !table.is_float() || idx.is_float() {
        return Err(dtype_err(
            "Gather: a float table and integer indices are expected",
        ));
    }
    let tdims = table.dims();
    let idims = idx.dims();
    let n: usize = idims.iter().product();
    let flat = match reshape_dyn(idx, &[n])? {
        Dyn::I1(i) => i,
        _ => unreachable!("reshape of integers to rank 1"),
    };
    let picked = map_float!(table, t => t.select(0, flat.clone()), "Gather");
    let mut target = idims;
    target.extend_from_slice(&tdims[1..]);
    reshape_dyn(picked, &target)
}

/// Dropout mask drawn from the DRBG: 0 with probability `rate`, otherwise 1/(1 − rate).
fn dropout<B: Backend>(
    device: &B::Device,
    t: Dyn<B>,
    rate: f64,
    seed: u64,
    node: &str,
) -> Result<Dyn<B>, ExecError> {
    if !(0.0..1.0).contains(&rate) {
        return Err(ExecError::Numeric(format!(
            "Dropout: rate {rate} outside [0, 1)"
        )));
    }
    // A zero rate keeps everything and scales by 1: the mask is all ones, and multiplying by 1.0 is
    // exact, so the value is returned as it is. Drawing that mask on the host cost 11 ms per call
    // on a ViT-tiny batch, 275 of the 300 ms of its forward pass.
    if rate == 0.0 {
        return Ok(t);
    }
    let dims = t.dims();
    let count: usize = dims.iter().product();
    let mut d = Drbg::new(&format!("{DROPOUT_DOMAIN}/{node}"), seed);
    let keep = 1.0 / (1.0 - rate as f32);
    let mask: Vec<f32> = (0..count)
        .map(|_| {
            if (d.next_unit_f32() as f64) < rate {
                0.0
            } else {
                keep
            }
        })
        .collect();
    let m = Tensor::<B, 1>::from_data(TensorData::new(mask, [count]), device);
    let m = reshape_to(m, &dims)?;
    binary("Mul", t, m)
}

fn from_tensor_f32<B: Backend>(device: &B::Device, t: &TensorF32) -> Result<Dyn<B>, ExecError> {
    from_tensor_f32_owned(device, t.clone())
}

/// Like [`from_tensor_f32`], moving the host buffer instead of copying it: a copy of a 113 MB
/// tensor cost 55 ms on the host, before any transfer.
fn from_tensor_f32_owned<B: Backend>(
    device: &B::Device,
    t: TensorF32,
) -> Result<Dyn<B>, ExecError> {
    let dims: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
    let n = t.data.len();
    let flat = Tensor::<B, 1>::from_data(TensorData::new(t.data, [n]), device);
    reshape_to(flat, &dims)
}

/// An index tensor is delivered as a file of floats with integer values.
///
/// When the input is used as an index (`limit`, from validation, ADR-0029), every value must be in
/// `[0, limit)`: an out-of-range index is a named error here, before any computation, never a read
/// the GPU would silently clamp.
fn int_from_tensor_f32<B: Backend>(
    device: &B::Device,
    name: &str,
    t: &TensorF32,
    limit: Option<u64>,
) -> Result<Dyn<B>, ExecError> {
    let mut values = Vec::with_capacity(t.data.len());
    for (k, v) in t.data.iter().enumerate() {
        if !v.is_finite() || v.fract() != 0.0 || v.abs() > 16_777_216.0 {
            return Err(ExecError::Numeric(format!(
                "'{name}': value {v} is not a representable integer index"
            )));
        }
        if let Some(l) = limit
            && (*v < 0.0 || *v as u64 >= l)
        {
            return Err(ExecError::Numeric(format!(
                "'{name}': value {v} at position {k} is outside [0, {l}), the axis it indexes"
            )));
        }
        values.push(*v as i64);
    }
    let dims: Vec<usize> = t.shape.iter().map(|d| *d as usize).collect();
    let flat = Tensor::<B, 1, Int>::from_data(TensorData::new(values, [t.data.len()]), device);
    reshape_int(flat, &dims).map_err(|_| {
        shape_err(format!(
            "'{name}': indices of rank {}, expected 1 to 4",
            dims.len()
        ))
    })
}

fn to_tensor_f32<B: Backend, const D: usize>(t: Tensor<B, D>) -> Result<TensorF32, ExecError> {
    let shape: Vec<u64> = t.dims().iter().map(|d| *d as u64).collect();
    // `into_vec` takes the buffer over when the backend's bytes allow it (no copy on LibTorch; on
    // WGPU it still copies): `to_vec` copied every gradient once more on the host.
    let values = t
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| ExecError::Dtype(format!("conversion to f32: {e:?}")))?;
    TensorF32::new(shape, values).map_err(|e| ExecError::Shape(e.to_string()))
}

/// Intermediate value of a `Formula`: a number (constant subexpression) or a tensor.
enum Val<B: Backend> {
    Num(f64),
    T(Dyn<B>),
}

/// Applies `f` to a float tensor of any rank.
macro_rules! float_fn {
    ($t:expr, $x:ident => $e:expr) => {
        map_float!($t, $x => $e, "Formula")
    };
}

/// `x^p` for a literal exponent. An integer of absolute value ≤ 8 is computed by repeated
/// multiplication: exact for a negative base (`(-2)^3 = -8`, where `powf` would return `NaN`) and
/// identical across backends. Other exponents go through `powf`, as in PyTorch.
fn formula_pow<B: Backend>(x: Dyn<B>, p: f64) -> Result<Dyn<B>, ExecError> {
    if p.fract() == 0.0 && p.abs() <= 8.0 {
        let n = p.abs() as u32;
        if n == 0 {
            return Ok(float_fn!(x, t => t.zeros_like().add_scalar(1.0)));
        }
        let mut acc = x.clone();
        for _ in 1..n {
            acc = binary("Mul", acc, x.clone())?;
        }
        return Ok(if p < 0.0 {
            float_fn!(acc, t => t.recip())
        } else {
            acc
        });
    }
    Ok(float_fn!(x, t => t.powf_scalar(p as f32)))
}

fn formula_call<B: Backend>(
    func: super::formula::Func,
    x: Dyn<B>,
    y: Option<Val<B>>,
) -> Result<Dyn<B>, ExecError> {
    use super::formula::Func;
    Ok(match func {
        Func::Exp => float_fn!(x, t => t.exp()),
        Func::Log => float_fn!(x, t => t.log()),
        Func::Log1p => float_fn!(x, t => t.log1p()),
        Func::Sqrt => float_fn!(x, t => t.sqrt()),
        Func::Rsqrt => float_fn!(x, t => t.sqrt().recip()),
        Func::Abs => float_fn!(x, t => t.abs()),
        Func::Tanh => float_fn!(x, t => activation::tanh(t)),
        Func::Sigmoid => float_fn!(x, t => activation::sigmoid(t)),
        Func::Relu => float_fn!(x, t => activation::relu(t)),
        Func::Sin => float_fn!(x, t => t.sin()),
        Func::Cos => float_fn!(x, t => t.cos()),
        Func::Erf => float_fn!(x, t => t.erf()),
        Func::Tan => float_fn!(x, t => t.tan()),
        Func::Asin => float_fn!(x, t => t.asin()),
        Func::Acos => float_fn!(x, t => t.acos()),
        Func::Atan => float_fn!(x, t => t.atan()),
        Func::Sinh => float_fn!(x, t => t.sinh()),
        Func::Cosh => float_fn!(x, t => t.cosh()),
        Func::Asinh => float_fn!(x, t => t.asinh()),
        Func::Acosh => float_fn!(x, t => t.acosh()),
        Func::Atanh => float_fn!(x, t => t.atanh()),
        Func::Floor => float_fn!(x, t => t.floor()),
        Func::Ceil => float_fn!(x, t => t.ceil()),
        Func::Trunc => float_fn!(x, t => t.trunc()),
        Func::Sign => float_fn!(x, t => t.sign()),
        Func::Round => float_fn!(x, t => round_half_even(t)),
        // Stable form: never takes exp of a large positive number.
        Func::Softplus => {
            float_fn!(x, t => activation::relu(t.clone()).add(t.abs().neg().exp().log1p()))
        }
        Func::Atan2 => {
            // atan2(y, x): `x` here is the first argument (y), `y` the second one.
            let other = match y {
                Some(Val::T(o)) => o,
                Some(Val::Num(v)) => constant_like(&x, v)?,
                None => return Err(shape_err("Formula: atan2 takes two arguments")),
            };
            let rank = x.dims().len().max(other.dims().len());
            let (a, b) = broadcast_pair(align(x, rank)?, align(other, rank)?)?;
            zip_float!(a, b, p, q => p.atan2(q), "Formula")
        }
        Func::Max | Func::Min => {
            let max = func == Func::Max;
            match y {
                Some(Val::Num(v)) => {
                    let v = v as f32;
                    if max {
                        float_fn!(x, t => t.clamp_min(v))
                    } else {
                        float_fn!(x, t => t.clamp_max(v))
                    }
                }
                Some(Val::T(y)) => {
                    let rank = x.dims().len().max(y.dims().len());
                    let (x, y) = (align(x, rank)?, align(y, rank)?);
                    let (x, y) = broadcast_pair(x, y)?;
                    if max {
                        zip_float!(x, y, a, b => a.max_pair(b), "Formula")
                    } else {
                        zip_float!(x, y, a, b => a.min_pair(b), "Formula")
                    }
                }
                None => return Err(shape_err("Formula: max and min take two arguments")),
            }
        }
    })
}

/// A tensor of the shape of `t`, filled with `v`, without a gradient path.
fn constant_like<B: Backend>(t: &Dyn<B>, v: f64) -> Result<Dyn<B>, ExecError> {
    Ok(
        float_fn!(t.clone(), u => u.detach().set_require_grad(false).zeros_like().add_scalar(v as f32)),
    )
}

/// Rounding half to even, as PyTorch (`round(2.5) = 2`, `round(-0.5) = -0`), on every backend:
/// the GPU's native rounding leaves ties to the implementation (SPIR-V `Round`). Written with
/// `floor` and exact comparisons: `x − floor(x)` is exact in f32. The derivative is zero, as that
/// of every piecewise-constant function.
fn round_half_even<B: Backend, const D: usize>(t: Tensor<B, D>) -> Tensor<B, D> {
    let f = t.clone().floor();
    let frac = t.sub(f.clone());
    let above = frac.clone().greater_elem(0.5).float();
    let tie = frac.equal_elem(0.5).float();
    // `f` odd: f − 2·floor(f / 2) = 1.
    let half = f.clone().div_scalar(2.0).floor().mul_scalar(2.0);
    let odd = f.clone().sub(half).equal_elem(1.0).float();
    f.add(above).add(tie.mul(odd))
}

/// Expands two tensors of the same rank to their common shape (which `max_pair` requires, unlike
/// arithmetic operations, which broadcast on their own).
fn broadcast_pair<B: Backend>(x: Dyn<B>, y: Dyn<B>) -> Result<(Dyn<B>, Dyn<B>), ExecError> {
    let (dx, dy) = (x.dims(), y.dims());
    if dx == dy {
        return Ok((x, y));
    }
    let mut target = Vec::with_capacity(dx.len());
    for (a, b) in dx.iter().zip(&dy) {
        target.push(match (*a, *b) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => {
                return Err(shape_err(format!(
                    "Formula: {dx:?} and {dy:?} do not broadcast"
                )));
            }
        });
    }
    let expand = |t: Dyn<B>| -> Result<Dyn<B>, ExecError> {
        Ok(match t {
            Dyn::F1(v) => Dyn::F1(v.expand([target[0]])),
            Dyn::F2(v) => Dyn::F2(v.expand([target[0], target[1]])),
            Dyn::F3(v) => Dyn::F3(v.expand([target[0], target[1], target[2]])),
            Dyn::F4(v) => Dyn::F4(v.expand([target[0], target[1], target[2], target[3]])),
            _ => return Err(dtype_err("Formula expects float tensors")),
        })
    };
    Ok((expand(x)?, expand(y)?))
}

fn formula_eval<B: Backend>(
    e: &super::formula::Expr,
    inputs: &[Dyn<B>],
) -> Result<Val<B>, ExecError> {
    use super::formula::{BinOp, Expr};
    if let Some(v) = e.constant_value() {
        return Ok(Val::Num(v));
    }
    Ok(Val::T(match e {
        Expr::Num(_) => unreachable!("constant folded"),
        Expr::Var(i) => inputs
            .get(*i)
            .cloned()
            .ok_or_else(|| shape_err("Formula: missing input"))?,
        Expr::Neg(x) => match formula_eval(x, inputs)? {
            Val::T(t) => float_fn!(t, v => v.neg()),
            Val::Num(_) => unreachable!("constant folded"),
        },
        Expr::Pow(x, p) => match formula_eval(x, inputs)? {
            Val::T(t) => formula_pow(t, *p)?,
            Val::Num(_) => unreachable!("constant folded"),
        },
        Expr::Bin(op, l, r) => {
            let (l, r) = (formula_eval(l, inputs)?, formula_eval(r, inputs)?);
            match (l, r) {
                (Val::T(x), Val::T(y)) => {
                    let name = match op {
                        BinOp::Add => "Add",
                        BinOp::Sub => "Sub",
                        BinOp::Mul => "Mul",
                        BinOp::Div => "Div",
                    };
                    binary(name, x, y)?
                }
                (Val::T(x), Val::Num(v)) => {
                    let v = v as f32;
                    match op {
                        BinOp::Add => float_fn!(x, t => t.add_scalar(v)),
                        BinOp::Sub => float_fn!(x, t => t.sub_scalar(v)),
                        BinOp::Mul => float_fn!(x, t => t.mul_scalar(v)),
                        BinOp::Div => float_fn!(x, t => t.div_scalar(v)),
                    }
                }
                (Val::Num(v), Val::T(y)) => {
                    let v = v as f32;
                    match op {
                        BinOp::Add => float_fn!(y, t => t.add_scalar(v)),
                        BinOp::Sub => float_fn!(y, t => t.neg().add_scalar(v)),
                        BinOp::Mul => float_fn!(y, t => t.mul_scalar(v)),
                        BinOp::Div => float_fn!(y, t => t.recip().mul_scalar(v)),
                    }
                }
                (Val::Num(_), Val::Num(_)) => unreachable!("constant folded"),
            }
        }
        Expr::Call(func, args) => {
            let mut vals = args
                .iter()
                .map(|a| formula_eval(a, inputs))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter();
            let first = vals.next();
            let second = vals.next();
            let (x, y) = match (first, second) {
                (Some(Val::T(x)), y) => (x, y),
                // atan2(2, a) is not symmetric: the constant becomes a tensor, in its place.
                (Some(Val::Num(v)), Some(Val::T(y))) if *func == super::formula::Func::Atan2 => {
                    (constant_like(&y, v)?, Some(Val::T(y)))
                }
                // max(2, a): the tensor goes first; the function is symmetric.
                (Some(Val::Num(v)), Some(Val::T(y))) => (y, Some(Val::Num(v))),
                _ => unreachable!("constant folded"),
            };
            formula_call(*func, x, y)?
        }
    }))
}

/// Evaluates the `Formula` node. The output has the broadcast shape of all inputs, which
/// validation has checked are all used.
fn formula<B: Backend>(expr: &str, inputs: Vec<Dyn<B>>) -> Result<Dyn<B>, ExecError> {
    let e = super::formula::check(expr, inputs.len()).map_err(shape_err)?;
    if inputs.iter().any(|t| !t.is_float()) {
        return Err(dtype_err("Formula expects float tensors"));
    }
    match formula_eval(&e, &inputs)? {
        Val::T(t) => Ok(t),
        Val::Num(_) => Err(shape_err("Formula: the expression uses no input")),
    }
}

/// Executes a node.
fn apply<B: Backend>(
    device: &B::Device,
    node: &Node,
    inputs: Vec<Dyn<B>>,
    seed: u64,
) -> Result<Dyn<B>, ExecError> {
    let attrs = &node.attrs;
    let mut it = inputs.into_iter();
    let mut next = |what: &str| {
        it.next()
            .ok_or_else(|| shape_err(format!("{what}: missing input")))
    };
    Ok(match node.op.as_str() {
        "Add" | "Sub" | "Mul" | "Div" => {
            let a = next(&node.op)?;
            let b = next(&node.op)?;
            binary(&node.op, a, b)?
        }
        "MatMul" => {
            let a = next("MatMul")?;
            let b = next("MatMul")?;
            matmul(a, b)?
        }
        "Formula" => {
            let expr = match attrs.get("expr") {
                Some(Attr::Text(e)) => e.as_str(),
                _ => return Err(shape_err("Formula: missing expr attribute")),
            };
            formula(expr, it.collect())?
        }
        "Relu" => map_float!(next("Relu")?, x => activation::relu(x), "Relu"),
        "Gelu" => match attrs.get("approximate") {
            // `approximate = "tanh"` (PyTorch's `gelu(approximate="tanh")`, Gemma, GPT-2):
            // `x/2·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, the cube by products (a GPU `pow` of a
            // negative base is undefined).
            Some(Attr::Text(a)) if a == "tanh" => {
                map_float!(next("Gelu")?, x => {
                    let cube = x.clone().mul(x.clone()).mul(x.clone());
                    let inner = x.clone().add(cube.mul_scalar(0.044_715)).mul_scalar((std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2) as f32);
                    x.mul(activation::tanh(inner).add_scalar(1.0)).mul_scalar(0.5)
                }, "Gelu")
            }
            _ => map_float!(next("Gelu")?, x => activation::gelu(x), "Gelu"),
        },
        "Sigmoid" => map_float!(next("Sigmoid")?, x => activation::sigmoid(x), "Sigmoid"),
        "Tanh" => map_float!(next("Tanh")?, x => activation::tanh(x), "Tanh"),
        "Rsqrt" => map_float!(next("Rsqrt")?, x => x.sqrt().recip(), "Rsqrt"),
        "ReduceMean" => {
            let x = next("ReduceMean")?;
            let dims = x.dims();
            let axis = norm_axis(attr_int(attrs, "axis", -1), dims.len())?;
            let reduced = mean_axis(x, axis)?;
            if attr_int(attrs, "keepdims", 1) == 0 {
                let mut target = dims;
                target.remove(axis);
                reshape_dyn(reduced, &target)?
            } else {
                reduced
            }
        }
        "Softmax" => softmax_axis(next("Softmax")?, attr_int(attrs, "axis", -1))?,
        "CausalMask" => causal_mask(device, next("CausalMask")?)?,
        "Slice" => {
            let x = next("Slice")?;
            let rank = x.dims().len() as i64;
            let axis = attr_int(attrs, "axis", 0);
            let axis = norm_axis(axis, rank as usize)?;
            let (start, end) = (attr_int(attrs, "start", 0), attr_int(attrs, "end", 0));
            let size = x.dims()[axis] as i64;
            if start < 0 || end <= start || end > size {
                return Err(shape_err(format!(
                    "Slice: range [{start}, {end}) outside axis {axis} of size {size}"
                )));
            }
            let (s, len) = (start as usize, (end - start) as usize);
            map_any!(x, t => t.narrow(axis, s, len), "Slice")
        }
        "LayerNorm" => {
            let x = next("LayerNorm")?;
            let gain = it.next();
            let bias = it.next();
            layer_norm(
                x,
                gain,
                bias,
                attr_int(attrs, "axis", -1),
                attr_float(attrs, "epsilon", 1e-5),
            )?
        }
        "RMSNorm" => {
            let x = next("RMSNorm")?;
            let gain = it.next();
            rms_norm(
                x,
                gain,
                attr_int(attrs, "axis", -1),
                attr_float(attrs, "epsilon", 1e-5),
            )?
        }
        "Conv2d" => {
            let x = as_f4(next("Conv2d")?, "Conv2d")?;
            let w = as_f4(next("Conv2d")?, "Conv2d (kernel)")?;
            let bias = it.next().map(|b| as_f1(b, "Conv2d (bias)")).transpose()?;
            let stride = attr_pair(attrs, "stride", [1, 1])?;
            let padding = attr_pair(attrs, "padding", [0, 0])?;
            let dilation = attr_pair(attrs, "dilation", [1, 1])?;
            let groups = usize::try_from(attr_int(attrs, "groups", 1).max(1))
                .map_err(|_| shape_err("invalid groups"))?;
            Dyn::F4(extra::conv2d(x, w, bias, stride, padding, dilation, groups))
        }
        "MaxPool2d" | "AvgPool2d" => {
            let x = as_f4(next(&node.op)?, &node.op)?;
            let kernel = attr_pair(attrs, "kernel", [2, 2])?;
            let stride = attr_pair(attrs, "stride", kernel)?;
            let padding = attr_pair(attrs, "padding", [0, 0])?;
            if node.op == "MaxPool2d" {
                Dyn::F4(max_pool2d(x, kernel, stride, padding, [1, 1], false))
            } else {
                Dyn::F4(avg_pool2d(x, kernel, stride, padding, true, false))
            }
        }
        "BatchNorm2d" => {
            let x = next("BatchNorm2d")?;
            let gamma = next("BatchNorm2d")?;
            let beta = next("BatchNorm2d")?;
            batch_norm(x, gamma, beta, attr_float(attrs, "epsilon", 1e-5))?
        }
        "Reshape" => {
            let x = next("Reshape")?;
            let spec = attrs
                .get("shape")
                .and_then(Attr::as_ints)
                .ok_or_else(|| shape_err("Reshape: missing shape attribute"))?;
            let target = reshape_target(&x.dims(), spec)?;
            reshape_dyn(x, &target)?
        }
        "Transpose" => {
            let x = next("Transpose")?;
            let perm = attrs
                .get("perm")
                .and_then(Attr::as_ints)
                .ok_or_else(|| shape_err("Transpose: missing perm attribute"))?;
            permute_dyn(x, perm)?
        }
        "Flatten" => {
            let x = next("Flatten")?;
            let dims = x.dims();
            let axis = usize::try_from(attr_int(attrs, "axis", 1))
                .map_err(|_| shape_err("Flatten: negative axis"))?;
            if axis > dims.len() {
                return Err(shape_err("Flatten: axis beyond rank"));
            }
            let head: usize = dims[..axis].iter().product();
            let tail: usize = dims[axis..].iter().product();
            reshape_dyn(x, &[head, tail])?
        }
        "Concat" => {
            let parts: Vec<Dyn<B>> = it.collect();
            let rank = parts.first().map(|p| p.dims().len()).unwrap_or(0);
            let axis = norm_axis(attr_int(attrs, "axis", 0), rank)?;
            concat(parts, axis)?
        }
        "ConvTranspose2d" => {
            let x = next("ConvTranspose2d")?;
            let w = next("ConvTranspose2d")?;
            extra::conv_transpose(x, w, it.next(), attrs)?
        }
        "Pad" => extra::pad(next("Pad")?, attrs)?,
        "Resize" => extra::resize(device, next("Resize")?, attrs)?,
        "Flip" => extra::flip(next("Flip")?, attrs)?,
        "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd" => {
            extra::reduce(&node.op, next(&node.op)?, attrs)?
        }
        "CumSum" => extra::cumsum(next("CumSum")?, attrs)?,
        "IndexSelect" => extra::index_select(device, next("IndexSelect")?, attrs)?,
        "IndexAdd" => {
            let x = next("IndexAdd")?;
            let src = next("IndexAdd")?;
            extra::index_add(device, x, src, attrs)?
        }
        "ArgMax" | "ArgMin" => ints::arg_extremum(next(&node.op)?, attrs, node.op == "ArgMax")?,
        "ArgTopK" => ints::arg_order(device, next("ArgTopK")?, attrs, false)?,
        "ArgSort" => ints::arg_order(device, next("ArgSort")?, attrs, true)?,
        "Cast" => ints::cast(next("Cast")?)?,
        "GatherElements" => {
            let x = next("GatherElements")?;
            let idx = next("GatherElements")?;
            ints::gather_elements(x, idx, attrs)?
        }
        "ScatterElements" => {
            let x = next("ScatterElements")?;
            let idx = next("ScatterElements")?;
            let src = next("ScatterElements")?;
            ints::scatter_elements(x, idx, src, attrs)?
        }
        "Gather" => {
            let table = next("Gather")?;
            let idx = next("Gather")?;
            gather_rows(table, idx)?
        }
        "Dropout" => {
            let x = next("Dropout")?;
            dropout(
                device,
                x,
                attr_float(attrs, "rate", 0.0),
                seed,
                &node.output,
            )?
        }
        other => {
            return Err(shape_err(format!(
                "operator '{other}' unknown to the executor: validation should have rejected it"
            )));
        }
    })
}

/// Loss of the micro-batch from an already computed environment.
fn loss_of<B: Backend>(
    graph: &ModelGraphV1,
    env: &BTreeMap<String, Dyn<B>>,
) -> Result<Tensor<B, 1>, ExecError> {
    let prediction = env
        .get(graph.loss.prediction())
        .cloned()
        .ok_or_else(|| ExecError::Missing(graph.loss.prediction().to_owned()))?;
    let target = env
        .get(&graph.loss.target().name)
        .cloned()
        .ok_or_else(|| ExecError::Missing(graph.loss.target().name.clone()))?;
    Ok(match &graph.loss {
        Loss::MeanSquaredError { .. } => {
            let diff = binary("Sub", prediction, target)?;
            match diff {
                Dyn::F1(d) => d.clone().mul(d).mean(),
                Dyn::F2(d) => d.clone().mul(d).mean(),
                Dyn::F3(d) => d.clone().mul(d).mean(),
                Dyn::F4(d) => d.clone().mul(d).mean(),
                _ => return Err(dtype_err("loss: non-float prediction")),
            }
        }
        Loss::CrossEntropy { .. } => {
            let logits = match prediction {
                Dyn::F2(l) => l,
                other => {
                    return Err(shape_err(format!(
                        "cross-entropy: logits of rank {}, expected 2",
                        other.dims().len()
                    )));
                }
            };
            // One class per example ([N]) or per token ([N, T]): tokens are flattened the same way
            // the logits were.
            let classes = match target {
                Dyn::I1(t) => t,
                Dyn::I2(t) => {
                    let [a, b] = t.dims();
                    t.reshape([a * b])
                }
                _ => {
                    return Err(dtype_err(
                        "cross-entropy: expected an integer target of rank 1 or 2",
                    ));
                }
            };
            let n = classes.dims()[0];
            if logits.dims()[0] != n {
                return Err(shape_err(format!(
                    "cross-entropy: {} logits for {n} targets",
                    logits.dims()[0]
                )));
            }
            let logp = activation::log_softmax(logits, 1);
            let picked = logp.gather(1, classes.reshape([n, 1]));
            picked.mean().neg()
        }
    })
}

/// Bytes of a device tensor (f32 or i64).
fn dyn_bytes<B: Backend>(t: &Dyn<B>) -> u64 {
    let n: u64 = t.dims().iter().map(|d| *d as u64).product();
    let width = if t.is_float() { 4 } else { 8 };
    n * width
}

fn env_bytes<B: Backend>(env: &BTreeMap<String, Dyn<B>>) -> u64 {
    env.values().map(dyn_bytes).sum()
}

/// Forward pass only: the value of a named tensor of the graph, without autodiff tape or loss.
///
/// Used to evaluate a model and to observe its outputs position by position, which is the only
/// way to check that causal attention does not read the future. The loss target is not required:
/// only the graph inputs are.
pub fn forward<B: Backend>(
    device: &B::Device,
    prepared: &Prepared,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
    output: &str,
) -> Result<TensorF32, ExecError> {
    let graph = &prepared.graph;
    let mut env: BTreeMap<String, Dyn<B>> = BTreeMap::new();
    for spec in &graph.inputs {
        let t = inputs
            .get(&spec.name)
            .ok_or_else(|| ExecError::Missing(spec.name.clone()))?;
        t.check_finite()
            .map_err(|e| ExecError::Numeric(format!("input '{}': {e}", spec.name)))?;
        let tensor = if spec.dtype == "i64" {
            int_from_tensor_f32::<B>(
                device,
                &spec.name,
                t,
                prepared.index_limits.get(&spec.name).copied(),
            )?
        } else {
            from_tensor_f32::<B>(device, t)?
        };
        env.insert(spec.name.clone(), tensor);
    }
    for p in &graph.parameters {
        env.insert(p.name.clone(), load_parameter::<B>(device, p, parameters)?);
    }
    for name in &prepared.report.order {
        let node = &graph.nodes[prepared.by_output[name]];
        let args = node
            .inputs
            .iter()
            .map(|t| {
                env.get(t)
                    .cloned()
                    .ok_or_else(|| ExecError::Missing(t.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let started = profiling().then(std::time::Instant::now);
        let out = apply(device, node, args, seed)?;
        if let Some(t0) = started {
            let _ = B::sync(device);
            record_profile(&node.op, t0.elapsed());
        }
        env.insert(node.output.clone(), out);
        if node.output == output {
            break;
        }
    }
    let t = env
        .remove(output)
        .ok_or_else(|| ExecError::Missing(output.to_owned()))?;
    host_of(t)
}

/// [`forward`] on the CPU reference profile.
pub fn forward_cpu(
    prepared: &Prepared,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
    output: &str,
) -> Result<TensorF32, ExecError> {
    forward::<burn::backend::Flex>(
        &burn::backend::flex::FlexDevice,
        prepared,
        parameters,
        inputs,
        seed,
        output,
    )
}

/// `gradient_with` on the CPU reference profile (Flex), for callers that do not depend on Burn
/// (the control plane, the command line).
pub fn gradient_cpu_with(
    prepared: &Prepared,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
    options: ExecOptions,
) -> Result<StepResult, ExecError> {
    gradient_with::<burn::backend::Autodiff<burn::backend::Flex>>(
        &burn::backend::flex::FlexDevice,
        prepared,
        parameters,
        inputs,
        seed,
        options,
    )
}

/// `gradient` on the CPU reference profile (Flex), for callers that do not depend on Burn.
pub fn gradient_cpu(
    prepared: &Prepared,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
) -> Result<StepResult, ExecError> {
    gradient::<burn::backend::Autodiff<burn::backend::Flex>>(
        &burn::backend::flex::FlexDevice,
        prepared,
        parameters,
        inputs,
        seed,
    )
}

/// Checks the datasets of the micro-batch and moves them onto the device, without a host copy.
fn load_datasets_owned<B: Backend>(
    device: &B::Device,
    graph: &ModelGraphV1,
    limits: &BTreeMap<String, u64>,
    mut inputs: BTreeMap<String, TensorF32>,
) -> Result<BTreeMap<String, Dyn<B>>, ExecError> {
    let mut env = BTreeMap::new();
    let mut batch: Option<usize> = None;
    let mut specs: Vec<&super::TensorSpec> = graph.inputs.iter().collect();
    specs.push(graph.loss.target());
    for spec in specs {
        let t = inputs
            .remove(&spec.name)
            .ok_or_else(|| ExecError::Missing(spec.name.clone()))?;
        if t.shape.len() != spec.shape.len() {
            return Err(shape_err(format!(
                "input '{}': rank {} provided, {} declared",
                spec.name,
                t.shape.len(),
                spec.shape.len()
            )));
        }
        for (i, (got, want)) in t.shape.iter().zip(spec.shape.iter()).enumerate() {
            if *want == 0 {
                match batch {
                    None => batch = Some(*got as usize),
                    Some(n) if n as u64 != *got => {
                        return Err(shape_err(format!(
                            "input '{}': batch of {got} examples, {n} expected",
                            spec.name
                        )));
                    }
                    _ => {}
                }
            } else if got != want {
                return Err(shape_err(format!(
                    "input '{}': dimension {i} is {got}, {want} declared",
                    spec.name
                )));
            }
        }
        t.check_finite()
            .map_err(|e| ExecError::Numeric(format!("input '{}': {e}", spec.name)))?;
        let tensor = if spec.dtype == "i64" {
            int_from_tensor_f32::<B>(device, &spec.name, &t, limits.get(&spec.name).copied())?
        } else {
            from_tensor_f32_owned::<B>(device, t)?
        };
        env.insert(spec.name.clone(), tensor);
    }
    Ok(env)
}

/// Like [`load_parameter`], moving the host buffer to the device without a copy.
fn load_parameter_owned<B: Backend>(
    device: &B::Device,
    spec: &ParamSpec,
    t: TensorF32,
) -> Result<Dyn<B>, ExecError> {
    if t.shape != spec.shape {
        return Err(shape_err(format!(
            "parameter '{}': shape {:?} provided, {:?} declared",
            spec.name, t.shape, spec.shape
        )));
    }
    t.check_finite()
        .map_err(|e| ExecError::Numeric(format!("parameter '{}': {e}", spec.name)))?;
    from_tensor_f32_owned::<B>(device, t)
}

/// Loads a parameter onto the device, checking its shape and finiteness.
fn load_parameter<B: Backend>(
    device: &B::Device,
    spec: &ParamSpec,
    parameters: &BTreeMap<String, TensorF32>,
) -> Result<Dyn<B>, ExecError> {
    let t = parameters
        .get(&spec.name)
        .ok_or_else(|| ExecError::Missing(spec.name.clone()))?;
    if t.shape != spec.shape {
        return Err(shape_err(format!(
            "parameter '{}': shape {:?} provided, {:?} declared",
            spec.name, t.shape, spec.shape
        )));
    }
    t.check_finite()
        .map_err(|e| ExecError::Numeric(format!("parameter '{}': {e}", spec.name)))?;
    from_tensor_f32::<B>(device, t)
}

fn require_grad<B: AutodiffBackend>(t: Dyn<B>) -> Dyn<B> {
    match t {
        Dyn::F1(x) => Dyn::F1(x.require_grad()),
        Dyn::F2(x) => Dyn::F2(x.require_grad()),
        Dyn::F3(x) => Dyn::F3(x.require_grad()),
        Dyn::F4(x) => Dyn::F4(x.require_grad()),
        other => other,
    }
}

fn to_autodiff<B: AutodiffBackend>(t: Dyn<B::InnerBackend>) -> Dyn<B> {
    match t {
        Dyn::F1(x) => Dyn::F1(Tensor::from_inner(x)),
        Dyn::F2(x) => Dyn::F2(Tensor::from_inner(x)),
        Dyn::F3(x) => Dyn::F3(Tensor::from_inner(x)),
        Dyn::F4(x) => Dyn::F4(Tensor::from_inner(x)),
        Dyn::I1(x) => Dyn::I1(Tensor::from_inner(x)),
        Dyn::I2(x) => Dyn::I2(Tensor::from_inner(x)),
        Dyn::I3(x) => Dyn::I3(Tensor::from_inner(x)),
        Dyn::I4(x) => Dyn::I4(Tensor::from_inner(x)),
    }
}

/// Gradient of a tracked tensor, returned as a `Dyn` on the inner backend.
fn grad_of<B: AutodiffBackend>(t: &Dyn<B>, grads: &B::Gradients) -> Option<Dyn<B::InnerBackend>> {
    match t {
        Dyn::F1(x) => x.grad(grads).map(Dyn::F1),
        Dyn::F2(x) => x.grad(grads).map(Dyn::F2),
        Dyn::F3(x) => x.grad(grads).map(Dyn::F3),
        Dyn::F4(x) => x.grad(grads).map(Dyn::F4),
        _ => None,
    }
}

fn add_inner<B: Backend>(a: Dyn<B>, b: Dyn<B>) -> Result<Dyn<B>, ExecError> {
    binary("Add", a, b)
}

fn host_of<B: Backend>(t: Dyn<B>) -> Result<TensorF32, ExecError> {
    match t {
        Dyn::F1(x) => to_tensor_f32(x),
        Dyn::F2(x) => to_tensor_f32(x),
        Dyn::F3(x) => to_tensor_f32(x),
        Dyn::F4(x) => to_tensor_f32(x),
        // An integer (an `ArgMax` read by `eval-model`) is exact in f32: values below 2²⁴.
        Dyn::I1(x) => to_tensor_f32(x.float()),
        Dyn::I2(x) => to_tensor_f32(x.float()),
        Dyn::I3(x) => to_tensor_f32(x.float()),
        Dyn::I4(x) => to_tensor_f32(x.float()),
    }
}

/// Computes the loss of a micro-batch and the gradient of each trained parameter.
///
/// `parameters` and `inputs` are indexed by name; the loss target is looked up in `inputs` under
/// the name declared in the graph. `seed` fixes the Dropout masks.
pub fn gradient<B: AutodiffBackend>(
    device: &B::Device,
    prepared: &Prepared,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
) -> Result<StepResult, ExecError> {
    gradient_with::<B>(
        device,
        prepared,
        parameters,
        inputs,
        seed,
        ExecOptions::default(),
    )
}

/// Like [`gradient`], with the memory levers of ADR-0019.
///
/// With `segments > 1`, the forward pass runs **without an autodiff tape** and keeps only the
/// segment boundaries; each segment is then recomputed during the backward pass, seeded by the
/// gradient of its outputs. The result is the same gradient, for less memory and roughly a third
/// more compute.
///
/// With `offload_parameters`, the parameters of a segment are loaded onto the device only when
/// that segment runs: the reference value stays in host memory.
pub fn gradient_with<B: AutodiffBackend>(
    device: &B::Device,
    prepared: &Prepared,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
    options: ExecOptions,
) -> Result<StepResult, ExecError> {
    gradient_with_bounds::<B>(device, prepared, parameters, inputs, seed, options, None)
}

/// Like [`gradient_with`], with imposed segment boundaries (`0 = b₀ < … < b_K = n`): these are
/// the stage boundaries, which makes this execution the reference for distributed training
/// (D2).
pub fn gradient_with_bounds<B: AutodiffBackend>(
    device: &B::Device,
    prepared: &Prepared,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
    options: ExecOptions,
    options_bounds: Option<&[u32]>,
) -> Result<StepResult, ExecError> {
    gradient_owned::<B>(
        device,
        prepared,
        parameters.clone(),
        inputs.clone(),
        seed,
        options,
        options_bounds,
    )
}

/// Like [`gradient_with`], taking the parameters and the micro-batch over: each tensor moves to
/// the device without a host copy (the executor and the validator have no further use for them).
pub fn gradient_owned_with<B: AutodiffBackend>(
    device: &B::Device,
    prepared: &Prepared,
    parameters: BTreeMap<String, TensorF32>,
    inputs: BTreeMap<String, TensorF32>,
    seed: u64,
    options: ExecOptions,
) -> Result<StepResult, ExecError> {
    gradient_owned::<B>(device, prepared, parameters, inputs, seed, options, None)
}

fn gradient_owned<B: AutodiffBackend>(
    device: &B::Device,
    prepared: &Prepared,
    mut parameters: BTreeMap<String, TensorF32>,
    inputs: BTreeMap<String, TensorF32>,
    seed: u64,
    options: ExecOptions,
    options_bounds: Option<&[u32]>,
) -> Result<StepResult, ExecError> {
    if let Some(b) = options_bounds
        && (b.len() < 2
            || b[0] != 0
            || *b.last().expect("non-empty") as usize != prepared.report.order.len()
            || b.windows(2).any(|w| w[0] >= w[1]))
    {
        return Err(shape_err(format!("invalid segment boundaries {b:?}")));
    }
    if options.segments == 0 {
        return Err(shape_err("segments must be at least 1"));
    }
    let graph = &prepared.graph;
    let order = &prepared.report.order;
    let n = order.len();
    let segments = options.segments.min(n).max(1);
    let node_of = |name: &str| -> &Node { &graph.nodes[prepared.by_output[name]] };
    let spec_of =
        |name: &str| -> Option<&ParamSpec> { graph.parameters.iter().find(|p| p.name == name) };

    // Boundaries: `segments` contiguous slices of the execution order, balanced by node count, or
    // the ones imposed by the caller.
    let bounds: Vec<(usize, usize)> = match &options_bounds {
        Some(b) => b
            .windows(2)
            .map(|w| (w[0] as usize, w[1] as usize))
            .collect(),
        None => (0..segments)
            .map(|s| (n * s / segments, n * (s + 1) / segments))
            .filter(|(a, b)| a < b)
            .collect(),
    };

    // Last consumer of each tensor, to free what is no longer needed.
    let mut last_use: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, name) in order.iter().enumerate() {
        for input in &node_of(name).inputs {
            last_use.insert(input.as_str(), i);
        }
    }
    // A tensor consumed by the loss lives until the end.
    last_use.insert(graph.loss.prediction(), n);
    last_use.insert(&graph.loss.target().name, n);

    let datasets =
        load_datasets_owned::<B::InnerBackend>(device, graph, &prepared.index_limits, inputs)?;
    let dataset_bytes = env_bytes(&datasets);

    // Parameters: loaded once, or segment by segment with parameter offloading.
    let mut resident: BTreeMap<String, Dyn<B::InnerBackend>> = BTreeMap::new();
    if !options.offload_parameters {
        for p in &graph.parameters {
            let t = parameters
                .remove(&p.name)
                .ok_or_else(|| ExecError::Missing(p.name.clone()))?;
            resident.insert(
                p.name.clone(),
                load_parameter_owned::<B::InnerBackend>(device, p, t)?,
            );
        }
    } else {
        // Shapes and finiteness are still checked, before any computation.
        for p in &graph.parameters {
            let t = parameters
                .get(&p.name)
                .ok_or_else(|| ExecError::Missing(p.name.clone()))?;
            if t.shape != p.shape {
                return Err(shape_err(format!(
                    "parameter '{}': shape {:?} provided, {:?} declared",
                    p.name, t.shape, p.shape
                )));
            }
            t.check_finite()
                .map_err(|e| ExecError::Numeric(format!("parameter '{}': {e}", p.name)))?;
        }
    }

    // Parameters used by a segment, in order.
    let segment_params = |range: (usize, usize)| -> Vec<&ParamSpec> {
        let mut out: Vec<&ParamSpec> = Vec::new();
        for name in &order[range.0..range.1] {
            for input in &node_of(name).inputs {
                if let Some(p) = spec_of(input)
                    && !out.iter().any(|q| q.name == p.name)
                {
                    out.push(p);
                }
            }
        }
        out
    };

    let mut peak_activation_bytes = 0u64;
    let mut peak_parameter_bytes = if options.offload_parameters {
        0
    } else {
        env_bytes(&resident)
    };

    // ---- Forward pass on the inner backend, without an autodiff tape.
    //
    // Only the tensors still needed survive, and we record what enters each segment: that is
    // everything the backward pass will need to recover.
    let mut live: BTreeMap<String, Dyn<B::InnerBackend>> = datasets.clone();
    let mut saved: BTreeMap<String, Dyn<B::InnerBackend>> = BTreeMap::new();
    let mut segment_inputs: Vec<Vec<String>> = Vec::with_capacity(bounds.len());
    for (s, range) in bounds.iter().enumerate() {
        // What this segment consumes that comes from before it (excluding parameters).
        let mut needed: Vec<String> = Vec::new();
        for name in &order[range.0..range.1] {
            for input in &node_of(name).inputs {
                if spec_of(input).is_some() || needed.iter().any(|x| x == input) {
                    continue;
                }
                let produced_here = order[range.0..range.1].iter().any(|o| o == input);
                if !produced_here {
                    needed.push(input.clone());
                }
            }
        }
        for t in &needed {
            let v = live
                .get(t)
                .cloned()
                .ok_or_else(|| ExecError::Missing(t.clone()))?;
            // A boundary is made contiguous, as when it crosses the network between two stages:
            // a transposed view would make the next matrix product accumulate in a different
            // order, and segmented execution would stop being the exact reference for distributed
            // execution (D2). A dataset has just been loaded from the host: it already is.
            let v = if v.is_float() && !datasets.contains_key(t) {
                from_tensor_f32::<B::InnerBackend>(device, &host_of(v)?)?
            } else {
                v
            };
            saved.insert(t.clone(), v);
        }
        segment_inputs.push(needed);
        // The last segment is computed once, with the tape, below: its outputs feed no later
        // segment. Computed here too, it doubled the forward pass of every unsegmented step.
        if s + 1 == bounds.len() {
            break;
        }

        for (i, name) in order.iter().enumerate().take(range.1).skip(range.0) {
            let node = node_of(name);
            let args = node
                .inputs
                .iter()
                .map(|t| {
                    live.get(t)
                        .cloned()
                        .or_else(|| resident.get(t).cloned())
                        .map(Ok)
                        .unwrap_or_else(|| {
                            // Parameter offloading: the parameter is loaded for the duration of this node.
                            let p = spec_of(t).ok_or_else(|| ExecError::Missing(t.clone()))?;
                            load_parameter::<B::InnerBackend>(device, p, &parameters)
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let out = apply(device, node, args, seed)?;
            live.insert(node.output.clone(), out);
            // Free what nobody needs any more.
            live.retain(|k, _| {
                datasets.contains_key(k) || last_use.get(k.as_str()).is_some_and(|l| *l > i)
            });
            peak_activation_bytes = peak_activation_bytes
                .max(env_bytes(&live) + env_bytes(&saved) - dataset_bytes.min(env_bytes(&live)));
        }
    }

    // ---- Loss, on the last segment rebuilt with autodiff.
    let mut param_grads: BTreeMap<String, Dyn<B::InnerBackend>> = BTreeMap::new();
    let mut boundary_grads: BTreeMap<String, Dyn<B::InnerBackend>> = BTreeMap::new();
    let mut loss_value = f32::NAN;

    for (s, range) in bounds.iter().enumerate().rev() {
        let mut env: BTreeMap<String, Dyn<B>> = BTreeMap::new();
        let mut tracked_inputs: Vec<(String, Dyn<B>)> = Vec::new();
        let mut tracked_params: Vec<(String, Dyn<B>)> = Vec::new();

        for name in &segment_inputs[s] {
            let inner = saved
                .get(name)
                .cloned()
                .ok_or_else(|| ExecError::Missing(name.clone()))?;
            let t = to_autodiff::<B>(inner);
            if datasets.contains_key(name) {
                env.insert(name.clone(), t);
            } else {
                let t = require_grad::<B>(t);
                tracked_inputs.push((name.clone(), t.clone()));
                env.insert(name.clone(), t);
            }
        }
        // The loss needs the target, which no node consumes.
        if s + 1 == bounds.len() {
            let name = &graph.loss.target().name;
            let inner = datasets
                .get(name)
                .cloned()
                .ok_or_else(|| ExecError::Missing(name.clone()))?;
            env.insert(name.clone(), to_autodiff::<B>(inner));
        }
        let mut segment_param_bytes = 0u64;
        for p in segment_params(*range) {
            let inner = match resident.get(&p.name) {
                Some(t) => t.clone(),
                None => load_parameter::<B::InnerBackend>(device, p, &parameters)?,
            };
            segment_param_bytes += dyn_bytes(&inner);
            let t = to_autodiff::<B>(inner);
            let t = if p.trainable {
                let t = require_grad::<B>(t);
                tracked_params.push((p.name.clone(), t.clone()));
                t
            } else {
                t
            };
            env.insert(p.name.clone(), t);
        }
        if options.offload_parameters {
            peak_parameter_bytes = peak_parameter_bytes.max(segment_param_bytes);
        }

        for name in &order[range.0..range.1] {
            let node = node_of(name);
            let args = node
                .inputs
                .iter()
                .map(|t| {
                    env.get(t)
                        .cloned()
                        .ok_or_else(|| ExecError::Missing(t.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let started = profiling().then(std::time::Instant::now);
            let out = apply(device, node, args, seed)?;
            if let Some(t0) = started {
                let _ = B::sync(device);
                record_profile(&node.op, t0.elapsed());
            }
            env.insert(node.output.clone(), out);
        }
        peak_activation_bytes = peak_activation_bytes.max(env_bytes(&env) + env_bytes(&saved));

        // Seed: the loss for the last segment, otherwise the sum of the outputs weighted by the
        // gradient that has already reached them. `d((y·g).sum())/dy = g`, so this is exact.
        let seed_scalar: Tensor<B, 1> = if s + 1 == bounds.len() {
            let l = loss_of(graph, &env)?;
            loss_value = l.clone().into_scalar().elem();
            if !loss_value.is_finite() {
                return Err(ExecError::Numeric(format!("non-finite loss: {loss_value}")));
            }
            l
        } else {
            let mut acc: Option<Tensor<B, 1>> = None;
            for name in &order[range.0..range.1] {
                let Some(g) = boundary_grads.get(name) else {
                    continue;
                };
                let y = env
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ExecError::Missing(name.clone()))?;
                let product = binary("Mul", y, to_autodiff::<B>(g.clone()))?;
                let sum: Tensor<B, 1> = match product {
                    Dyn::F1(x) => x.sum(),
                    Dyn::F2(x) => x.sum(),
                    Dyn::F3(x) => x.sum(),
                    Dyn::F4(x) => x.sum(),
                    _ => return Err(dtype_err("non-float seed")),
                };
                acc = Some(match acc {
                    Some(a) => a.add(sum),
                    None => sum,
                });
            }
            match acc {
                Some(a) => a,
                // No tensor of this segment influences the loss: nothing to backpropagate.
                None => continue,
            }
        };

        let grads = seed_scalar.backward();
        for (name, t) in &tracked_params {
            if let Some(g) = grad_of::<B>(t, &grads) {
                let merged = match param_grads.remove(name) {
                    Some(prev) => add_inner(prev, g)?,
                    None => g,
                };
                param_grads.insert(name.clone(), merged);
            }
        }
        for (name, t) in &tracked_inputs {
            if let Some(g) = grad_of::<B>(t, &grads) {
                let merged = match boundary_grads.remove(name) {
                    Some(prev) => add_inner(prev, g)?,
                    None => g,
                };
                boundary_grads.insert(name.clone(), merged);
            }
        }
    }

    // ---- Gradients returned to the host, for trained parameters only.
    let mut gradients = BTreeMap::new();
    for p in &graph.parameters {
        if !p.trainable {
            continue;
        }
        let g = match param_grads.remove(&p.name) {
            Some(g) => host_of(g)?,
            // A trained parameter that does not influence the loss has a zero gradient, not a missing one.
            None => {
                let count: usize = p.shape.iter().product::<u64>() as usize;
                TensorF32::new(p.shape.clone(), vec![0.0; count])
                    .map_err(|e| ExecError::Shape(e.to_string()))?
            }
        };
        g.check_finite()
            .map_err(|e| ExecError::Numeric(format!("gradient of '{}': {e}", p.name)))?;
        gradients.insert(p.name.clone(), g);
    }
    Ok(StepResult {
        loss: loss_value,
        gradients,
        peak_activation_bytes,
        peak_parameter_bytes,
    })
}

// ---------------------------------------------------------------------------------------------
// Stages (gg-server/docs/ROADMAP.md, P11 D2): a stage runs on its own, on a machine that holds only its
// parameters. The forward pass returns its outputs; the backward pass recomputes the stage from
// its inputs, then backpropagates the gradients received from the following stages (or the loss,
// for the last one). This is the segmented execution of `gradient_with`, spread across machines.
// ---------------------------------------------------------------------------------------------

/// What the backward pass of a stage returns.
#[derive(Clone, Debug)]
pub struct StageBackward {
    /// Loss of the micro-batch, for the last stage.
    pub loss: Option<f32>,
    /// Gradients of the trained parameters of the stage.
    pub gradients: BTreeMap<String, TensorF32>,
    /// Gradients with respect to the stage inputs that are not datasets.
    pub input_gradients: BTreeMap<String, TensorF32>,
    /// Bytes of tensors held at the peak, measured.
    pub peak_bytes: u64,
}

fn dataset_spec<'a>(graph: &'a ModelGraphV1, name: &str) -> Option<&'a super::TensorSpec> {
    graph
        .inputs
        .iter()
        .chain(std::iter::once(graph.loss.target()))
        .find(|s| s.name == name)
}

/// Loads the inputs of a stage: datasets with their own type, activations as floats.
fn stage_inputs<B: Backend>(
    device: &B::Device,
    graph: &ModelGraphV1,
    limits: &BTreeMap<String, u64>,
    stage: &super::stages::StageSpec,
    inputs: &BTreeMap<String, TensorF32>,
) -> Result<BTreeMap<String, Dyn<B>>, ExecError> {
    let mut env = BTreeMap::new();
    for name in &stage.inputs {
        let t = inputs
            .get(name)
            .ok_or_else(|| ExecError::Missing(name.clone()))?;
        t.check_finite()
            .map_err(|e| ExecError::Numeric(format!("input '{name}': {e}")))?;
        let v = match dataset_spec(graph, name) {
            Some(spec) if spec.dtype == "i64" => {
                int_from_tensor_f32::<B>(device, name, t, limits.get(name).copied())?
            }
            _ => from_tensor_f32::<B>(device, t)?,
        };
        env.insert(name.clone(), v);
    }
    Ok(env)
}

fn stage_param_specs<'a>(
    graph: &'a ModelGraphV1,
    stage: &super::stages::StageSpec,
) -> Result<Vec<&'a ParamSpec>, ExecError> {
    stage
        .params
        .iter()
        .map(|n| {
            graph
                .parameters
                .iter()
                .find(|p| &p.name == n)
                .ok_or_else(|| ExecError::Missing(n.clone()))
        })
        .collect()
}

/// Forward pass of a stage: its outputs, without an autodiff tape.
pub fn stage_forward<B: Backend>(
    device: &B::Device,
    prepared: &Prepared,
    stage: &super::stages::StageSpec,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
) -> Result<(BTreeMap<String, TensorF32>, u64), ExecError> {
    let graph = &prepared.graph;
    let order = &prepared.report.order;
    let mut env = stage_inputs::<B>(device, graph, &prepared.index_limits, stage, inputs)?;
    for p in stage_param_specs(graph, stage)? {
        env.insert(p.name.clone(), load_parameter::<B>(device, p, parameters)?);
    }
    // Last use within the stage; outputs live until the end.
    let mut last_use: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, name) in order.iter().enumerate().take(stage.end).skip(stage.first) {
        for input in &graph.nodes[prepared.by_output[name]].inputs {
            last_use.insert(input.as_str(), i);
        }
    }
    for o in &stage.outputs {
        last_use.insert(o.as_str(), usize::MAX);
    }
    let mut peak = env_bytes(&env);
    for (i, name) in order.iter().enumerate().take(stage.end).skip(stage.first) {
        let node = &graph.nodes[prepared.by_output[name]];
        let args = node
            .inputs
            .iter()
            .map(|t| {
                env.get(t)
                    .cloned()
                    .ok_or_else(|| ExecError::Missing(t.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out = apply(device, node, args, seed)?;
        env.insert(node.output.clone(), out);
        peak = peak.max(env_bytes(&env));
        env.retain(|k, _| last_use.get(k.as_str()).is_some_and(|l| *l > i));
    }
    let mut outputs = BTreeMap::new();
    for o in &stage.outputs {
        let t = env.remove(o).ok_or_else(|| ExecError::Missing(o.clone()))?;
        let host = host_of(t)?;
        host.check_finite()
            .map_err(|e| ExecError::Numeric(format!("output '{o}': {e}")))?;
        outputs.insert(o.clone(), host);
    }
    Ok((outputs, peak))
}

/// Backward pass of a stage. `output_gradients` gives, for each output of the stage, the gradients
/// received from its consumers, **summed in the given order** (that of decreasing stages, like
/// segmented execution, for a byte-identical result).
pub fn stage_backward<B: AutodiffBackend>(
    device: &B::Device,
    prepared: &Prepared,
    stage: &super::stages::StageSpec,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    output_gradients: &BTreeMap<String, Vec<TensorF32>>,
    seed: u64,
) -> Result<StageBackward, ExecError> {
    let graph = &prepared.graph;
    let order = &prepared.report.order;
    let mut inner =
        stage_inputs::<B::InnerBackend>(device, graph, &prepared.index_limits, stage, inputs)?;
    let mut env: BTreeMap<String, Dyn<B>> = BTreeMap::new();
    let mut tracked_inputs: Vec<(String, Dyn<B>)> = Vec::new();
    // In order of first use, like segmented execution: autodiff accumulates contributions in the
    // order in which tensors were registered, and that is what makes the gradient
    // byte-identical.
    for name in &stage.inputs {
        let name = name.clone();
        let v = inner
            .remove(&name)
            .ok_or_else(|| ExecError::Missing(name.clone()))?;
        let t = to_autodiff::<B>(v);
        if dataset_spec(graph, &name).is_some() {
            env.insert(name, t);
        } else {
            let t = require_grad::<B>(t);
            tracked_inputs.push((name.clone(), t.clone()));
            env.insert(name, t);
        }
    }
    let mut tracked_params: Vec<(String, Dyn<B>)> = Vec::new();
    for p in stage_param_specs(graph, stage)? {
        let t = to_autodiff::<B>(load_parameter::<B::InnerBackend>(device, p, parameters)?);
        let t = if p.trainable {
            let t = require_grad::<B>(t);
            tracked_params.push((p.name.clone(), t.clone()));
            t
        } else {
            t
        };
        env.insert(p.name.clone(), t);
    }
    for name in order.iter().take(stage.end).skip(stage.first) {
        let node = &graph.nodes[prepared.by_output[name]];
        let args = node
            .inputs
            .iter()
            .map(|t| {
                env.get(t)
                    .cloned()
                    .ok_or_else(|| ExecError::Missing(t.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out = apply(device, node, args, seed)?;
        env.insert(node.output.clone(), out);
    }
    let mut peak_bytes = env_bytes(&env);
    let mut loss = None;
    let seed_scalar: Option<Tensor<B, 1>> = if stage.last {
        let l = loss_of(graph, &env)?;
        let v: f32 = l.clone().into_scalar().elem();
        if !v.is_finite() {
            return Err(ExecError::Numeric(format!("non-finite loss: {v}")));
        }
        loss = Some(v);
        Some(l)
    } else {
        let mut acc: Option<Tensor<B, 1>> = None;
        for name in &stage.outputs {
            let Some(list) = output_gradients.get(name).filter(|l| !l.is_empty()) else {
                continue;
            };
            let mut g = from_tensor_f32::<B::InnerBackend>(device, &list[0])?;
            for more in &list[1..] {
                g = add_inner(g, from_tensor_f32::<B::InnerBackend>(device, more)?)?;
            }
            peak_bytes += dyn_bytes(&g);
            let y = env
                .get(name)
                .cloned()
                .ok_or_else(|| ExecError::Missing(name.clone()))?;
            if y.dims() != g.dims() {
                return Err(shape_err(format!(
                    "gradient received for '{name}' has shape {:?}, the output is {:?}",
                    g.dims(),
                    y.dims()
                )));
            }
            let product = binary("Mul", y, to_autodiff::<B>(g))?;
            let sum: Tensor<B, 1> = match product {
                Dyn::F1(x) => x.sum(),
                Dyn::F2(x) => x.sum(),
                Dyn::F3(x) => x.sum(),
                Dyn::F4(x) => x.sum(),
                _ => return Err(dtype_err("non-float seed")),
            };
            acc = Some(match acc {
                Some(a) => a.add(sum),
                None => sum,
            });
        }
        acc
    };
    let mut gradients = BTreeMap::new();
    let mut input_gradients = BTreeMap::new();
    let grads = seed_scalar.map(|s| s.backward());
    for (name, t) in &tracked_params {
        let g = match grads.as_ref().and_then(|gs| grad_of::<B>(t, gs)) {
            Some(g) => host_of(g)?,
            None => zeros_of(&t.dims())?,
        };
        g.check_finite()
            .map_err(|e| ExecError::Numeric(format!("gradient of '{name}': {e}")))?;
        gradients.insert(name.clone(), g);
    }
    for (name, t) in &tracked_inputs {
        let g = match grads.as_ref().and_then(|gs| grad_of::<B>(t, gs)) {
            Some(g) => host_of(g)?,
            None => zeros_of(&t.dims())?,
        };
        g.check_finite()
            .map_err(|e| ExecError::Numeric(format!("gradient of input '{name}': {e}")))?;
        input_gradients.insert(name.clone(), g);
    }
    Ok(StageBackward {
        loss,
        gradients,
        input_gradients,
        peak_bytes,
    })
}

fn zeros_of(dims: &[usize]) -> Result<TensorF32, ExecError> {
    let shape: Vec<u64> = dims.iter().map(|d| *d as u64).collect();
    let n: usize = dims.iter().product();
    TensorF32::new(shape, vec![0.0; n]).map_err(|e| ExecError::Shape(e.to_string()))
}

/// [`stage_forward`] on the CPU reference profile.
pub fn stage_forward_cpu(
    prepared: &Prepared,
    stage: &super::stages::StageSpec,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
) -> Result<(BTreeMap<String, TensorF32>, u64), ExecError> {
    stage_forward::<burn::backend::Flex>(
        &burn::backend::flex::FlexDevice,
        prepared,
        stage,
        parameters,
        inputs,
        seed,
    )
}

/// [`stage_backward`] on the CPU reference profile.
pub fn stage_backward_cpu(
    prepared: &Prepared,
    stage: &super::stages::StageSpec,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    output_gradients: &BTreeMap<String, Vec<TensorF32>>,
    seed: u64,
) -> Result<StageBackward, ExecError> {
    stage_backward::<burn::backend::Autodiff<burn::backend::Flex>>(
        &burn::backend::flex::FlexDevice,
        prepared,
        stage,
        parameters,
        inputs,
        output_gradients,
        seed,
    )
}

// ---------------------------------------------------------------------------------------------
// Per-operator profile (D5): `GG_PROFILE=1` synchronizes the device after each node of the
// forward pass and accumulates time per operator; `take_profile` returns it. Used to find where
// the time goes before touching kernels; when disabled, it costs only one check per node.
// ---------------------------------------------------------------------------------------------

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GG_PROFILE").is_ok_and(|v| v == "1"))
}

static PROFILE: std::sync::Mutex<BTreeMap<String, (u32, std::time::Duration)>> =
    std::sync::Mutex::new(BTreeMap::new());

fn record_profile(op: &str, d: std::time::Duration) {
    if let Ok(mut p) = PROFILE.lock() {
        let e = p.entry(op.to_owned()).or_default();
        e.0 += 1;
        e.1 += d;
    }
}

/// Accumulated time per operator since the last call (empty without `GG_PROFILE=1`).
pub fn take_profile() -> BTreeMap<String, (u32, std::time::Duration)> {
    PROFILE
        .lock()
        .map(|mut p| std::mem::take(&mut *p))
        .unwrap_or_default()
}

// Operators of `gg-ops/7` (ADR-0027) and `gg-ops/8` (ADR-0029). Declared after the macros above,
// which they use.
mod extra;
mod ints;
#[cfg(feature = "cuda")]
pub mod torch;

pub use ints::line_order;
