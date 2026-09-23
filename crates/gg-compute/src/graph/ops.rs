//! Operator profile (ADR-0016).
//!
//! The platform does not "support ONNX": it supports a **published, versioned profile**. Each
//! operator enters it with its shape rule, its cost and what it requires of its attributes. A graph
//! that falls outside the profile is rejected at submission, naming what is missing.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Version of the published profile: its operators **and** its table generators. `gg-ops/2` adds
/// `CausalMask` (ADR-0020), `gg-ops/3` adds `Slice` and the position tables (ADR-0021),
/// `gg-ops/4` rotary tables with explicit frequencies, the form of any RoPE scaling (ADR-0022),
/// `gg-ops/5` `ReduceMean` and `Rsqrt`, used by the hand-written normalizations of Hugging Face
/// models (ADR-0023). Each version is a superset of the previous one: any description valid under
/// an earlier version is still valid, unless it carried an unknown field, rejected since
/// `gg-ops/4`. `gg-ops/6` adds `Formula`, an element-wise expression written in the description,
/// and description functions (ADR-0025). `gg-ops/7` adds the operators that real PyTorch models
/// were measured to lack (ADR-0027): transposed convolution, padding, resizing, flips,
/// reductions other than the mean, the cumulative sum, selection and accumulation by constant
/// indices, and fifteen functions of `Formula` (inverse trigonometric and hyperbolic functions,
/// `atan2`, rounding, `sign`). `gg-ops/8` adds integer tensors computed inside the graph
/// (ADR-0029): `ArgMax`, `ArgMin`, `ArgTopK`, `ArgSort`, `Cast`, `GatherElements`,
/// `ScatterElements`, and `Gather` over tables of rank 3 and 4.
pub const PROFILE_VERSION: &str = "gg-ops/8";

/// Largest axis an index is computed over or into (ADR-0029): every integer of the graph is in
/// `[0, 2²⁴]`, exact in i32 (the GPU's integer type) and in the f32 of `GGT1` files.
pub const MAX_INDEX_AXIS: u64 = 1 << 24;

/// Type of a tensor between nodes (ADR-0029). `i64` is the type of index inputs and of the
/// integers computed by `ArgMax`, `ArgMin`, `ArgTopK` and `ArgSort`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// Integer (index); held in the backend's integer type, values in `[0, 2²⁴]`.
    I64,
}

impl Dtype {
    /// The name used in descriptions.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::I64 => "i64",
        }
    }
}

/// Type of a node's output from the types of its inputs, or the error naming the input (by
/// position) whose type the operator does not accept.
pub fn infer_dtype(op: &str, inputs: &[Dtype]) -> Result<Dtype, (usize, OpError)> {
    use Dtype::{F32, I64};
    let expect = |want: &[Dtype]| -> Result<(), (usize, OpError)> {
        for (k, (got, w)) in inputs.iter().zip(want).enumerate() {
            if got != w {
                return Err((
                    k,
                    err(format!(
                        "{op} expects {} for input {k}, got {}",
                        w.name(),
                        got.name()
                    )),
                ));
            }
        }
        Ok(())
    };
    match op {
        "ArgMax" | "ArgMin" | "ArgTopK" | "ArgSort" => expect(&[F32]).map(|_| I64),
        "Cast" => expect(&[I64]).map(|_| F32),
        "Gather" | "GatherElements" => expect(&[F32, I64]).map(|_| F32),
        "ScatterElements" => expect(&[F32, I64, F32]).map(|_| F32),
        // Shape operators keep the type; a concatenation needs one type.
        "Reshape" | "Transpose" | "Slice" | "Flatten" | "Concat" => {
            let t = inputs[0];
            if let Some(k) = inputs.iter().position(|d| *d != t) {
                return Err((
                    k,
                    err(format!(
                        "{op}: inputs of different types ({} and {})",
                        t.name(),
                        inputs[k].name()
                    )),
                ));
            }
            Ok(t)
        }
        _ => {
            if let Some(k) = inputs.iter().position(|d| *d != F32) {
                return Err((
                    k,
                    err(format!(
                        "{op} computes on f32; input {k} is an integer tensor (i64): only Gather, \
GatherElements, ScatterElements, Cast and the shape operators take integers (ADR-0029)"
                    )),
                ));
            }
            Ok(F32)
        }
    }
}

/// Maximum number of constant indices carried by an `IndexSelect` or `IndexAdd` node. The indices
/// are written in the description: a relative position table of 1024 × 1024 fits.
pub const MAX_INDICES: usize = 1 << 20;

/// Maximum size of a dimension written in an attribute (`Resize` size, `Pad` padding).
pub const MAX_ATTR_DIM: i64 = 1 << 20;

/// Value written at future positions by `CausalMask`. Finite rather than `-inf`: after a `Softmax`
/// the weight is exactly zero in both cases (`exp(-1e9 - m)` underflows to zero in f32), but a
/// finite value never produces a `NaN` if an entire row was masked or if the mask was followed by
/// something other than a `Softmax`.
pub const CAUSAL_FILL: f32 = -1.0e9;

/// Attribute value of a node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Attr {
    /// Integer.
    Int(i64),
    /// Float.
    Float(f64),
    /// List of integers (shapes, permutations, convolution strides).
    Ints(Vec<i64>),
    /// Short text.
    Text(String),
}

impl Attr {
    /// Integer, if it is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attr::Int(v) => Some(*v),
            _ => None,
        }
    }
    /// Float, if it is one (an integer is accepted).
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Attr::Float(v) => Some(*v),
            Attr::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
    /// List of integers, if it is one.
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            Attr::Ints(v) => Some(v),
            _ => None,
        }
    }
}

/// What an operator promises, beyond its computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    /// Two runs on the reference profile produce the same bytes.
    Bitwise,
    /// Reproducible only within a tolerance (summation orders).
    Tolerance,
    /// Depends on randomness, reproducible only through the protocol's DRBG.
    Seeded,
}

/// Published description of an operator.
#[derive(Clone, Debug, Serialize)]
pub struct OpSpec {
    /// Name in the profile.
    pub name: &'static str,
    /// ONNX equivalent, when one exists, with the targeted opset version.
    pub onnx: Option<&'static str>,
    /// Expected number of inputs (min, max).
    pub arity: (usize, usize),
    /// Recognized attributes. Any other attribute is rejected rather than ignored.
    pub attributes: &'static [&'static str],
    /// Required attributes.
    pub required: &'static [&'static str],
    /// What the operator guarantees from one run to the next.
    pub determinism: Determinism,
    /// One sentence on what it does.
    pub summary: &'static str,
}

/// Error of an operator on a given node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpError(pub String);

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Shape of a tensor; `None` marks the batch dimension, unknown in the description.
pub type Shape = Vec<Option<u64>>;

fn dims(shape: &Shape) -> Vec<String> {
    shape
        .iter()
        .map(|d| d.map(|v| v.to_string()).unwrap_or_else(|| "N".into()))
        .collect()
}

/// Human-readable shape, for error messages.
pub fn show(shape: &Shape) -> String {
    format!("[{}]", dims(shape).join(", "))
}

fn err(msg: impl Into<String>) -> OpError {
    OpError(msg.into())
}

/// The profile: everything the platform can execute today.
pub const PROFILE: &[OpSpec] = &[
    OpSpec {
        name: "MatMul",
        onnx: Some("MatMul (opset 13)"),
        arity: (2, 2),
        attributes: &[],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "matrix product over the last two dimensions",
    },
    OpSpec {
        name: "Add",
        onnx: Some("Add (opset 14)"),
        arity: (2, 2),
        attributes: &[],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "sum, broadcasting over dimensions of size 1",
    },
    OpSpec {
        name: "Sub",
        onnx: Some("Sub (opset 14)"),
        arity: (2, 2),
        attributes: &[],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "difference, same broadcasting rules",
    },
    OpSpec {
        name: "Mul",
        onnx: Some("Mul (opset 14)"),
        arity: (2, 2),
        attributes: &[],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "element-wise product",
    },
    OpSpec {
        name: "Div",
        onnx: Some("Div (opset 14)"),
        arity: (2, 2),
        attributes: &[],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "element-wise quotient",
    },
    OpSpec {
        name: "Relu",
        onnx: Some("Relu (opset 14)"),
        arity: (1, 1),
        attributes: &[],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "rectified linear unit",
    },
    OpSpec {
        name: "Gelu",
        onnx: Some("Gelu (opset 20)"),
        arity: (1, 1),
        attributes: &["approximate"],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "Gaussian error linear unit",
    },
    OpSpec {
        name: "Sigmoid",
        onnx: Some("Sigmoid (opset 13)"),
        arity: (1, 1),
        attributes: &[],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "sigmoid",
    },
    OpSpec {
        name: "Tanh",
        onnx: Some("Tanh (opset 13)"),
        arity: (1, 1),
        attributes: &[],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "hyperbolic tangent",
    },
    OpSpec {
        name: "Softmax",
        onnx: Some("Softmax (opset 13)"),
        arity: (1, 1),
        attributes: &["axis"],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "exponential normalization along an axis",
    },
    OpSpec {
        name: "Slice",
        onnx: Some("Slice (opset 13)"),
        arity: (1, 1),
        attributes: &["axis", "start", "end"],
        required: &["axis", "start", "end"],
        determinism: Determinism::Bitwise,
        summary: "contiguous slice [start, end) along an axis; used for learned positions and \
RoPE-style rotations",
    },
    OpSpec {
        name: "CausalMask",
        onnx: Some("Trilu (opset 14) + Where (opset 16)"),
        arity: (1, 1),
        attributes: &[],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "masks the future positions of an attention score matrix [.., T, T], \
to be placed before the Softmax",
    },
    OpSpec {
        name: "ReduceMean",
        onnx: Some("ReduceMean (opset 18)"),
        arity: (1, 1),
        attributes: &["axis", "keepdims"],
        required: &["axis"],
        determinism: Determinism::Tolerance,
        summary: "mean along an axis, kept with size 1 (keepdims = 1, default) or removed; \
used by hand-written normalizations, such as the Hugging Face RMSNorm",
    },
    OpSpec {
        name: "Rsqrt",
        onnx: Some("Sqrt (opset 13) + Reciprocal (opset 13)"),
        arity: (1, 1),
        attributes: &[],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "reciprocal square root, element-wise",
    },
    OpSpec {
        name: "Formula",
        onnx: None,
        arity: (1, 4),
        attributes: &["expr"],
        required: &["expr"],
        determinism: Determinism::Tolerance,
        summary: "element-wise expression over inputs a, b, c, d (broadcast as for Add): \
+ - * /, power with a literal exponent, exp, log, log1p, sqrt, rsqrt, abs, tanh, sigmoid, relu, sin, \
cos, erf, softplus, max, min, tan, asin, acos, atan, sinh, cosh, asinh, acosh, atanh, atan2, floor, \
ceil, round (half to even), trunc, sign; differentiated automatically (zero derivative for the \
piecewise-constant functions)",
    },
    OpSpec {
        name: "LayerNorm",
        onnx: Some("LayerNormalization (opset 17)"),
        arity: (1, 3),
        attributes: &["axis", "epsilon"],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "layer normalization, with optional gain and bias",
    },
    OpSpec {
        name: "RMSNorm",
        onnx: Some("RMSNormalization (opset 23)"),
        arity: (1, 2),
        attributes: &["axis", "epsilon"],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "root mean square normalization",
    },
    OpSpec {
        name: "Conv2d",
        onnx: Some("Conv (opset 11, 2D)"),
        arity: (2, 3),
        attributes: &["stride", "padding", "dilation", "groups"],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "two-dimensional convolution",
    },
    OpSpec {
        name: "MaxPool2d",
        onnx: Some("MaxPool (opset 12, 2D)"),
        arity: (1, 1),
        attributes: &["kernel", "stride", "padding"],
        required: &["kernel"],
        determinism: Determinism::Bitwise,
        summary: "max pooling",
    },
    OpSpec {
        name: "AvgPool2d",
        onnx: Some("AveragePool (opset 11, 2D)"),
        arity: (1, 1),
        attributes: &["kernel", "stride", "padding"],
        required: &["kernel"],
        determinism: Determinism::Tolerance,
        summary: "average pooling",
    },
    OpSpec {
        name: "BatchNorm2d",
        onnx: Some("BatchNormalization (opset 15)"),
        arity: (3, 5),
        attributes: &["epsilon", "momentum"],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "batch normalization over channels",
    },
    OpSpec {
        name: "Reshape",
        onnx: Some("Reshape (opset 14)"),
        arity: (1, 1),
        attributes: &["shape"],
        required: &["shape"],
        determinism: Determinism::Bitwise,
        summary: "shape change, -1 for an inferred dimension",
    },
    OpSpec {
        name: "Transpose",
        onnx: Some("Transpose (opset 13)"),
        arity: (1, 1),
        attributes: &["perm"],
        required: &["perm"],
        determinism: Determinism::Bitwise,
        summary: "permutation of dimensions",
    },
    OpSpec {
        name: "Flatten",
        onnx: Some("Flatten (opset 13)"),
        arity: (1, 1),
        attributes: &["axis"],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "flattening from an axis onward",
    },
    OpSpec {
        name: "Concat",
        onnx: Some("Concat (opset 13)"),
        arity: (2, 8),
        attributes: &["axis"],
        required: &["axis"],
        determinism: Determinism::Bitwise,
        summary: "concatenation along an axis",
    },
    OpSpec {
        name: "Gather",
        onnx: Some("Gather (opset 13)"),
        arity: (2, 2),
        attributes: &["axis"],
        required: &[],
        determinism: Determinism::Bitwise,
        summary: "index selection, which yields embeddings",
    },
    OpSpec {
        name: "Dropout",
        onnx: Some("Dropout (opset 13)"),
        arity: (1, 1),
        attributes: &["rate"],
        required: &["rate"],
        determinism: Determinism::Seeded,
        summary: "random dropout, drawn from the protocol's DRBG",
    },
    OpSpec {
        name: "ConvTranspose2d",
        onnx: Some("ConvTranspose (opset 11, 2D)"),
        arity: (2, 3),
        attributes: &["stride", "padding", "output_padding", "dilation", "groups"],
        required: &[],
        determinism: Determinism::Tolerance,
        summary: "two-dimensional transposed convolution, kernel [C_in, C_out/groups, KH, KW] \
(PyTorch layout); a 1D one is written with a height of 1",
    },
    OpSpec {
        name: "Pad",
        onnx: Some("Pad (opset 18)"),
        arity: (1, 1),
        attributes: &["pads", "mode", "value"],
        required: &["pads"],
        determinism: Determinism::Bitwise,
        summary: "padding: pads = [begin of each axis…, end of each axis…] (ONNX order), mode \
constant (value), reflect (mirror without the edge) or edge (repeat the edge); never along the \
batch axis",
    },
    OpSpec {
        name: "Resize",
        onnx: Some("Resize (opset 19)"),
        arity: (1, 1),
        attributes: &["size", "mode", "align_corners"],
        required: &["size"],
        determinism: Determinism::Tolerance,
        summary: "resizing of the last axis ([N, C, L]) or the last two ([N, C, H, W]) to `size`, \
mode nearest or linear (bilinear in 2D), source coordinates computed as PyTorch does",
    },
    OpSpec {
        name: "Flip",
        onnx: Some("Slice with step -1 (opset 13)"),
        arity: (1, 1),
        attributes: &["axes"],
        required: &["axes"],
        determinism: Determinism::Bitwise,
        summary: "reverses the order of the elements along the given axes",
    },
    OpSpec {
        name: "ReduceSum",
        onnx: Some("ReduceSum (opset 13)"),
        arity: (1, 1),
        attributes: &["axis", "keepdims"],
        required: &["axis"],
        determinism: Determinism::Tolerance,
        summary: "sum along an axis, kept with size 1 (keepdims = 1, default) or removed",
    },
    OpSpec {
        name: "ReduceMax",
        onnx: Some("ReduceMax (opset 18)"),
        arity: (1, 1),
        attributes: &["axis", "keepdims"],
        required: &["axis"],
        determinism: Determinism::Bitwise,
        summary: "maximum along an axis; the gradient is shared equally between tied maxima, \
as torch.amax does",
    },
    OpSpec {
        name: "ReduceMin",
        onnx: Some("ReduceMin (opset 18)"),
        arity: (1, 1),
        attributes: &["axis", "keepdims"],
        required: &["axis"],
        determinism: Determinism::Bitwise,
        summary: "minimum along an axis; the gradient is shared equally between tied minima",
    },
    OpSpec {
        name: "ReduceProd",
        onnx: Some("ReduceProd (opset 18)"),
        arity: (1, 1),
        attributes: &["axis", "keepdims"],
        required: &["axis"],
        determinism: Determinism::Tolerance,
        summary: "product along an axis, computed by pairwise products: exact gradient even \
with zeros",
    },
    OpSpec {
        name: "CumSum",
        onnx: Some("CumSum (opset 14)"),
        arity: (1, 1),
        attributes: &["axis"],
        required: &["axis"],
        determinism: Determinism::Tolerance,
        summary: "inclusive cumulative sum along an axis",
    },
    OpSpec {
        name: "IndexSelect",
        onnx: Some("Gather (opset 13), constant indices"),
        arity: (1, 1),
        attributes: &["axis", "index"],
        required: &["axis", "index"],
        determinism: Determinism::Bitwise,
        summary: "selects, along an axis, the slices listed in `index` (constant, written in \
the description, repetitions allowed): relative position tables, fixed graphs",
    },
    OpSpec {
        name: "IndexAdd",
        onnx: Some("ScatterElements (opset 18, reduction add), constant indices"),
        arity: (2, 2),
        attributes: &["axis", "index"],
        required: &["axis", "index"],
        determinism: Determinism::Tolerance,
        summary: "x + src accumulated along an axis at the positions listed in `index` \
(torch.index_add): message passing over a fixed graph",
    },
    OpSpec {
        name: "ArgMax",
        onnx: Some("ArgMax (opset 13, select_last_index = 0)"),
        arity: (1, 1),
        attributes: &["axis", "keepdims"],
        required: &["axis"],
        determinism: Determinism::Bitwise,
        summary: "index (i64) of the maximum along an axis; the first one among ties, a NaN counting \
as the largest; no gradient",
    },
    OpSpec {
        name: "ArgMin",
        onnx: Some("ArgMin (opset 13, select_last_index = 0)"),
        arity: (1, 1),
        attributes: &["axis", "keepdims"],
        required: &["axis"],
        determinism: Determinism::Bitwise,
        summary: "index (i64) of the minimum along an axis; the first one among ties; no gradient",
    },
    OpSpec {
        name: "ArgTopK",
        onnx: Some("TopK (opset 11), indices output"),
        arity: (1, 1),
        attributes: &["axis", "k", "largest"],
        required: &["axis", "k"],
        determinism: Determinism::Bitwise,
        summary: "indices (i64) of the k largest (largest = 1, default) or smallest elements along an \
axis, in order, the lower index first among ties; the values are GatherElements(x, indices)",
    },
    OpSpec {
        name: "ArgSort",
        onnx: None,
        arity: (1, 1),
        attributes: &["axis", "descending"],
        required: &["axis"],
        determinism: Determinism::Bitwise,
        summary: "stable sorting permutation (i64) along an axis, ascending or descending; the \
sorted values are GatherElements(x, permutation)",
    },
    OpSpec {
        name: "Cast",
        onnx: Some("Cast (opset 13)"),
        arity: (1, 1),
        attributes: &["to"],
        required: &["to"],
        determinism: Determinism::Bitwise,
        summary: "an integer tensor as f32 (to = \"f32\"), exact; no gradient",
    },
    OpSpec {
        name: "GatherElements",
        onnx: Some("GatherElements (opset 13)"),
        arity: (2, 2),
        attributes: &["axis"],
        required: &["axis"],
        determinism: Determinism::Bitwise,
        summary: "out[…, j, …] = x[…, index[…, j, …], …] along an axis (torch.gather), index an i64 \
tensor of the rank of x; the gradient reaches the selected positions",
    },
    OpSpec {
        name: "ScatterElements",
        onnx: Some("ScatterElements (opset 18)"),
        arity: (3, 3),
        attributes: &["axis", "reduction"],
        required: &["axis"],
        determinism: Determinism::Tolerance,
        summary: "x with src accumulated (reduction = add, default) or written (none: indices known \
to be distinct along the axis) at the positions of an i64 index (torch.scatter_add, scatter)",
    },
];

/// Looks up an operator of the profile.
pub fn lookup(name: &str) -> Option<&'static OpSpec> {
    PROFILE.iter().find(|o| o.name == name)
}

/// Profile operators closest to an unknown name, to help without guessing.
pub fn suggestions(name: &str) -> Vec<&'static str> {
    let lower = name.to_ascii_lowercase();
    PROFILE
        .iter()
        .filter(|o| {
            let n = o.name.to_ascii_lowercase();
            n.starts_with(&lower[..lower.len().min(3)]) || n.contains(&lower) || lower.contains(&n)
        })
        .map(|o| o.name)
        .take(3)
        .collect()
}

fn broadcast(a: &Shape, b: &Shape) -> Result<Shape, OpError> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = a.get(a.len().wrapping_sub(rank - i)).copied();
        let db = b.get(b.len().wrapping_sub(rank - i)).copied();
        let dim = match (da.flatten(), db.flatten()) {
            (Some(x), Some(y)) if x == y => Some(x),
            (Some(1), Some(y)) => Some(y),
            (Some(x), Some(1)) => Some(x),
            (None, Some(1)) | (Some(1), None) | (None, None) => None,
            (None, Some(y)) => Some(y),
            (Some(x), None) => Some(x),
            (Some(x), Some(y)) => {
                return Err(err(format!(
                    "incompatible shapes: {} and {} do not broadcast ({x} vs {y})",
                    show(a),
                    show(b)
                )));
            }
        };
        out.push(dim);
    }
    Ok(out)
}

fn attr_ints(attrs: &BTreeMap<String, Attr>, name: &str, default: &[i64]) -> Vec<i64> {
    attrs
        .get(name)
        .and_then(|a| a.as_ints().map(|v| v.to_vec()))
        .unwrap_or_else(|| default.to_vec())
}

fn pair(values: &[i64], name: &str) -> Result<(i64, i64), OpError> {
    match values.len() {
        1 => Ok((values[0], values[0])),
        2 => Ok((values[0], values[1])),
        n => Err(err(format!("{name} expects 1 or 2 integers, {n} given"))),
    }
}

/// Output shape of a node, or the exact error that prevents it.
pub fn infer(op: &str, inputs: &[Shape], attrs: &BTreeMap<String, Attr>) -> Result<Shape, OpError> {
    match op {
        "Add" | "Sub" | "Mul" | "Div" => broadcast(&inputs[0], &inputs[1]),
        "Formula" => {
            let expr = match attrs.get("expr") {
                Some(Attr::Text(e)) => e,
                _ => return Err(err("Formula: the expr attribute must be a string")),
            };
            super::formula::check(expr, inputs.len()).map_err(err)?;
            let mut out = inputs[0].clone();
            for s in &inputs[1..] {
                out = broadcast(&out, s)?;
            }
            Ok(out)
        }
        "Gelu" => match attrs.get("approximate") {
            None => Ok(inputs[0].clone()),
            Some(Attr::Text(a)) if a == "none" || a == "tanh" => Ok(inputs[0].clone()),
            Some(other) => Err(err(format!(
                "Gelu: approximate must be \"none\" or \"tanh\", got {other:?}"
            ))),
        },
        "Relu" | "Sigmoid" | "Tanh" | "Softmax" | "Dropout" | "Rsqrt" => Ok(inputs[0].clone()),
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd" => {
            reduce_shape(op, &inputs[0], attrs)
        }
        "Slice" => {
            let x = &inputs[0];
            let rank = x.len() as i64;
            let get = |k: &str| {
                attrs
                    .get(k)
                    .and_then(Attr::as_int)
                    .ok_or_else(|| err(format!("Slice: the {k} attribute must be an integer")))
            };
            let (axis, start, end) = (get("axis")?, get("start")?, get("end")?);
            let axis = if axis < 0 { axis + rank } else { axis };
            if axis < 0 || axis >= rank {
                return Err(err(format!(
                    "Slice: axis {} outside the rank {rank} of {}",
                    get("axis")?,
                    show(x)
                )));
            }
            let axis = axis as usize;
            let Some(size) = x[axis] else {
                return Err(err(format!(
                    "Slice: axis {axis} of {} is the batch axis, which cannot be sliced",
                    show(x)
                )));
            };
            if start < 0 || end <= start || end as u64 > size {
                return Err(err(format!(
                    "Slice: range [{start}, {end}) outside axis {axis} of size {size}"
                )));
            }
            let mut out = x.clone();
            out[axis] = Some((end - start) as u64);
            Ok(out)
        }
        "CausalMask" => {
            let x = &inputs[0];
            if x.len() < 2 {
                return Err(err(format!(
                    "CausalMask expects attention scores [.., T, T], got {}",
                    show(x)
                )));
            }
            let (rows, cols) = (x[x.len() - 2], x[x.len() - 1]);
            match (rows, cols) {
                (Some(r), Some(c)) if r != c => Err(err(format!(
                    "CausalMask expects a square matrix over the last two axes (queries and \
keys of the same length), got {}",
                    show(x)
                ))),
                (None, _) | (_, None) => Err(err(format!(
                    "CausalMask expects a known sequence length over the last two axes, \
got {}: the batch axis cannot appear there",
                    show(x)
                ))),
                _ => Ok(x.clone()),
            }
        }
        "LayerNorm" | "RMSNorm" | "BatchNorm2d" => Ok(inputs[0].clone()),
        "MatMul" => {
            let (a, b) = (&inputs[0], &inputs[1]);
            if a.len() < 2 || b.len() < 2 {
                return Err(err(format!(
                    "MatMul requires two tensors of at least two dimensions, got {} and {}",
                    show(a),
                    show(b)
                )));
            }
            let (ak, bk) = (a[a.len() - 1], b[b.len() - 2]);
            if let (Some(x), Some(y)) = (ak, bk)
                && x != y
            {
                return Err(err(format!(
                    "MatMul: {} and {} do not match ({x} vs {y} on the inner dimension)",
                    show(a),
                    show(b)
                )));
            }
            // Broadcast batch dimensions, then (m, n).
            let batch_a: Shape = a[..a.len() - 2].to_vec();
            let batch_b: Shape = b[..b.len() - 2].to_vec();
            let mut out = broadcast(&batch_a, &batch_b)?;
            out.push(a[a.len() - 2]);
            out.push(b[b.len() - 1]);
            Ok(out)
        }
        "Conv2d" => {
            let (x, w) = (&inputs[0], &inputs[1]);
            if x.len() != 4 || w.len() != 4 {
                return Err(err(format!(
                    "Conv2d expects an input [N, C, H, W] and a kernel [F, C/g, KH, KW], got {} and {}",
                    show(x),
                    show(w)
                )));
            }
            let groups = attrs
                .get("groups")
                .and_then(Attr::as_int)
                .unwrap_or(1)
                .max(1) as u64;
            let (sh, sw) = pair(&attr_ints(attrs, "stride", &[1]), "stride")?;
            let (ph, pw) = pair(&attr_ints(attrs, "padding", &[0]), "padding")?;
            let (dh, dw) = pair(&attr_ints(attrs, "dilation", &[1]), "dilation")?;
            if let (Some(cin), Some(wc)) = (x[1], w[1])
                && cin != wc * groups
            {
                return Err(err(format!(
                    "Conv2d: {cin} input channels, the kernel expects {} (groups = {groups})",
                    wc * groups
                )));
            }
            let out_dim =
                |size: Option<u64>, k: Option<u64>, s: i64, p: i64, d: i64| match (size, k) {
                    (Some(i), Some(k)) => {
                        let eff = d * (k as i64 - 1) + 1;
                        let v = (i as i64 + 2 * p - eff) / s + 1;
                        (v > 0).then_some(v as u64)
                    }
                    _ => None,
                };
            let h = out_dim(x[2], w[2], sh, ph, dh);
            let ow = out_dim(x[3], w[3], sw, pw, dw);
            if h.is_none() && x[2].is_some() {
                return Err(err(
                    "Conv2d: the window exceeds the input height".to_string()
                ));
            }
            if ow.is_none() && x[3].is_some() {
                return Err(err("Conv2d: the window exceeds the input width".to_string()));
            }
            Ok(vec![x[0], w[0], h, ow])
        }
        "MaxPool2d" | "AvgPool2d" => {
            let x = &inputs[0];
            if x.len() != 4 {
                return Err(err(format!("{op} expects [N, C, H, W], got {}", show(x))));
            }
            let (kh, kw) = pair(&attr_ints(attrs, "kernel", &[2]), "kernel")?;
            let (sh, sw) = pair(&attr_ints(attrs, "stride", &[kh, kw]), "stride")?;
            let (ph, pw) = pair(&attr_ints(attrs, "padding", &[0]), "padding")?;
            let out_dim = |size: Option<u64>, k: i64, s: i64, p: i64| {
                size.and_then(|i| {
                    let v = (i as i64 + 2 * p - k) / s + 1;
                    (v > 0).then_some(v as u64)
                })
            };
            Ok(vec![
                x[0],
                x[1],
                out_dim(x[2], kh, sh, ph),
                out_dim(x[3], kw, sw, pw),
            ])
        }
        "Reshape" => {
            let x = &inputs[0];
            let target = attr_ints(attrs, "shape", &[]);
            if target.is_empty() {
                return Err(err("Reshape: the shape attribute is empty".to_string()));
            }
            if target.iter().filter(|d| **d == -1).count() > 1 {
                return Err(err("Reshape: only one dimension can be -1".to_string()));
            }
            // The batch stays unknown: by convention, the first dimension carries it.
            let known: u64 = x.iter().skip(1).filter_map(|d| *d).product();
            let fixed: i64 = target.iter().skip(1).filter(|d| **d > 0).product();
            let mut out: Shape = Vec::with_capacity(target.len());
            for (i, d) in target.iter().enumerate() {
                out.push(match d {
                    0 => x.get(i).copied().flatten(),
                    -1 if i == 0 => None,
                    -1 if fixed > 0 && known.is_multiple_of(fixed as u64) => {
                        Some(known / fixed as u64)
                    }
                    -1 => None,
                    v if *v > 0 => Some(*v as u64),
                    _ => return Err(err(format!("Reshape: invalid dimension {d}"))),
                });
            }
            Ok(out)
        }
        "Transpose" => {
            let x = &inputs[0];
            let perm = attr_ints(attrs, "perm", &[]);
            if perm.len() != x.len() {
                return Err(err(format!(
                    "Transpose: perm has {} elements for a tensor of rank {}",
                    perm.len(),
                    x.len()
                )));
            }
            let mut seen = vec![false; x.len()];
            let mut out = Vec::with_capacity(x.len());
            for p in &perm {
                let idx = usize::try_from(*p)
                    .map_err(|_| err(format!("Transpose: invalid index {p}")))?;
                if idx >= x.len() || seen[idx] {
                    return Err(err(format!(
                        "Transpose: perm {perm:?} is not a permutation"
                    )));
                }
                seen[idx] = true;
                out.push(x[idx]);
            }
            Ok(out)
        }
        "Flatten" => {
            let x = &inputs[0];
            let axis = attrs.get("axis").and_then(Attr::as_int).unwrap_or(1);
            let axis = usize::try_from(axis).map_err(|_| err("Flatten: negative axis"))?;
            if axis > x.len() {
                return Err(err(format!("Flatten: axis {axis} beyond rank {}", x.len())));
            }
            let head: Option<u64> = x[..axis].iter().try_fold(1u64, |a, d| d.map(|v| a * v));
            let tail: Option<u64> = x[axis..].iter().try_fold(1u64, |a, d| d.map(|v| a * v));
            Ok(vec![head, tail])
        }
        "Concat" => {
            let axis = attrs
                .get("axis")
                .and_then(Attr::as_int)
                .ok_or_else(|| err("Concat: the axis attribute is required"))?;
            let axis = usize::try_from(axis).map_err(|_| err("Concat: negative axis"))?;
            let first = &inputs[0];
            if axis >= first.len() {
                return Err(err(format!(
                    "Concat: axis {axis} beyond rank {}",
                    first.len()
                )));
            }
            let mut out = first.clone();
            let mut total = first[axis];
            for other in &inputs[1..] {
                if other.len() != first.len() {
                    return Err(err(format!(
                        "Concat: different ranks, {} and {}",
                        show(first),
                        show(other)
                    )));
                }
                for (i, (a, b)) in first.iter().zip(other.iter()).enumerate() {
                    if i != axis
                        && let (Some(x), Some(y)) = (a, b)
                        && x != y
                    {
                        return Err(err(format!(
                            "Concat: {} and {} differ outside axis {axis}",
                            show(first),
                            show(other)
                        )));
                    }
                }
                total = match (total, other[axis]) {
                    (Some(a), Some(b)) => Some(a + b),
                    _ => None,
                };
            }
            out[axis] = total;
            Ok(out)
        }
        "Gather" => {
            let (data, idx) = (&inputs[0], &inputs[1]);
            let axis = attrs.get("axis").and_then(Attr::as_int).unwrap_or(0);
            // Row selection: axis 0 of a table of rank 2 to 4 (an embedding, the weights of the
            // experts chosen by a router, ADR-0029), by indices of any rank. Any other case is
            // rejected here, not discovered at execution time.
            if axis != 0 {
                return Err(err(format!(
                    "Gather: only axis = 0 is executed (row selection), got {axis}"
                )));
            }
            if !(2..=4).contains(&data.len()) {
                return Err(err(format!(
                    "Gather: the table must be of rank 2 to 4 [entries, …], got {}",
                    show(data)
                )));
            }
            if data[0].is_none() {
                return Err(err(format!(
                    "Gather: the table {} has the batch axis first; selecting along it would mix \
examples",
                    show(data)
                )));
            }
            if idx.is_empty() {
                return Err(err(format!(
                    "Gather: the indices must be of rank 1 or more, got {}",
                    show(idx)
                )));
            }
            let mut out = idx.clone();
            out.extend_from_slice(&data[1..]);
            Ok(out)
        }
        "ArgMax" | "ArgMin" => {
            let out = reduce_shape(op, &inputs[0], attrs)?;
            index_axis(op, &inputs[0], attrs)?;
            Ok(out)
        }
        "ArgTopK" => {
            let x = &inputs[0];
            let (axis, size) = index_axis(op, x, attrs)?;
            let k = attrs
                .get("k")
                .and_then(Attr::as_int)
                .ok_or_else(|| err("ArgTopK: the k attribute must be an integer"))?;
            if k < 1 || k as u64 > size {
                return Err(err(format!(
                    "ArgTopK: k = {k} for an axis of {size}; expected 1 to {size}"
                )));
            }
            flag(op, attrs, "largest", 1)?;
            let mut out = x.clone();
            out[axis] = Some(k as u64);
            Ok(out)
        }
        "ArgSort" => {
            index_axis(op, &inputs[0], attrs)?;
            flag(op, attrs, "descending", 0)?;
            Ok(inputs[0].clone())
        }
        "Cast" => match attrs.get("to") {
            Some(Attr::Text(t)) if t == "f32" => Ok(inputs[0].clone()),
            _ => Err(err("Cast: to must be \"f32\" (an integer tensor as f32)")),
        },
        "GatherElements" => {
            let (x, idx) = (&inputs[0], &inputs[1]);
            let axis = elements_axis(op, x, idx, attrs)?;
            for (i, (a, b)) in x.iter().zip(idx).enumerate() {
                if i != axis
                    && let (Some(a), Some(b)) = (a, b)
                    && b > a
                {
                    return Err(err(format!(
                        "GatherElements: the index {} is larger than x {} on axis {i}",
                        show(idx),
                        show(x)
                    )));
                }
            }
            Ok(idx.clone())
        }
        "ScatterElements" => {
            let (x, idx, src) = (&inputs[0], &inputs[1], &inputs[2]);
            let axis = elements_axis(op, x, idx, attrs)?;
            if src != idx {
                return Err(err(format!(
                    "ScatterElements: src {} must have the shape of the index {}",
                    show(src),
                    show(idx)
                )));
            }
            for (i, (a, b)) in x.iter().zip(idx).enumerate() {
                if i != axis && a != b {
                    return Err(err(format!(
                        "ScatterElements: the index {} and x {} differ on axis {i}, outside the \
scatter axis {axis}",
                        show(idx),
                        show(x)
                    )));
                }
            }
            match attrs.get("reduction") {
                None => {}
                Some(Attr::Text(r)) if r == "add" || r == "none" => {}
                Some(other) => {
                    return Err(err(format!(
                        "ScatterElements: reduction {other:?}; add or none (amax, amin, mul are \
not executed, ADR-0029)"
                    )));
                }
            }
            Ok(x.clone())
        }
        "ConvTranspose2d" => conv_transpose_shape(inputs, attrs),
        "Pad" => pad_shape(&inputs[0], attrs),
        "Resize" => resize_shape(&inputs[0], attrs),
        "Flip" => {
            let x = &inputs[0];
            let axes = attr_ints(attrs, "axes", &[]);
            if axes.is_empty() {
                return Err(err("Flip: the axes attribute is an empty list"));
            }
            let mut seen = vec![false; x.len()];
            for a in &axes {
                let i = axis_of("Flip", *a, x)?;
                if seen[i] {
                    return Err(err(format!("Flip: axis {a} appears twice")));
                }
                seen[i] = true;
            }
            Ok(x.clone())
        }
        "CumSum" => {
            let x = &inputs[0];
            let axis = attrs
                .get("axis")
                .and_then(Attr::as_int)
                .ok_or_else(|| err("CumSum: the axis attribute must be an integer"))?;
            axis_of("CumSum", axis, x)?;
            Ok(x.clone())
        }
        "IndexSelect" => {
            let x = &inputs[0];
            let (axis, index) = index_attrs("IndexSelect", x, attrs)?;
            let mut out = x.clone();
            out[axis] = Some(index.len() as u64);
            Ok(out)
        }
        "IndexAdd" => {
            let (x, src) = (&inputs[0], &inputs[1]);
            let (axis, index) = index_attrs("IndexAdd", x, attrs)?;
            if src.len() != x.len() {
                return Err(err(format!(
                    "IndexAdd: src {} and x {} do not have the same rank",
                    show(src),
                    show(x)
                )));
            }
            for (i, (a, b)) in x.iter().zip(src).enumerate() {
                let want = if i == axis {
                    Some(index.len() as u64)
                } else {
                    *a
                };
                if want != *b {
                    return Err(err(format!(
                        "IndexAdd: src must be {} ({} indices along axis {axis}), got {}",
                        show(&{
                            let mut w = x.clone();
                            w[axis] = Some(index.len() as u64);
                            w
                        }),
                        index.len(),
                        show(src)
                    )));
                }
            }
            Ok(x.clone())
        }
        other => Err(err(format!(
            "operator {other} is not in profile {PROFILE_VERSION}"
        ))),
    }
}

/// Non-negative index of `axis` (negative values count from the end) in `x`, refusing the batch
/// axis: an operator that works along it would mix examples, which the execution model splits.
fn axis_of(op: &str, axis: i64, x: &Shape) -> Result<usize, OpError> {
    let rank = x.len() as i64;
    let a = if axis < 0 { axis + rank } else { axis };
    if a < 0 || a >= rank {
        return Err(err(format!(
            "{op}: axis {axis} outside the rank {rank} of {}",
            show(x)
        )));
    }
    let a = a as usize;
    if x[a].is_none() {
        return Err(err(format!(
            "{op}: axis {a} of {} is the batch axis; the operator would mix examples",
            show(x)
        )));
    }
    Ok(a)
}

/// Axis (non-batch) and size of the axis an index is computed over; bounded by `MAX_INDEX_AXIS`.
fn index_axis(
    op: &str,
    x: &Shape,
    attrs: &BTreeMap<String, Attr>,
) -> Result<(usize, u64), OpError> {
    let axis = attrs
        .get("axis")
        .and_then(Attr::as_int)
        .ok_or_else(|| err(format!("{op}: the axis attribute must be an integer")))?;
    let a = axis_of(op, axis, x)?;
    let size = x[a].expect("batch axis refused by axis_of");
    if size > MAX_INDEX_AXIS {
        return Err(err(format!(
            "{op}: axis {a} has {size} elements; an index is computed over at most \
{MAX_INDEX_AXIS}"
        )));
    }
    Ok((a, size))
}

/// A 0/1 attribute.
fn flag(
    op: &str,
    attrs: &BTreeMap<String, Attr>,
    name: &str,
    default: i64,
) -> Result<bool, OpError> {
    match attrs.get(name).map(Attr::as_int) {
        None => Ok(default == 1),
        Some(Some(v @ (0 | 1))) => Ok(v == 1),
        _ => Err(err(format!("{op}: {name} must be 0 or 1"))),
    }
}

/// Axis of `GatherElements` / `ScatterElements`: same rank for x and the index, batch axis in
/// both or in neither, never along it, the axis indexed at most `MAX_INDEX_AXIS`.
fn elements_axis(
    op: &str,
    x: &Shape,
    idx: &Shape,
    attrs: &BTreeMap<String, Attr>,
) -> Result<usize, OpError> {
    if x.len() != idx.len() {
        return Err(err(format!(
            "{op}: the index {} and x {} must have the same rank",
            show(idx),
            show(x)
        )));
    }
    if x[0].is_none() != idx[0].is_none() {
        return Err(err(format!(
            "{op}: x {} and the index {} must both carry the batch axis first, or neither",
            show(x),
            show(idx)
        )));
    }
    let (axis, _) = index_axis(op, x, attrs)?;
    if idx[axis].is_none() {
        return Err(err(format!(
            "{op}: the index {} has the batch axis on the indexed axis {axis}",
            show(idx)
        )));
    }
    Ok(axis)
}

/// Shape of a reduction along one axis (`ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`,
/// `ReduceProd`).
fn reduce_shape(op: &str, x: &Shape, attrs: &BTreeMap<String, Attr>) -> Result<Shape, OpError> {
    let axis = attrs
        .get("axis")
        .and_then(Attr::as_int)
        .ok_or_else(|| err(format!("{op}: the axis attribute must be an integer")))?;
    let rank = x.len() as i64;
    let a = if axis < 0 { axis + rank } else { axis };
    if a < 0 || a >= rank {
        return Err(err(format!(
            "{op}: axis {axis} outside the rank {rank} of {}",
            show(x)
        )));
    }
    let a = a as usize;
    if x[a].is_none() {
        return Err(err(format!(
            "{op}: axis {a} of {} is the batch axis; a reduction over the batch would mix \
examples",
            show(x)
        )));
    }
    let keep = attrs.get("keepdims").and_then(Attr::as_int).unwrap_or(1);
    match keep {
        1 => {
            let mut out = x.clone();
            out[a] = Some(1);
            Ok(out)
        }
        0 if x.len() > 1 => {
            let mut out = x.clone();
            out.remove(a);
            Ok(out)
        }
        0 => Err(err(format!(
            "{op}: a rank-1 tensor cannot be reduced to a scalar"
        ))),
        k => Err(err(format!("{op}: keepdims must be 0 or 1, got {k}"))),
    }
}

/// Axis and constant indices of `IndexSelect` / `IndexAdd`, checked against `x`.
fn index_attrs<'a>(
    op: &str,
    x: &Shape,
    attrs: &'a BTreeMap<String, Attr>,
) -> Result<(usize, &'a [i64]), OpError> {
    let axis = attrs
        .get("axis")
        .and_then(Attr::as_int)
        .ok_or_else(|| err(format!("{op}: the axis attribute must be an integer")))?;
    let axis = axis_of(op, axis, x)?;
    let index = attrs.get("index").and_then(Attr::as_ints).ok_or_else(|| {
        err(format!(
            "{op}: the index attribute must be a list of integers"
        ))
    })?;
    if index.is_empty() || index.len() > MAX_INDICES {
        return Err(err(format!(
            "{op}: {} indices, expected 1 to {MAX_INDICES}",
            index.len()
        )));
    }
    let size = x[axis].expect("batch axis refused by axis_of");
    if let Some((k, v)) = index
        .iter()
        .enumerate()
        .find(|(_, v)| **v < 0 || **v as u64 >= size)
    {
        return Err(err(format!(
            "{op}: index {k} is {v}, outside axis {axis} of size {size}"
        )));
    }
    Ok((axis, index))
}

/// Non-negative integers of a spatial attribute, one or two values.
fn spatial(
    op: &str,
    attrs: &BTreeMap<String, Attr>,
    name: &str,
    default: i64,
    min: i64,
) -> Result<(i64, i64), OpError> {
    let (a, b) = pair(&attr_ints(attrs, name, &[default]), name)?;
    if a < min || b < min || a > MAX_ATTR_DIM || b > MAX_ATTR_DIM {
        return Err(err(format!(
            "{op}: {name} ({a}, {b}) outside [{min}, {MAX_ATTR_DIM}]"
        )));
    }
    Ok((a, b))
}

/// Output shape of `ConvTranspose2d`: `(i − 1)·s − 2p + d·(k − 1) + output_padding + 1`.
fn conv_transpose_shape(
    inputs: &[Shape],
    attrs: &BTreeMap<String, Attr>,
) -> Result<Shape, OpError> {
    let (x, w) = (&inputs[0], &inputs[1]);
    if x.len() != 4 || w.len() != 4 {
        return Err(err(format!(
            "ConvTranspose2d expects an input [N, C_in, H, W] and a kernel [C_in, C_out/g, KH, KW], \
got {} and {}",
            show(x),
            show(w)
        )));
    }
    let op = "ConvTranspose2d";
    let groups = attrs.get("groups").and_then(Attr::as_int).unwrap_or(1);
    if !(1..=MAX_ATTR_DIM).contains(&groups) {
        return Err(err(format!("{op}: groups {groups} must be at least 1")));
    }
    let groups = groups as u64;
    let (sh, sw) = spatial(op, attrs, "stride", 1, 1)?;
    let (ph, pw) = spatial(op, attrs, "padding", 0, 0)?;
    let (dh, dw) = spatial(op, attrs, "dilation", 1, 1)?;
    let (oh, ow) = spatial(op, attrs, "output_padding", 0, 0)?;
    if oh >= sh.max(dh) || ow >= sw.max(dw) {
        return Err(err(format!(
            "{op}: output_padding ({oh}, {ow}) must be smaller than the stride or the dilation"
        )));
    }
    let (Some(cin), Some(wc), Some(per_group), Some(kh), Some(kw)) = (x[1], w[0], w[1], w[2], w[3])
    else {
        return Err(err(format!(
            "{op}: channels and kernel must be known, got {} and {}",
            show(x),
            show(w)
        )));
    };
    if cin != wc {
        return Err(err(format!(
            "{op}: {cin} input channels, the kernel {} expects {wc}",
            show(w)
        )));
    }
    if !cin.is_multiple_of(groups) {
        return Err(err(format!(
            "{op}: {cin} input channels are not divisible into {groups} groups"
        )));
    }
    let cout = per_group * groups;
    if let Some(b) = inputs.get(2)
        && (b.len() != 1 || b[0] != Some(cout))
    {
        return Err(err(format!(
            "{op}: the bias must be [{cout}], got {}",
            show(b)
        )));
    }
    let out_dim = |size: Option<u64>, k: u64, s: i64, p: i64, d: i64, o: i64| {
        size.map(|i| (i as i64 - 1) * s - 2 * p + d * (k as i64 - 1) + o + 1)
    };
    let h = out_dim(x[2], kh, sh, ph, dh, oh);
    let wd = out_dim(x[3], kw, sw, pw, dw, ow);
    for (v, what) in [(h, "height"), (wd, "width")] {
        if let Some(v) = v
            && v <= 0
        {
            return Err(err(format!(
                "{op}: the padding removes the whole output {what} ({v})"
            )));
        }
    }
    Ok(vec![
        x[0],
        Some(cout),
        h.map(|v| v as u64),
        wd.map(|v| v as u64),
    ])
}

/// Padding mode of `Pad`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PadMode {
    /// A constant value.
    Constant(f32),
    /// Mirror without the edge (`[1, 2, 3]` padded by 2 on the left: `[3, 2, 1, 2, 3]`).
    Reflect,
    /// Repetition of the edge.
    Edge,
}

/// Mode and per-axis padding `(begin, end)` of a `Pad` node, checked against the shape `x`.
pub fn pad_spec(
    x: &Shape,
    attrs: &BTreeMap<String, Attr>,
) -> Result<(PadMode, Vec<(usize, usize)>), OpError> {
    let pads = attrs
        .get("pads")
        .and_then(Attr::as_ints)
        .ok_or_else(|| err("Pad: the pads attribute must be a list of integers"))?;
    let rank = x.len();
    if pads.len() != 2 * rank {
        return Err(err(format!(
            "Pad: pads has {} values for a tensor of rank {rank}; {} expected (begins, then ends)",
            pads.len(),
            2 * rank
        )));
    }
    if let Some(p) = pads.iter().find(|p| !(0..=MAX_ATTR_DIM).contains(*p)) {
        return Err(err(format!(
            "Pad: padding {p} outside [0, {MAX_ATTR_DIM}]; a negative padding is a Slice"
        )));
    }
    let mode = match attrs.get("mode") {
        None => "constant",
        Some(Attr::Text(m)) => m.as_str(),
        Some(_) => return Err(err("Pad: mode must be text")),
    };
    let value = attrs.get("value");
    let mode = match mode {
        "constant" => {
            let v = match value {
                None => 0.0,
                Some(v) => v
                    .as_float()
                    .ok_or_else(|| err("Pad: value must be a number"))?,
            };
            if !v.is_finite() {
                return Err(err(format!("Pad: value {v} is not finite")));
            }
            PadMode::Constant(v as f32)
        }
        "reflect" | "edge" if value.is_some() => {
            return Err(err(format!(
                "Pad: a value only goes with the constant mode, not {mode}"
            )));
        }
        "reflect" => PadMode::Reflect,
        "edge" => PadMode::Edge,
        other => {
            return Err(err(format!(
                "Pad: unknown mode '{other}' (constant, reflect, edge)"
            )));
        }
    };
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let (b, e) = (pads[i] as usize, pads[rank + i] as usize);
        match x[i] {
            None if b != 0 || e != 0 => {
                return Err(err(format!(
                    "Pad: axis {i} of {} is the batch axis, which cannot be padded",
                    show(x)
                )));
            }
            Some(size) if mode == PadMode::Reflect && (b as u64 >= size || e as u64 >= size) => {
                return Err(err(format!(
                    "Pad: a reflection of ({b}, {e}) needs more than {size} elements along axis {i}"
                )));
            }
            _ => {}
        }
        out.push((b, e));
    }
    Ok((mode, out))
}

fn pad_shape(x: &Shape, attrs: &BTreeMap<String, Attr>) -> Result<Shape, OpError> {
    let (_, pads) = pad_spec(x, attrs)?;
    Ok(x.iter()
        .zip(&pads)
        .map(|(d, (b, e))| d.map(|v| v + (*b + *e) as u64))
        .collect())
}

/// Interpolation of `Resize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeMode {
    /// Nearest neighbor, `src = min(floor(dst · in/out), in − 1)` in f32, as PyTorch's `nearest`.
    Nearest,
    /// Linear interpolation (bilinear in 2D), half-pixel coordinates or aligned corners.
    Linear {
        /// Corners aligned (`align_corners=True` in PyTorch).
        align_corners: bool,
    },
}

/// Mode and target size of a `Resize` node, checked against the shape `x`.
pub fn resize_spec(
    x: &Shape,
    attrs: &BTreeMap<String, Attr>,
) -> Result<(ResizeMode, Vec<usize>), OpError> {
    if !(3..=4).contains(&x.len()) {
        return Err(err(format!(
            "Resize expects [N, C, L] or [N, C, H, W], got {}",
            show(x)
        )));
    }
    let size = attrs
        .get("size")
        .and_then(Attr::as_ints)
        .ok_or_else(|| err("Resize: the size attribute must be a list of integers"))?;
    let spatial_axes = x.len() - 2;
    if size.len() != spatial_axes {
        return Err(err(format!(
            "Resize: {} sizes for {} spatial axes of {}",
            size.len(),
            spatial_axes,
            show(x)
        )));
    }
    if let Some(v) = size.iter().find(|v| !(1..=MAX_ATTR_DIM).contains(*v)) {
        return Err(err(format!("Resize: size {v} outside [1, {MAX_ATTR_DIM}]")));
    }
    if x[2..].iter().any(|d| d.is_none()) {
        return Err(err(format!(
            "Resize: the spatial axes of {} must be known",
            show(x)
        )));
    }
    let align = match attrs.get("align_corners") {
        None => None,
        Some(Attr::Int(v @ (0 | 1))) => Some(*v == 1),
        Some(_) => return Err(err("Resize: align_corners must be 0 or 1")),
    };
    let mode = match attrs.get("mode") {
        None => "nearest",
        Some(Attr::Text(m)) => m.as_str(),
        Some(_) => return Err(err("Resize: mode must be text")),
    };
    let mode = match (mode, align) {
        ("nearest", None) => ResizeMode::Nearest,
        ("nearest", Some(_)) => {
            return Err(err("Resize: align_corners only applies to the linear mode"));
        }
        ("linear", a) => ResizeMode::Linear {
            align_corners: a.unwrap_or(false),
        },
        (other, _) => {
            return Err(err(format!(
                "Resize: unknown mode '{other}' (nearest, linear)"
            )));
        }
    };
    Ok((mode, size.iter().map(|v| *v as usize).collect()))
}

fn resize_shape(x: &Shape, attrs: &BTreeMap<String, Attr>) -> Result<Shape, OpError> {
    let (_, size) = resize_spec(x, attrs)?;
    let mut out = x[..2].to_vec();
    out.extend(size.iter().map(|v| Some(*v as u64)));
    Ok(out)
}

/// Expression of an already validated `Formula` node.
fn formula_of(attrs: &BTreeMap<String, Attr>) -> Option<super::formula::Expr> {
    match attrs.get("expr") {
        Some(Attr::Text(e)) => super::formula::parse(e).ok(),
        _ => None,
    }
}

/// Elements a node keeps for the backward pass, for **one** example: its output, plus what its
/// computation creates and automatic differentiation may keep (the intermediate steps of a
/// `Formula`, the mask of a `ReduceMax`, the interpolation steps of a `Resize`). This is the
/// activation memory estimator of each operator (ADR-0016).
pub fn activation_elements(
    op: &str,
    inputs: &[Shape],
    output: &Shape,
    attrs: &BTreeMap<String, Attr>,
) -> u64 {
    let count = |s: &Shape| -> u64 { s.iter().map(|d| d.unwrap_or(1)).product() };
    let out = count(output);
    match op {
        "Formula" => out.saturating_mul(formula_of(attrs).map(|e| e.intermediates()).unwrap_or(1)),
        // The equality mask and the difference to the extremum, both of input size.
        "ReduceMax" | "ReduceMin" => out.saturating_add(count(&inputs[0]).saturating_mul(2)),
        // Pairwise products: n/2 + n/4 + … < n.
        "ReduceProd" => out.saturating_add(count(&inputs[0])),
        // ADR-0029: the index (8 bytes, two units) kept for the backward pass.
        "GatherElements" => out.saturating_add(count(&inputs[1]).saturating_mul(2)),
        // The index, and for an assignment the mask of written positions.
        "ScatterElements" => {
            let mask = if matches!(attrs.get("reduction"), Some(Attr::Text(r)) if r == "none") {
                out
            } else {
                0
            };
            out.saturating_add(count(&inputs[1]).saturating_mul(2))
                .saturating_add(mask)
        }
        // The filled tensor into which the input is copied.
        "Pad" => out.saturating_mul(2),
        "Resize" => {
            let x = &inputs[0];
            let (mode, size) = match resize_spec(x, attrs) {
                Ok(v) => v,
                Err(_) => return out,
            };
            // First pass along the last axis: [.., H, W'] (2D) or directly the output (1D).
            let first: u64 = if size.len() == 2 {
                x[..x.len() - 1]
                    .iter()
                    .map(|d| d.unwrap_or(1))
                    .product::<u64>()
                    * size[1] as u64
            } else {
                out
            };
            match mode {
                ResizeMode::Nearest if size.len() == 2 => out.saturating_add(first),
                ResizeMode::Nearest => out,
                // Two selections, two products and a sum per axis.
                ResizeMode::Linear { .. } if size.len() == 2 => first
                    .saturating_mul(5)
                    .saturating_add(out.saturating_mul(5)),
                ResizeMode::Linear { .. } => out.saturating_mul(5),
            }
        }
        _ => out,
    }
}

/// Size of the axis named by the `axis` attribute (1 if unknown).
fn axis_size(x: &Shape, attrs: &BTreeMap<String, Attr>) -> u64 {
    let r = x.len() as i64;
    let a = attrs.get("axis").and_then(Attr::as_int).unwrap_or(0);
    let a = if a < 0 { a + r } else { a };
    x.get(a.max(0) as usize).copied().flatten().unwrap_or(1)
}

/// FLOPs of a node for **one** example, from the output shape and the inputs.
pub fn flops(op: &str, inputs: &[Shape], output: &Shape, attrs: &BTreeMap<String, Attr>) -> u64 {
    let count = |s: &Shape| -> u64 { s.iter().map(|d| d.unwrap_or(1)).product() };
    match op {
        "Formula" => {
            formula_of(attrs)
                .map(|e| e.flops_per_element())
                .unwrap_or(1)
                .max(1)
                * count(output)
        }
        "MatMul" => {
            let k = inputs[0].last().copied().flatten().unwrap_or(1);
            2 * count(output) * k
        }
        "Conv2d" => {
            let w = &inputs[1];
            let per_output = w[1].unwrap_or(1) * w[2].unwrap_or(1) * w[3].unwrap_or(1);
            2 * count(output) * per_output
        }
        "ConvTranspose2d" => {
            // Each input element is multiplied by a kernel slice [C_out/g, KH, KW].
            let w = &inputs[1];
            let per_input = w[1].unwrap_or(1) * w[2].unwrap_or(1) * w[3].unwrap_or(1);
            2 * count(&inputs[0]) * per_input
        }
        // One operation per element read; comparisons, mask and selection for the extrema.
        "ReduceSum" | "ReduceProd" => count(&inputs[0]),
        "ReduceMax" | "ReduceMin" => 3 * count(&inputs[0]),
        "CumSum" => count(output),
        "IndexAdd" => count(&inputs[1]),
        "Resize" => match resize_spec(&inputs[0], attrs) {
            // Two products and a sum per interpolated axis.
            Ok((ResizeMode::Linear { .. }, size)) => 3 * size.len() as u64 * count(output),
            _ => 0,
        },
        // Copies.
        "Pad" | "Flip" | "IndexSelect" => 0,
        // ADR-0029. Comparisons, mask and integer minimum.
        "ArgMax" | "ArgMin" => 3 * count(&inputs[0]),
        // A comparison sort of each line: n·⌈log₂ n⌉.
        "ArgTopK" | "ArgSort" => {
            let n = axis_size(&inputs[0], attrs).max(2);
            count(&inputs[0]) * (64 - (n - 1).leading_zeros() as u64)
        }
        "GatherElements" | "Cast" => count(output),
        "ScatterElements" => {
            let per = if matches!(attrs.get("reduction"), Some(Attr::Text(r)) if r == "none") {
                2
            } else {
                1
            };
            per * count(&inputs[1])
        }
        "LayerNorm" | "RMSNorm" | "Softmax" => 5 * count(output),
        "Gelu" | "Sigmoid" | "Tanh" | "Rsqrt" => 4 * count(output),
        // One addition per element read.
        "ReduceMean" => count(&inputs[0]),
        "BatchNorm2d" => 4 * count(output),
        "MaxPool2d" | "AvgPool2d" => count(output),
        // One comparison and one selection per element; no backward cost beyond a mask.
        "CausalMask" => count(output),
        // A copy.
        "Slice" => 0,
        "Reshape" | "Transpose" | "Flatten" | "Concat" | "Gather" | "Dropout" => 0,
        _ => count(output),
    }
}
