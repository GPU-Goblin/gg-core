//! Operators added with `gg-ops/7` (ADR-0027): transposed convolution, padding, resizing, flips,
//! reductions other than the mean, cumulative sum, selection and accumulation by constant indices.
//!
//! Each one is written with Burn operations whose derivative Burn provides, on the CPU as on the
//! GPU; where Burn's own operator does not do what PyTorch does (the GPU has no bilinear backward
//! pass, `max_dim` gives the gradient to a single maximum), it is composed here from operations
//! that do, and the comment says why.

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::module::conv_transpose2d;
use burn::tensor::ops::{ConvTransposeOptions, PadMode as BurnPad};
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData};

use super::super::ops::{self, Attr, PadMode, ResizeMode, Shape};
use super::{
    Dyn, ExecError, as_f1, as_f4, attr_int, attr_pair, dtype_err, norm_axis, reshape_dyn,
    reshape_to, shape_err,
};

/// The concrete shape of an executed tensor, in the form validation uses (every axis known).
fn known(dims: &[usize]) -> Shape {
    dims.iter().map(|d| Some(*d as u64)).collect()
}

fn op_err(e: ops::OpError) -> ExecError {
    shape_err(e.0)
}

/// Removes the reduced axis when `keepdims = 0`.
fn drop_axis<B: Backend>(
    t: Dyn<B>,
    dims: Vec<usize>,
    axis: usize,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    if attr_int(attrs, "keepdims", 1) == 0 {
        let mut target = dims;
        target.remove(axis);
        reshape_dyn(t, &target)
    } else {
        Ok(t)
    }
}

/// `ConvTranspose2d`: kernel `[C_in, C_out/g, KH, KW]`, as `torch.nn.ConvTranspose2d`.
pub(super) fn conv_transpose<B: Backend>(
    x: Dyn<B>,
    w: Dyn<B>,
    bias: Option<Dyn<B>>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let x = as_f4(x, "ConvTranspose2d")?;
    let w = as_f4(w, "ConvTranspose2d (kernel)")?;
    let bias = bias
        .map(|b| as_f1(b, "ConvTranspose2d (bias)"))
        .transpose()?;
    let stride = attr_pair(attrs, "stride", [1, 1])?;
    let padding = attr_pair(attrs, "padding", [0, 0])?;
    let padding_out = attr_pair(attrs, "output_padding", [0, 0])?;
    let dilation = attr_pair(attrs, "dilation", [1, 1])?;
    let groups = usize::try_from(attr_int(attrs, "groups", 1).max(1))
        .map_err(|_| shape_err("invalid groups"))?;
    let [_, cin, _, _] = x.dims();
    let [wc, _, _, _] = w.dims();
    if cin != wc || !cin.is_multiple_of(groups) || stride.contains(&0) || dilation.contains(&0) {
        return Err(shape_err(format!(
            "ConvTranspose2d: input {:?} and kernel {:?} do not match (groups {groups})",
            x.dims(),
            w.dims()
        )));
    }
    let [_, _, kh, kw] = w.dims();
    let kernel = [kh, kw];
    let short = |a: usize| dilation[a] == 1 && kernel[a] < stride[a];
    if !short(0) && !short(1) {
        return Ok(Dyn::F4(conv_transpose2d(
            x,
            w,
            bias,
            ConvTransposeOptions::new(stride, padding, padding_out, dilation, groups),
        )));
    }
    // A kernel shorter than its stride: on WGPU/Vulkan, Burn 0.21 computes this transposed
    // convolution wrongly, forward included (see `conv2d`). Rewritten exactly: the transposed
    // convolution with stride equal to the kernel lays each input position's `k` outputs side by
    // side; zeros then widen each block to `stride`, and the padding crops the result as PyTorch
    // does. The bias is added last, since the inserted positions receive it too.
    let input = [x.dims()[2], x.dims()[3]];
    let (mut st, mut pad, mut pad_out) = (stride, padding, padding_out);
    for a in 0..2 {
        if short(a) {
            (st[a], pad[a], pad_out[a]) = (kernel[a], 0, 0);
        }
    }
    let mut y = conv_transpose2d(
        x,
        w,
        None,
        ConvTransposeOptions::new(st, pad, pad_out, dilation, groups),
    );
    for a in 0..2 {
        if !short(a) {
            continue;
        }
        let (l, k, s, p, po) = (input[a], kernel[a], stride[a], padding[a], padding_out[a]);
        let [n, c, h, wd] = y.dims();
        let mut pairs = vec![(0, 0); 5];
        pairs[3 + a] = (0, s - k);
        y = if a == 0 {
            y.reshape([n, c, l, k, wd])
                .pad(pairs, BurnPad::Constant(0.0))
                .reshape([n, c, l * s, wd])
        } else {
            y.reshape([n, c, h, l, k])
                .pad(pairs, BurnPad::Constant(0.0))
                .reshape([n, c, h, l * s])
        };
        // PyTorch's length: (l − 1)·s − 2p + k + output_padding, read from position p.
        let out = ((l - 1) * s + k + po)
            .checked_sub(2 * p)
            .ok_or_else(|| shape_err("ConvTranspose2d: padding larger than the output"))?;
        let end = p + out;
        let mut pairs = vec![(0, 0); 4];
        pairs[2 + a] = (0, end.saturating_sub(l * s));
        y = y.pad(pairs, BurnPad::Constant(0.0));
        let [n, c, h, wd] = y.dims();
        y = if a == 0 {
            y.slice([0..n, 0..c, p..end, 0..wd])
        } else {
            y.slice([0..n, 0..c, 0..h, p..end])
        };
    }
    if let Some(b) = bias {
        let c = b.dims()[0];
        y = y + b.reshape([1, c, 1, 1]);
    }
    Ok(Dyn::F4(y))
}

/// `Pad`: constant, reflection or repetition of the edge, per axis.
pub(super) fn pad<B: Backend>(
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let (mode, pairs) = ops::pad_spec(&known(&x.dims()), attrs).map_err(op_err)?;
    if pairs.iter().all(|(b, e)| *b == 0 && *e == 0) {
        return Ok(x);
    }
    let mode = match mode {
        PadMode::Constant(v) => BurnPad::Constant(v),
        PadMode::Reflect => BurnPad::Reflect,
        PadMode::Edge => BurnPad::Edge,
    };
    Ok(map_float!(x, t => t.pad(pairs.clone(), mode), "Pad"))
}

/// A 2-D convolution whose kernel is shorter than its stride along a spatial axis (the 1×1,
/// stride-2 shortcut of a ResNet), rewritten into an exactly equivalent one whose stride equals its
/// kernel: the padded input is cut into windows of `stride` positions, of which only the first
/// `kernel` are ever read, and those are kept. Other axes are left as they are.
///
/// Why: on WGPU/Vulkan, Burn 0.21 computes a wrong gradient with respect to the input of a
/// convolution whose kernel is shorter than its stride (every such case measured: 1/2, 1/3, 2/3,
/// 3/4; 100 % relative error), while stride equal to or shorter than the kernel is exact. A ResNet18
/// trained on a GPU worker was rejected by the validator for it. The rewrite runs on every engine:
/// one code path, the same function.
pub(super) fn conv2d<B: Backend>(
    x: Tensor<B, 4>,
    w: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Tensor<B, 4> {
    use burn::tensor::module::conv2d as burn_conv2d;
    use burn::tensor::ops::ConvOptions;
    let [_, _, kh, kw] = w.dims();
    let kernel = [kh, kw];
    let short = |a: usize| dilation[a] == 1 && kernel[a] < stride[a];
    if !short(0) && !short(1) {
        return burn_conv2d(
            x,
            w,
            bias,
            ConvOptions::new(stride, padding, dilation, groups),
        );
    }
    let mut x = x;
    let (mut stride, mut padding) = (stride, padding);
    for a in 0..2 {
        if !short(a) {
            continue;
        }
        let (k, st, p) = (kernel[a], stride[a], padding[a]);
        let len = x.dims()[2 + a] + 2 * p;
        if len < k {
            // Refused by validation; left to the convolution to report.
            continue;
        }
        let out = (len - k) / st + 1;
        let need = out * st;
        // Pad, then complete or cut to `out` whole windows (the positions added past the end are
        // never read: they lie beyond the first `k` of the last window).
        let mut pairs = vec![(0, 0); 4];
        pairs[2 + a] = (p, p + need.saturating_sub(len));
        x = x.pad(pairs, BurnPad::Constant(0.0));
        let [n, c, h, wd] = x.dims();
        x = if a == 0 {
            x.slice([0..n, 0..c, 0..need, 0..wd])
                .reshape([n, c, out, st, wd])
                .slice([0..n, 0..c, 0..out, 0..k, 0..wd])
                .reshape([n, c, out * k, wd])
        } else {
            x.slice([0..n, 0..c, 0..h, 0..need])
                .reshape([n, c, h, out, st])
                .slice([0..n, 0..c, 0..h, 0..out, 0..k])
                .reshape([n, c, h, out * k])
        };
        stride[a] = k;
        padding[a] = 0;
    }
    burn_conv2d(
        x,
        w,
        bias,
        ConvOptions::new(stride, padding, dilation, groups),
    )
}

/// Source indices and weights of a linear interpolation along one axis, computed in f32 as
/// PyTorch does (`area_pixel_compute_source_index`, then `guard_index_and_lambda`): the lower
/// index, the upper one (the same at the last element) and the weights `(1 − λ, λ)`.
fn linear_taps(
    input: usize,
    output: usize,
    align_corners: bool,
) -> (Vec<i64>, Vec<i64>, Vec<f32>, Vec<f32>) {
    let scale = if align_corners {
        if output > 1 {
            (input as f32 - 1.0) / (output as f32 - 1.0)
        } else {
            0.0
        }
    } else {
        input as f32 / output as f32
    };
    let mut lo = Vec::with_capacity(output);
    let mut hi = Vec::with_capacity(output);
    let mut w0 = Vec::with_capacity(output);
    let mut w1 = Vec::with_capacity(output);
    for o in 0..output {
        let src = if align_corners {
            scale * o as f32
        } else {
            (scale * (o as f32 + 0.5) - 0.5).max(0.0)
        };
        let i0 = (src.floor() as usize).min(input - 1);
        let lambda = (src - i0 as f32).clamp(0.0, 1.0);
        lo.push(i0 as i64);
        hi.push((i0 + usize::from(i0 < input - 1)) as i64);
        w0.push(1.0 - lambda);
        w1.push(lambda);
    }
    (lo, hi, w0, w1)
}

/// Source index of each output position for `nearest`, as PyTorch computes it:
/// `min(floor(dst · (in/out)), in − 1)`, the ratio in f32.
fn nearest_taps(input: usize, output: usize) -> Vec<i64> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|o| ((o as f32 * scale).floor() as i64).min(input as i64 - 1))
        .collect()
}

fn int_vec<B: Backend>(device: &B::Device, v: Vec<i64>) -> Tensor<B, 1, Int> {
    let n = v.len();
    Tensor::<B, 1, Int>::from_data(TensorData::new(v, [n]), device)
}

/// A vector of weights shaped to broadcast along `axis` of a rank-`rank` tensor.
fn weights_along<B: Backend>(
    device: &B::Device,
    w: Vec<f32>,
    axis: usize,
    rank: usize,
) -> Result<Dyn<B>, ExecError> {
    let n = w.len();
    let mut dims = vec![1usize; rank];
    dims[axis] = n;
    reshape_to(
        Tensor::<B, 1>::from_data(TensorData::new(w, [n]), device),
        &dims,
    )
}

fn resize_axis<B: Backend>(
    device: &B::Device,
    x: Dyn<B>,
    axis: usize,
    output: usize,
    mode: ResizeMode,
) -> Result<Dyn<B>, ExecError> {
    let dims = x.dims();
    let input = dims[axis];
    if input == output {
        // Nearest: every index maps to itself; linear: λ = 0 everywhere. Nothing to compute.
        return Ok(x);
    }
    match mode {
        ResizeMode::Nearest => {
            let idx = int_vec::<B>(device, nearest_taps(input, output));
            Ok(map_float!(x, t => t.select(axis, idx.clone()), "Resize"))
        }
        ResizeMode::Linear { align_corners } => {
            let (lo, hi, w0, w1) = linear_taps(input, output, align_corners);
            let (lo, hi) = (int_vec::<B>(device, lo), int_vec::<B>(device, hi));
            let rank = dims.len();
            let a = map_float!(x.clone(), t => t.select(axis, lo.clone()), "Resize");
            let b = map_float!(x, t => t.select(axis, hi.clone()), "Resize");
            // `x[lo]·(1 − λ) + x[hi]·λ`, in that order, as PyTorch's kernel.
            let a = super::binary("Mul", a, weights_along(device, w0, axis, rank)?)?;
            let b = super::binary("Mul", b, weights_along(device, w1, axis, rank)?)?;
            super::binary("Add", a, b)
        }
    }
}

/// `Resize`: the last axis, then the one before it for a 2D resize (PyTorch interpolates along
/// the width inside the height, and the arithmetic is the same). Written with selections by index,
/// whose derivative is an accumulation: Burn's GPU backend has no backward pass for its bilinear
/// interpolation.
pub(super) fn resize<B: Backend>(
    device: &B::Device,
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let dims = x.dims();
    let (mode, size) = ops::resize_spec(&known(&dims), attrs).map_err(op_err)?;
    let rank = dims.len();
    let mut out = x;
    for (k, target) in size.iter().enumerate().rev() {
        out = resize_axis(device, out, rank - size.len() + k, *target, mode)?;
    }
    Ok(out)
}

/// `Flip` along each listed axis.
pub(super) fn flip<B: Backend>(
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let axes = attrs
        .get("axes")
        .and_then(Attr::as_ints)
        .ok_or_else(|| shape_err("Flip: missing axes attribute"))?;
    let rank = x.dims().len();
    let mut out = x;
    for a in axes {
        let a = norm_axis(*a, rank)? as isize;
        out = map_float!(out, t => t.flip([a]), "Flip");
    }
    Ok(out)
}

/// Extremum along an axis, with the gradient shared equally between tied extrema (as
/// `torch.amax`): `m + Σ((x − m)·mask) / Σ mask`, where `m` is the detached extremum and `mask`
/// marks the elements equal to it. The value is exactly `m` (the sum is of zeros); the derivative
/// is `mask / Σ mask`.
fn extremum<B: Backend, const D: usize>(t: Tensor<B, D>, axis: usize, max: bool) -> Tensor<B, D> {
    let frozen = t.clone().detach().set_require_grad(false);
    let m = if max {
        frozen.max_dim(axis)
    } else {
        frozen.min_dim(axis)
    };
    let full = m.clone().expand(t.dims());
    let mask = t.clone().equal(full.clone()).float();
    let count = mask.clone().sum_dim(axis);
    let zero = t.sub(full).mul(mask).sum_dim(axis).div(count);
    m.add(zero)
}

/// Product along an axis by pairwise products (`x[:n/2]·x[n/2:]`, repeated): the automatic
/// derivative stays exact with zeros, unlike `exp(Σ log x)`, and the depth is `log₂ n`.
fn product<B: Backend, const D: usize>(t: Tensor<B, D>, axis: usize) -> Tensor<B, D> {
    let mut cur = t;
    loop {
        let n = cur.dims()[axis];
        if n <= 1 {
            return cur;
        }
        let half = n / 2;
        let p = cur
            .clone()
            .narrow(axis, 0, half)
            .mul(cur.clone().narrow(axis, half, half));
        cur = if n % 2 == 1 {
            Tensor::cat(vec![p, cur.narrow(axis, 2 * half, 1)], axis)
        } else {
            p
        };
    }
}

/// `ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProd`.
pub(super) fn reduce<B: Backend>(
    op: &str,
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let dims = x.dims();
    let axis = norm_axis(attr_int(attrs, "axis", -1), dims.len())?;
    let reduced = match op {
        "ReduceSum" => map_float!(x, t => t.sum_dim(axis), op),
        "ReduceMax" => map_float!(x, t => extremum(t, axis, true), op),
        "ReduceMin" => map_float!(x, t => extremum(t, axis, false), op),
        "ReduceProd" => map_float!(x, t => product(t, axis), op),
        other => return Err(shape_err(format!("unknown reduction {other}"))),
    };
    drop_axis(reduced, dims, axis, attrs)
}

/// `CumSum`: inclusive cumulative sum.
pub(super) fn cumsum<B: Backend>(
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let axis = norm_axis(attr_int(attrs, "axis", 0), x.dims().len())?;
    Ok(map_float!(x, t => t.cumsum(axis), "CumSum"))
}

/// Axis and indices of `IndexSelect` / `IndexAdd`, checked again against the executed shape.
fn index_of<B: Backend>(
    device: &B::Device,
    op: &str,
    dims: &[usize],
    attrs: &BTreeMap<String, Attr>,
) -> Result<(usize, Tensor<B, 1, Int>, usize), ExecError> {
    let axis = norm_axis(attr_int(attrs, "axis", 0), dims.len())?;
    let index = attrs
        .get("index")
        .and_then(Attr::as_ints)
        .ok_or_else(|| shape_err(format!("{op}: missing index attribute")))?;
    let size = dims[axis] as i64;
    if index.is_empty() || index.iter().any(|v| *v < 0 || *v >= size) {
        return Err(shape_err(format!(
            "{op}: an index lies outside axis {axis} of size {size}"
        )));
    }
    Ok((axis, int_vec::<B>(device, index.to_vec()), index.len()))
}

/// `IndexSelect`: the slices listed in `index`, along `axis`.
pub(super) fn index_select<B: Backend>(
    device: &B::Device,
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let (axis, idx, _) = index_of::<B>(device, "IndexSelect", &x.dims(), attrs)?;
    Ok(map_float!(x, t => t.select(axis, idx.clone()), "IndexSelect"))
}

/// `IndexAdd`: `x` plus `src` accumulated at the positions listed in `index`, along `axis`, in
/// the order of the list (on the GPU too: one thread per output line runs through the list).
pub(super) fn index_add<B: Backend>(
    device: &B::Device,
    x: Dyn<B>,
    src: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let dims = x.dims();
    let (axis, idx, n) = index_of::<B>(device, "IndexAdd", &dims, attrs)?;
    let mut want = dims.clone();
    want[axis] = n;
    if src.dims() != want {
        return Err(shape_err(format!(
            "IndexAdd: src {:?}, expected {want:?}",
            src.dims()
        )));
    }
    if !src.is_float() {
        return Err(dtype_err("IndexAdd expects float tensors"));
    }
    Ok(
        zip_float!(x, src, a, b => a.select_assign(axis, idx.clone(), b, IndexingUpdateOp::Add), "IndexAdd"),
    )
}
