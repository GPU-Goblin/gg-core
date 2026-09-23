//! Integer tensors computed inside the graph (ADR-0029): `ArgMax`, `ArgMin`, `ArgTopK`,
//! `ArgSort`, `Cast`, `GatherElements`, `ScatterElements`.
//!
//! The choices are computed so that they cannot depend on the engine: the first index among ties
//! for `ArgMax`/`ArgMin` (an integer minimum over the mask of the extremum, exact whatever the
//! reduction order), a stable sort with an explicit comparison on the host for `ArgTopK`/`ArgSort`
//! (Burn's sorts are unstable, and its GPU backend sorts on the host anyway). A NaN counts as the
//! largest value, as in PyTorch. Integers carry no gradient.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData};

use super::super::ops::Attr;
use super::{Dyn, ExecError, attr_int, dtype_err, host_of, norm_axis, reshape_dyn, shape_err};

/// Index of the first extremum along `axis`, kept with size 1. The input is detached: an index
/// has no derivative, and nothing is recorded on the tape.
fn first_extremum<B: Backend, const D: usize>(
    t: Tensor<B, D>,
    axis: usize,
    max: bool,
) -> Tensor<B, D, Int> {
    let t = t.detach().set_require_grad(false);
    let dims = t.dims();
    let n = dims[axis];
    let m = if max {
        t.clone().max_dim(axis)
    } else {
        t.clone().min_dim(axis)
    };
    let hit = t.clone().equal(m.expand(dims));
    // A line with a NaN: its NaNs are the extremum (PyTorch's convention).
    let nan = t.clone().is_nan();
    let has_nan = nan.clone().any_dim(axis).expand(dims);
    let mask = hit.mask_where(has_nan, nan);
    let mut shape = [1usize; D];
    shape[axis] = n;
    let iota = Tensor::<B, 1, Int>::arange(0..n as i64, &t.device())
        .reshape(shape)
        .expand(dims);
    iota.mask_fill(mask.bool_not(), n as i64).min_dim(axis)
}

/// `ArgMax` / `ArgMin`.
pub(super) fn arg_extremum<B: Backend>(
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
    max: bool,
) -> Result<Dyn<B>, ExecError> {
    let dims = x.dims();
    let axis = norm_axis(attr_int(attrs, "axis", 0), dims.len())?;
    let out = match x {
        Dyn::F1(t) => Dyn::I1(first_extremum(t, axis, max)),
        Dyn::F2(t) => Dyn::I2(first_extremum(t, axis, max)),
        Dyn::F3(t) => Dyn::I3(first_extremum(t, axis, max)),
        Dyn::F4(t) => Dyn::I4(first_extremum(t, axis, max)),
        _ => return Err(dtype_err("ArgMax/ArgMin expect a float tensor")),
    };
    if attr_int(attrs, "keepdims", 1) == 0 {
        let mut target = dims;
        target.remove(axis);
        reshape_dyn(out, &target)
    } else {
        Ok(out)
    }
}

/// Total order of the sorts: a NaN is larger than everything, two NaNs are equal.
fn order(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).expect("not NaN"),
    }
}

/// For each line along `axis`, the first `k` positions of its stable order (descending or not);
/// ties keep the lower index first. Row-major, the axis cut to `k`.
pub fn line_order(
    values: &[f32],
    dims: &[usize],
    axis: usize,
    descending: bool,
    k: usize,
) -> Vec<i64> {
    let n = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let mut out = vec![0i64; outer * k * inner];
    let mut line: Vec<usize> = Vec::with_capacity(n);
    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| values[(o * n + j) * inner + i];
            line.clear();
            line.extend(0..n);
            line.sort_by(|&a, &b| {
                let c = order(at(a), at(b));
                let c = if descending { c.reverse() } else { c };
                c.then(a.cmp(&b))
            });
            for (j, p) in line.iter().take(k).enumerate() {
                out[(o * k + j) * inner + i] = *p as i64;
            }
        }
    }
    out
}

/// `ArgTopK` (`sort = false`) or `ArgSort` (`sort = true`): computed on the host from the values,
/// one read-back per node, and uploaded as integers.
pub(super) fn arg_order<B: Backend>(
    device: &B::Device,
    x: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
    sort: bool,
) -> Result<Dyn<B>, ExecError> {
    if !x.is_float() {
        return Err(dtype_err("ArgTopK/ArgSort expect a float tensor"));
    }
    let dims = x.dims();
    let axis = norm_axis(attr_int(attrs, "axis", 0), dims.len())?;
    let n = dims[axis];
    let (descending, k) = if sort {
        (attr_int(attrs, "descending", 0) == 1, n)
    } else {
        let k = usize::try_from(attr_int(attrs, "k", 0)).unwrap_or(0);
        if k == 0 || k > n {
            return Err(shape_err(format!("ArgTopK: k = {k} for an axis of {n}")));
        }
        (attr_int(attrs, "largest", 1) == 1, k)
    };
    let host = host_of(x)?;
    let idx = line_order(&host.data, &dims, axis, descending, k);
    let mut out_dims = dims;
    out_dims[axis] = k;
    let len = idx.len();
    let flat = Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [len]), device);
    reshape_dyn(Dyn::I1(flat), &out_dims)
}

/// `Cast`: an integer tensor as f32, exact below 2²⁴; no gradient.
pub(super) fn cast<B: Backend>(x: Dyn<B>) -> Result<Dyn<B>, ExecError> {
    Ok(match x {
        Dyn::I1(t) => Dyn::F1(t.float()),
        Dyn::I2(t) => Dyn::F2(t.float()),
        Dyn::I3(t) => Dyn::F3(t.float()),
        Dyn::I4(t) => Dyn::F4(t.float()),
        _ => return Err(dtype_err("Cast expects an integer tensor")),
    })
}

/// `GatherElements`: `out[…, j, …] = x[…, idx[…, j, …], …]`. `x` is first cut to the index on the
/// other axes (PyTorch reads that prefix), then gathered; the derivative is an accumulation into
/// the selected positions.
pub(super) fn gather_elements<B: Backend>(
    x: Dyn<B>,
    idx: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let (xd, id) = (x.dims(), idx.dims());
    if xd.len() != id.len() || !x.is_float() || idx.is_float() {
        return Err(shape_err(format!(
            "GatherElements: x {xd:?} (float) and index {id:?} (integer) of the same rank expected"
        )));
    }
    let axis = norm_axis(attr_int(attrs, "axis", 0), xd.len())?;
    let mut x = x;
    for a in 0..xd.len() {
        if a != axis && id[a] != xd[a] {
            if id[a] > xd[a] {
                return Err(shape_err(format!(
                    "GatherElements: index {id:?} larger than x {xd:?} on axis {a}"
                )));
            }
            x = map_float!(x, t => t.narrow(a, 0, id[a]), "GatherElements");
        }
    }
    Ok(match (x, idx) {
        (Dyn::F1(t), Dyn::I1(i)) => Dyn::F1(t.gather(axis, i)),
        (Dyn::F2(t), Dyn::I2(i)) => Dyn::F2(t.gather(axis, i)),
        (Dyn::F3(t), Dyn::I3(i)) => Dyn::F3(t.gather(axis, i)),
        (Dyn::F4(t), Dyn::I4(i)) => Dyn::F4(t.gather(axis, i)),
        _ => return Err(shape_err("GatherElements: ranks differ")),
    })
}

fn scatter_add<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    axis: usize,
    idx: Tensor<B, D, Int>,
    src: Tensor<B, D>,
    assign: bool,
) -> Tensor<B, D> {
    if !assign {
        return x.scatter(axis, idx, src, IndexingUpdateOp::Add);
    }
    // Distinct positions (validation): x·(1 − mask) + Σ src, exact; derivative (1 − mask) for x
    // and a gather for src.
    let ones = src.clone().detach().set_require_grad(false).ones_like();
    let zeros = x.clone().detach().set_require_grad(false).zeros_like();
    let mask = zeros
        .clone()
        .scatter(axis, idx.clone(), ones, IndexingUpdateOp::Add);
    let written = zeros.scatter(axis, idx, src, IndexingUpdateOp::Add);
    x.mul(mask.neg().add_scalar(1.0)).add(written)
}

/// `ScatterElements`, `reduction = add` (default) or `none` (assignment of distinct positions).
pub(super) fn scatter_elements<B: Backend>(
    x: Dyn<B>,
    idx: Dyn<B>,
    src: Dyn<B>,
    attrs: &BTreeMap<String, Attr>,
) -> Result<Dyn<B>, ExecError> {
    let (xd, id, sd) = (x.dims(), idx.dims(), src.dims());
    let axis = norm_axis(attr_int(attrs, "axis", 0), xd.len())?;
    if id != sd || xd.len() != id.len() || (0..xd.len()).any(|a| a != axis && xd[a] != id[a]) {
        return Err(shape_err(format!(
            "ScatterElements: x {xd:?}, index {id:?} and src {sd:?} do not match along axis {axis}"
        )));
    }
    let assign = matches!(attrs.get("reduction"), Some(Attr::Text(r)) if r == "none");
    Ok(match (x, idx, src) {
        (Dyn::F1(t), Dyn::I1(i), Dyn::F1(v)) => Dyn::F1(scatter_add(t, axis, i, v, assign)),
        (Dyn::F2(t), Dyn::I2(i), Dyn::F2(v)) => Dyn::F2(scatter_add(t, axis, i, v, assign)),
        (Dyn::F3(t), Dyn::I3(i), Dyn::F3(v)) => Dyn::F3(scatter_add(t, axis, i, v, assign)),
        (Dyn::F4(t), Dyn::I4(i), Dyn::F4(v)) => Dyn::F4(scatter_add(t, axis, i, v, assign)),
        _ => return Err(dtype_err("ScatterElements: float x and src, integer index")),
    })
}
