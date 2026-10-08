//! Tiling a matrix too large for a single machine (gg-server/docs/ROADMAP.md P11 D3, ADR-0019).
//!
//! The pass rewrites the description: a weight `W` becomes tiles `W.tile{i}of{T}`, each a separate
//! parameter, and the node that consumed it becomes `T` nodes joined back together by `Concat`. It
//! is the same computation, expressed with the profile's already verified operators. Since each
//! tile is a distinct node, the stage split (D2) can place them on different machines: a matrix
//! that no single card can hold is then spread across several.
//!
//! Three patterns are recognized, those of a language model:
//!
//! - `MatMul(x, W)`, `W` of shape `[input, output]`: tiles of **columns**;
//! - `MatMul(x, Transpose(W))`, `W` of shape `[output, input]` (an `nn.Linear` captured by the
//!   PyTorch driver): tiles of **rows** of `W`, hence of columns of the output;
//! - `Gather(W, ids)`, an embedding table `[vocabulary, width]`: tiles of **columns**, the width
//!   joined back together.
//!
//! In all three cases the output is split along its last axis and joined back by `Concat`, in
//! groups of at most eight. A weight consumed by several nodes (tied weights) is rejected with an
//! explicit error.

use std::collections::BTreeMap;

use gg_protocol::tensor_file::TensorF32;
use serde::{Deserialize, Serialize};

use super::ops::Attr;
use super::{GraphReport, ModelGraphV1, Node, ParamSpec};

/// Name of tile `i` out of `tiles` of a parameter.
pub fn tile_name(param: &str, i: usize, tiles: usize) -> String {
    format!("{param}.tile{i}of{tiles}")
}

/// A tiled parameter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tiled {
    /// Original parameter.
    pub param: String,
    /// Number of tiles.
    pub tiles: usize,
    /// Axis of the parameter along which it is split.
    pub axis: usize,
}

/// Sizes of `tiles` tiles of an axis of `size` elements; the first ones are larger by at most
/// one.
pub fn tile_sizes(size: u64, tiles: usize) -> Vec<u64> {
    let t = tiles as u64;
    (0..t).map(|i| size / t + u64::from(i < size % t)).collect()
}

fn rank_of(report: &GraphReport, tensor: &str) -> Option<usize> {
    report
        .shapes
        .get(tensor)
        .map(|s| s.trim_matches(|c| c == '[' || c == ']').split(',').count())
}

fn node(op: &str, inputs: Vec<String>, output: String, attrs: &[(&str, Attr)]) -> Node {
    Node {
        op: op.to_owned(),
        inputs,
        output,
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    }
}

/// `Concat` of `parts` along `axis`, in groups of at most eight, into `output`.
fn concat_into(nodes: &mut Vec<Node>, parts: Vec<String>, axis: usize, output: &str) {
    let mut level = parts;
    let mut round = 0;
    while level.len() > 8 {
        let mut next = Vec::new();
        for (g, chunk) in level.chunks(8).enumerate() {
            if chunk.len() == 1 {
                next.push(chunk[0].clone());
                continue;
            }
            let out = format!("{output}.cat{round}.{g}");
            nodes.push(node(
                "Concat",
                chunk.to_vec(),
                out.clone(),
                &[("axis", Attr::Int(axis as i64))],
            ));
            next.push(out);
        }
        level = next;
        round += 1;
    }
    nodes.push(node(
        "Concat",
        level,
        output.to_owned(),
        &[("axis", Attr::Int(axis as i64))],
    ));
}

/// Splits `param` into `tiles` tiles. Returns the rewritten description and what was done.
pub fn tile(
    graph: &ModelGraphV1,
    report: &GraphReport,
    param: &str,
    tiles: usize,
) -> Result<(ModelGraphV1, Tiled), String> {
    let spec = graph
        .parameters
        .iter()
        .find(|p| p.name == param)
        .ok_or_else(|| format!("unknown parameter '{param}'"))?;
    if spec.shape.len() != 2 {
        return Err(format!(
            "'{param}' has rank {}, only matrices can be tiled",
            spec.shape.len()
        ));
    }
    let consumers: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.inputs.iter().any(|i| i == param))
        .map(|(i, _)| i)
        .collect();
    if consumers.len() != 1 {
        return Err(format!(
            "'{param}' is consumed by {} nodes; a tied weight cannot be tiled",
            consumers.len()
        ));
    }
    let c = &graph.nodes[consumers[0]];
    // Replaced nodes, the parameter's axis, and the node whose output is split.
    let (removed, axis, final_node): (Vec<usize>, usize, &Node) = match c.op.as_str() {
        "MatMul" if c.inputs[1] == param => (vec![consumers[0]], 1, c),
        "Gather" if c.inputs[0] == param => (vec![consumers[0]], 1, c),
        "Transpose" => {
            let perm = c.attrs.get("perm").and_then(Attr::as_ints);
            if perm != Some(&[1, 0][..]) {
                return Err(format!(
                    "'{param}' goes through a transpose other than [1, 0]"
                ));
            }
            let users: Vec<usize> = graph
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.inputs.iter().any(|i| i == &c.output))
                .map(|(i, _)| i)
                .collect();
            match users.as_slice() {
                [u] if graph.nodes[*u].op == "MatMul" && graph.nodes[*u].inputs[1] == c.output => {
                    (vec![consumers[0], *u], 0, &graph.nodes[*u])
                }
                _ => {
                    return Err(format!(
                        "the transpose of '{param}' is not the second operand of a single MatMul"
                    ));
                }
            }
        }
        other => {
            return Err(format!(
                "'{param}' is consumed by {other}; only MatMul, Transpose then MatMul, and \
Gather can be tiled"
            ));
        }
    };
    let size = spec.shape[axis];
    if tiles < 2 || tiles as u64 > size {
        return Err(format!(
            "{tiles} tiles for an axis of {size}: must be 2 to {size}"
        ));
    }
    let out = final_node.output.clone();
    let out_rank =
        rank_of(report, &out).ok_or_else(|| format!("shape of '{out}' missing from the report"))?;
    let sizes = tile_sizes(size, tiles);
    let mut new_params: Vec<ParamSpec> = Vec::new();
    let mut new_nodes: Vec<Node> = Vec::new();
    let mut parts = Vec::new();
    for (i, s) in sizes.iter().enumerate() {
        let name = tile_name(param, i, tiles);
        let mut shape = spec.shape.clone();
        shape[axis] = *s;
        new_params.push(ParamSpec {
            name: name.clone(),
            shape,
            init: spec.init.clone(),
            trainable: spec.trainable,
        });
        let part = format!("{out}.tile{i}of{tiles}");
        match c.op.as_str() {
            "MatMul" => new_nodes.push(node(
                "MatMul",
                vec![c.inputs[0].clone(), name],
                part.clone(),
                &[],
            )),
            "Gather" => {
                let mut n = c.clone();
                n.inputs[0] = name;
                n.output = part.clone();
                new_nodes.push(n);
            }
            _ => {
                let t = format!("{}.tile{i}of{tiles}", c.output);
                new_nodes.push(node(
                    "Transpose",
                    vec![name],
                    t.clone(),
                    &[("perm", Attr::Ints(vec![1, 0]))],
                ));
                new_nodes.push(node(
                    "MatMul",
                    vec![final_node.inputs[0].clone(), t],
                    part.clone(),
                    &[],
                ));
            }
        }
        parts.push(part);
    }
    concat_into(&mut new_nodes, parts, out_rank - 1, &out);

    let mut g = graph.clone();
    let pos = g
        .parameters
        .iter()
        .position(|p| p.name == param)
        .expect("found");
    g.parameters.splice(pos..=pos, new_params);
    let first = *removed.iter().min().expect("non-empty");
    let mut nodes: Vec<Node> = Vec::with_capacity(g.nodes.len() + new_nodes.len());
    for (i, n) in g.nodes.iter().enumerate() {
        if i == first {
            nodes.extend(new_nodes.iter().cloned());
        }
        if !removed.contains(&i) {
            nodes.push(n.clone());
        }
    }
    g.nodes = nodes;
    Ok((
        g,
        Tiled {
            param: param.to_owned(),
            tiles,
            axis,
        },
    ))
}

/// Tiles every matrix larger than `max_param_bytes` bytes into as many tiles as needed.
pub fn auto_tile(
    graph: &ModelGraphV1,
    max_param_bytes: u64,
) -> Result<(ModelGraphV1, Vec<Tiled>), String> {
    let mut g = graph.clone();
    let mut done = Vec::new();
    let big: Vec<(String, u64, Vec<u64>)> = graph
        .parameters
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                p.shape.iter().product::<u64>() * 4,
                p.shape.clone(),
            )
        })
        .filter(|(_, b, _)| *b > max_param_bytes)
        .collect();
    for (name, bytes, _) in big {
        let tiles = bytes.div_ceil(max_param_bytes.max(1)) as usize;
        let report = super::validate(&g).map_err(|e| e.to_string())?;
        let (next, t) = tile(&g, &report, &name, tiles)?;
        super::validate(&next).map_err(|e| format!("after tiling '{name}': {e}"))?;
        g = next;
        done.push(t);
    }
    Ok((g, done))
}

/// Applies a tiling already decided (by [`auto_tile`], elsewhere) to `graph`, in its order: the
/// description it gives is the one `auto_tile` gave, so that a tiled description can be checked
/// against its source and its tiling.
pub fn apply(graph: &ModelGraphV1, tiled: &[Tiled]) -> Result<ModelGraphV1, String> {
    let mut g = graph.clone();
    for t in tiled {
        let report = super::validate(&g).map_err(|e| e.to_string())?;
        let (next, done) = tile(&g, &report, &t.param, t.tiles)?;
        if done != *t {
            return Err(format!(
                "'{}' tiles along axis {}, not {}",
                t.param, done.axis, t.axis
            ));
        }
        super::validate(&next).map_err(|e| format!("after tiling '{}': {e}", t.param))?;
        g = next;
    }
    Ok(g)
}

/// Splits the weight values the same way the description was split.
pub fn split_weights(
    weights: &BTreeMap<String, TensorF32>,
    tiled: &[Tiled],
) -> Result<BTreeMap<String, TensorF32>, String> {
    let mut out = weights.clone();
    for t in tiled {
        let w = out
            .remove(&t.param)
            .ok_or_else(|| format!("missing weight '{}'", t.param))?;
        let (rows, cols) = (w.shape[0] as usize, w.shape[1] as usize);
        let sizes = tile_sizes(w.shape[t.axis], t.tiles);
        let mut offset = 0usize;
        for (i, s) in sizes.iter().enumerate() {
            let s = *s as usize;
            let (shape, data): (Vec<u64>, Vec<f32>) = if t.axis == 0 {
                (
                    vec![s as u64, cols as u64],
                    w.data[offset * cols..(offset + s) * cols].to_vec(),
                )
            } else {
                (
                    vec![rows as u64, s as u64],
                    (0..rows)
                        .flat_map(|r| {
                            w.data[r * cols + offset..r * cols + offset + s]
                                .iter()
                                .copied()
                        })
                        .collect(),
                )
            };
            out.insert(
                tile_name(&t.param, i, t.tiles),
                TensorF32::new(shape, data).map_err(|e| e.to_string())?,
            );
            offset += s;
        }
    }
    Ok(out)
}

/// Joins tiles back into the original weights (the inverse of [`split_weights`]).
pub fn merge_weights(
    weights: &BTreeMap<String, TensorF32>,
    tiled: &[Tiled],
) -> Result<BTreeMap<String, TensorF32>, String> {
    let mut out = weights.clone();
    for t in tiled.iter().rev() {
        let parts: Vec<TensorF32> = (0..t.tiles)
            .map(|i| {
                out.remove(&tile_name(&t.param, i, t.tiles))
                    .ok_or_else(|| format!("missing tile {i} of '{}'", t.param))
            })
            .collect::<Result<_, _>>()?;
        let merged = if t.axis == 0 {
            let cols = parts[0].shape[1];
            let rows: u64 = parts.iter().map(|p| p.shape[0]).sum();
            TensorF32::new(
                vec![rows, cols],
                parts.iter().flat_map(|p| p.data.iter().copied()).collect(),
            )
        } else {
            let rows = parts[0].shape[0] as usize;
            let cols: u64 = parts.iter().map(|p| p.shape[1]).sum();
            let mut data = Vec::with_capacity(rows * cols as usize);
            for r in 0..rows {
                for p in &parts {
                    let c = p.shape[1] as usize;
                    data.extend_from_slice(&p.data[r * c..(r + 1) * c]);
                }
            }
            TensorF32::new(vec![rows as u64, cols], data)
        }
        .map_err(|e| e.to_string())?;
        out.insert(t.param.clone(), merged);
    }
    Ok(out)
}
