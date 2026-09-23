//! Profile `gg-ops/7` (ADR-0027): transposed convolution, padding, resizing, flips, reductions,
//! cumulative sum, selection and accumulation by constant indices. Like every added operator, each
//! one brings its oracle, its named rejections, its determinism test, its cost and its memory
//! estimate (ADR-0016).
//!
//! The oracle is written here in f64 from the operator's definition (PyTorch's, index by index),
//! independently of the executor. The derivative is checked against central finite differences of
//! that f64 definition: nothing in the oracle comes from Burn.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, Prepared};
use gg_compute::graph::{self, GraphError, ModelGraphV1};
use gg_protocol::tensor_file::TensorF32;
mod common;
use common::*;

// ---------------------------------------------------------------------------------------------
// f64 definitions, written from PyTorch's documentation.
// ---------------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn conv_transpose_ref(
    x: &T64,
    w: &T64,
    b: Option<&T64>,
    stride: [usize; 2],
    pad: [usize; 2],
    out_pad: [usize; 2],
    dil: [usize; 2],
    groups: usize,
) -> T64 {
    let (n, ci, hh, ww) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let (per, kh, kw) = (w.shape[1], w.shape[2], w.shape[3]);
    let co = per * groups;
    let oh = (hh - 1) * stride[0] + dil[0] * (kh - 1) + out_pad[0] + 1 - 2 * pad[0];
    let ow = (ww - 1) * stride[1] + dil[1] * (kw - 1) + out_pad[1] + 1 - 2 * pad[1];
    let mut out = T64::zeros(&[n, co, oh, ow]);
    let in_per_group = ci / groups;
    each_index(&x.shape, |i| {
        let g = i[1] / in_per_group;
        let v = x.get(i);
        for o in 0..per {
            for a in 0..kh {
                for c in 0..kw {
                    let r = (i[2] * stride[0] + a * dil[0]) as isize - pad[0] as isize;
                    let s = (i[3] * stride[1] + c * dil[1]) as isize - pad[1] as isize;
                    if r >= 0 && s >= 0 && (r as usize) < oh && (s as usize) < ow {
                        out.add_at(
                            &[i[0], g * per + o, r as usize, s as usize],
                            v * w.get(&[i[1], o, a, c]),
                        );
                    }
                }
            }
        }
    });
    if let Some(b) = b {
        let o = out.shape.clone();
        each_index(&o, |i| out.add_at(i, b.data[i[1]]));
    }
    out
}

#[derive(Clone, Copy)]
enum Mode {
    Constant(f64),
    Reflect,
    Edge,
}

fn pad_ref(x: &T64, pads: &[usize], mode: Mode) -> T64 {
    let r = x.shape.len();
    let shape: Vec<usize> = (0..r).map(|a| x.shape[a] + pads[a] + pads[r + a]).collect();
    let mut out = T64::zeros(&shape);
    each_index(&shape, |o| {
        let mut src = Vec::with_capacity(r);
        let mut outside = false;
        for a in 0..r {
            let j = o[a] as isize - pads[a] as isize;
            let size = x.shape[a] as isize;
            let j = match mode {
                _ if (0..size).contains(&j) => j,
                Mode::Constant(_) => {
                    outside = true;
                    0
                }
                Mode::Reflect if j < 0 => -j,
                Mode::Reflect => 2 * (size - 1) - j,
                Mode::Edge => j.clamp(0, size - 1),
            };
            src.push(j as usize);
        }
        let v = match (outside, mode) {
            (true, Mode::Constant(c)) => c,
            _ => x.get(&src),
        };
        let off = out.offset(o);
        out.data[off] = v;
    });
    out
}

/// Resizing along one axis: nearest (`floor(o·in/out)`) or linear (half-pixel or aligned
/// corners), in f64.
fn resize_axis_ref(x: &T64, axis: usize, out: usize, linear: bool, align: bool) -> T64 {
    let input = x.shape[axis];
    let mut shape = x.shape.clone();
    shape[axis] = out;
    let mut y = T64::zeros(&shape);
    each_index(&shape, |o| {
        let d = o[axis] as f64;
        let mut src = o.to_vec();
        let v = if !linear {
            src[axis] = ((d * input as f64 / out as f64).floor() as usize).min(input - 1);
            x.get(&src)
        } else {
            let c = if align {
                if out > 1 {
                    d * (input as f64 - 1.0) / (out as f64 - 1.0)
                } else {
                    0.0
                }
            } else {
                ((d + 0.5) * input as f64 / out as f64 - 0.5).max(0.0)
            };
            let i0 = (c.floor() as usize).min(input - 1);
            let i1 = (i0 + 1).min(input - 1);
            let l = c - i0 as f64;
            src[axis] = i0;
            let a = x.get(&src);
            src[axis] = i1;
            a * (1.0 - l) + x.get(&src) * l
        };
        let off = y.offset(o);
        y.data[off] = v;
    });
    y
}

fn reduce_ref(x: &T64, axis: usize, keep: bool, f: fn(&[f64]) -> f64) -> T64 {
    let mut shape = x.shape.clone();
    shape[axis] = 1;
    let mut out = T64::zeros(&shape);
    each_index(&shape, |o| {
        let mut idx = o.to_vec();
        let vals: Vec<f64> = (0..x.shape[axis])
            .map(|k| {
                idx[axis] = k;
                x.get(&idx)
            })
            .collect();
        let off = out.offset(o);
        out.data[off] = f(&vals);
    });
    if !keep {
        out.shape.remove(axis);
    }
    out
}

fn select_ref(x: &T64, axis: usize, index: &[usize]) -> T64 {
    let mut shape = x.shape.clone();
    shape[axis] = index.len();
    let mut out = T64::zeros(&shape);
    each_index(&shape, |o| {
        let mut src = o.to_vec();
        src[axis] = index[o[axis]];
        let off = out.offset(o);
        out.data[off] = x.get(&src);
    });
    out
}

fn index_add_ref(x: &T64, axis: usize, index: &[usize], src: &T64) -> T64 {
    let mut out = x.clone();
    each_index(&src.shape, |i| {
        let mut dst = i.to_vec();
        dst[axis] = index[i[axis]];
        out.add_at(&dst, src.get(i));
    });
    out
}

fn mul_broadcast(a: &T64, b: &T64) -> T64 {
    // `b` has a leading axis of 1, broadcast over the batch of `a`.
    let per = b.data.len();
    T64 {
        shape: a.shape.clone(),
        data: a
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| v * b.data[i % per])
            .collect(),
    }
}

fn cases() -> Vec<Case> {
    let node = |op: &str, inputs: &str, attrs: &str| {
        format!(r#"{{"op":"{op}","inputs":[{inputs}],"output":"pred","attrs":{attrs}}}"#)
    };
    vec![
        Case {
            label: "ConvTranspose2d stride dilation output_padding bias",
            input: vec![4, 3, 5],
            params: vec![("w", vec![4, 3, 3, 2]), ("b", vec![3])],
            nodes: node(
                "ConvTranspose2d",
                r#""h","w","b""#,
                r#"{"stride":[2,1],"padding":[1,0],"output_padding":[1,0],"dilation":[1,2]}"#,
            ),
            output: vec![3, 6, 7],
            reference: Box::new(|h, p| {
                conv_transpose_ref(h, &p["w"], Some(&p["b"]), [2, 1], [1, 0], [1, 0], [1, 2], 1)
            }),
        },
        Case {
            label: "ConvTranspose2d groups",
            input: vec![4, 3, 3],
            params: vec![("w", vec![4, 2, 2, 2])],
            nodes: node(
                "ConvTranspose2d",
                r#""h","w""#,
                r#"{"stride":[2,2],"groups":2}"#,
            ),
            output: vec![4, 6, 6],
            reference: Box::new(|h, p| {
                conv_transpose_ref(h, &p["w"], None, [2, 2], [0, 0], [0, 0], [1, 1], 2)
            }),
        },
        Case {
            label: "ConvTranspose2d as a 1D one (height 1)",
            input: vec![3, 1, 6],
            params: vec![("w", vec![3, 2, 1, 4]), ("b", vec![2])],
            nodes: node(
                "ConvTranspose2d",
                r#""h","w","b""#,
                r#"{"stride":[1,2],"padding":[0,1]}"#,
            ),
            output: vec![2, 1, 12],
            reference: Box::new(|h, p| {
                conv_transpose_ref(h, &p["w"], Some(&p["b"]), [1, 2], [0, 1], [0, 0], [1, 1], 1)
            }),
        },
        Case {
            label: "Pad constant",
            input: vec![3, 4, 5],
            params: vec![],
            nodes: node("Pad", r#""h""#, r#"{"pads":[0,1,0,2,0,0,3,1],"value":0.5}"#),
            output: vec![4, 7, 8],
            reference: Box::new(|h, _| pad_ref(h, &[0, 1, 0, 2, 0, 0, 3, 1], Mode::Constant(0.5))),
        },
        Case {
            label: "Pad reflect",
            input: vec![3, 6],
            params: vec![],
            nodes: node("Pad", r#""h""#, r#"{"pads":[0,2,4,0,1,5],"mode":"reflect"}"#),
            output: vec![6, 15],
            reference: Box::new(|h, _| pad_ref(h, &[0, 2, 4, 0, 1, 5], Mode::Reflect)),
        },
        Case {
            label: "Pad edge",
            input: vec![3, 4],
            params: vec![],
            nodes: node("Pad", r#""h""#, r#"{"pads":[0,0,3,0,2,1],"mode":"edge"}"#),
            output: vec![5, 8],
            reference: Box::new(|h, _| pad_ref(h, &[0, 0, 3, 0, 2, 1], Mode::Edge)),
        },
        Case {
            label: "Resize nearest 2D up",
            input: vec![2, 3, 4],
            params: vec![],
            nodes: node("Resize", r#""h""#, r#"{"size":[7,8]}"#),
            output: vec![2, 7, 8],
            reference: Box::new(|h, _| {
                resize_axis_ref(&resize_axis_ref(h, 3, 8, false, false), 2, 7, false, false)
            }),
        },
        Case {
            label: "Resize nearest 2D down",
            input: vec![2, 6, 6],
            params: vec![],
            nodes: node("Resize", r#""h""#, r#"{"size":[4,3],"mode":"nearest"}"#),
            output: vec![2, 4, 3],
            reference: Box::new(|h, _| {
                resize_axis_ref(&resize_axis_ref(h, 3, 3, false, false), 2, 4, false, false)
            }),
        },
        Case {
            label: "Resize nearest 1D",
            input: vec![3, 5],
            params: vec![],
            nodes: node("Resize", r#""h""#, r#"{"size":[12]}"#),
            output: vec![3, 12],
            reference: Box::new(|h, _| resize_axis_ref(h, 2, 12, false, false)),
        },
        Case {
            label: "Resize bilinear half-pixel",
            input: vec![2, 3, 4],
            params: vec![],
            nodes: node("Resize", r#""h""#, r#"{"size":[5,9],"mode":"linear"}"#),
            output: vec![2, 5, 9],
            reference: Box::new(|h, _| {
                resize_axis_ref(&resize_axis_ref(h, 3, 9, true, false), 2, 5, true, false)
            }),
        },
        Case {
            label: "Resize bilinear aligned corners",
            input: vec![2, 3, 4],
            params: vec![],
            nodes: node(
                "Resize",
                r#""h""#,
                r#"{"size":[6,7],"mode":"linear","align_corners":1}"#,
            ),
            output: vec![2, 6, 7],
            reference: Box::new(|h, _| {
                resize_axis_ref(&resize_axis_ref(h, 3, 7, true, true), 2, 6, true, true)
            }),
        },
        Case {
            label: "Resize bilinear down",
            input: vec![2, 8, 8],
            params: vec![],
            nodes: node("Resize", r#""h""#, r#"{"size":[3,5],"mode":"linear"}"#),
            output: vec![2, 3, 5],
            reference: Box::new(|h, _| {
                resize_axis_ref(&resize_axis_ref(h, 3, 5, true, false), 2, 3, true, false)
            }),
        },
        Case {
            label: "Resize linear 1D",
            input: vec![3, 4],
            params: vec![],
            nodes: node("Resize", r#""h""#, r#"{"size":[10],"mode":"linear"}"#),
            output: vec![3, 10],
            reference: Box::new(|h, _| resize_axis_ref(h, 2, 10, true, false)),
        },
        Case {
            label: "Flip two axes",
            input: vec![3, 4, 5],
            params: vec![],
            nodes: node("Flip", r#""h""#, r#"{"axes":[1,-1]}"#),
            output: vec![3, 4, 5],
            reference: Box::new(|h, _| {
                let mut out = h.clone();
                each_index(&h.shape, |i| {
                    let mut j = i.to_vec();
                    j[1] = h.shape[1] - 1 - i[1];
                    j[3] = h.shape[3] - 1 - i[3];
                    let off = out.offset(i);
                    out.data[off] = h.get(&j);
                });
                out
            }),
        },
        Case {
            label: "ReduceSum without the axis",
            input: vec![3, 4, 5],
            params: vec![],
            nodes: node("ReduceSum", r#""h""#, r#"{"axis":2,"keepdims":0}"#),
            output: vec![3, 5],
            reference: Box::new(|h, _| reduce_ref(h, 2, false, |v| v.iter().sum())),
        },
        Case {
            label: "ReduceMax",
            input: vec![5, 6],
            params: vec![],
            nodes: node("ReduceMax", r#""h""#, r#"{"axis":1}"#),
            output: vec![1, 6],
            reference: Box::new(|h, _| {
                reduce_ref(h, 1, true, |v| v.iter().cloned().fold(f64::MIN, f64::max))
            }),
        },
        Case {
            label: "ReduceMin last axis",
            input: vec![4, 7],
            params: vec![],
            nodes: node("ReduceMin", r#""h""#, r#"{"axis":-1,"keepdims":0}"#),
            output: vec![4],
            reference: Box::new(|h, _| {
                reduce_ref(h, 2, false, |v| v.iter().cloned().fold(f64::MAX, f64::min))
            }),
        },
        Case {
            label: "ReduceProd odd length",
            input: vec![3, 7],
            params: vec![],
            nodes: node("ReduceProd", r#""h""#, r#"{"axis":2}"#),
            output: vec![3, 1],
            reference: Box::new(|h, _| reduce_ref(h, 2, true, |v| v.iter().product())),
        },
        Case {
            label: "CumSum",
            input: vec![6, 3],
            params: vec![],
            nodes: node("CumSum", r#""h""#, r#"{"axis":1}"#),
            output: vec![6, 3],
            reference: Box::new(|h, _| {
                let mut out = h.clone();
                each_index(&h.shape, |i| {
                    let mut j = i.to_vec();
                    let mut s = 0.0;
                    for k in 0..=i[1] {
                        j[1] = k;
                        s += h.get(&j);
                    }
                    let off = out.offset(i);
                    out.data[off] = s;
                });
                out
            }),
        },
        Case {
            label: "IndexSelect with repetitions",
            input: vec![4, 3],
            params: vec![],
            nodes: node("IndexSelect", r#""h""#, r#"{"axis":1,"index":[2,0,2,1,3]}"#),
            output: vec![5, 3],
            reference: Box::new(|h, _| select_ref(h, 1, &[2, 0, 2, 1, 3])),
        },
        Case {
            label: "IndexSelect last axis of a rank-4 tensor",
            input: vec![2, 3, 4],
            params: vec![],
            nodes: node("IndexSelect", r#""h""#, r#"{"axis":3,"index":[3,3,0]}"#),
            output: vec![2, 3, 3],
            reference: Box::new(|h, _| select_ref(h, 3, &[3, 3, 0])),
        },
        Case {
            // Message passing over a fixed graph: messages read at the sources, weighted, then
            // accumulated at the destinations (node 1 receives twice).
            label: "IndexAdd message passing",
            input: vec![6, 4],
            params: vec![("q", vec![1, 7, 4])],
            nodes: r#"{"op":"IndexSelect","inputs":["h"],"output":"m","attrs":{"axis":1,"index":[0,1,2,3,4,5,3]}},
                      {"op":"Mul","inputs":["m","q"],"output":"mq"},
                      {"op":"IndexAdd","inputs":["h","mq"],"output":"pred","attrs":{"axis":1,"index":[1,2,3,4,5,0,1]}}"#
                .to_string(),
            output: vec![6, 4],
            reference: Box::new(|h, p| {
                let m = mul_broadcast(&select_ref(h, 1, &[0, 1, 2, 3, 4, 5, 3]), &p["q"]);
                index_add_ref(h, 1, &[1, 2, 3, 4, 5, 0, 1], &m)
            }),
        },
    ]
}

#[test]
fn each_operator_matches_its_definition_in_value_and_derivative() {
    let mut lines = Vec::new();
    for (k, case) in cases().iter().enumerate() {
        let (value, grad) = run(case, 10 + 3 * k as u64);
        lines.push(format!("| {:<52} | {value:.1e} | {grad:.1e} |", case.label));
    }
    eprintln!("| Case | value, relative gap | gradient, relative gap |");
    for l in lines {
        eprintln!("{l}");
    }
}

fn simple(op: &str, input: &[u64], attrs: &str, output: &[u64]) -> ModelGraphV1 {
    build(&Case {
        label: "reject",
        input: input.to_vec(),
        params: vec![],
        nodes: format!(r#"{{"op":"{op}","inputs":["h"],"output":"pred","attrs":{attrs}}}"#),
        output: output.to_vec(),
        reference: Box::new(|h, _| h.clone()),
    })
}

/// A tie between maxima shares the gradient equally (`torch.amax`); a zero in a product does
/// not break its derivative (`exp(Σ log x)` would).
#[test]
fn ties_and_zeros_have_the_right_derivative() {
    let json = |op: &str| {
        format!(
            r#"{{"format":"gg-graph/v1","name":"t","inputs":[{{"name":"x","shape":[0,4]}}],
          "parameters":[{{"name":"p","shape":[1,4],"init":"zeros"}}],
          "nodes":[{{"op":"Add","inputs":["x","p"],"output":"h"}},{{"op":"{op}","inputs":["h"],"output":"pred","attrs":{{"axis":1}}}}],
          "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,1]}}}}}}}}"#
        )
    };
    let inputs = |values: [f32; 4]| {
        BTreeMap::from([
            (
                "x".to_string(),
                TensorF32::new(vec![1, 4], values.to_vec()).unwrap(),
            ),
            (
                "y".to_string(),
                TensorF32::new(vec![1, 1], vec![0.0]).unwrap(),
            ),
        ])
    };
    for (op, values, expected) in [
        // max = 3, twice: d(3²)/dmax = 6, shared: 3 and 3.
        ("ReduceMax", [1.0, 3.0, 3.0, 2.0], [0.0, 3.0, 3.0, 0.0]),
        (
            "ReduceMin",
            [-1.0, 3.0, -1.0, -1.0],
            [-2.0 / 3.0, 0.0, -2.0 / 3.0, -2.0 / 3.0],
        ),
        // prod = 0; d(prod²)/dx = 2·prod·Π_{j≠i} x_j = 0 everywhere, yet finite.
        ("ReduceProd", [0.0, 3.0, 2.0, -1.0], [0.0, 0.0, 0.0, 0.0]),
    ] {
        let g: ModelGraphV1 = serde_json::from_str(&json(op)).unwrap();
        let prepared = Prepared::new(&g).unwrap();
        let params = exec::init_parameters(&g, 0);
        let step = exec::gradient_cpu(&prepared, &params, &inputs(values), 0).unwrap();
        let got = &step.gradients["p"].data;
        for (a, b) in got.iter().zip(expected) {
            assert!(
                (a - b).abs() <= 1e-6,
                "{op}: gradient {got:?}, expected {expected:?}"
            );
        }
    }
    // A single zero: d prod/dx_0 = 3·2·(−1) = −6, the others 0. Loss = (prod − y)², y = 1.
    let g: ModelGraphV1 = serde_json::from_str(&json("ReduceProd")).unwrap();
    let prepared = Prepared::new(&g).unwrap();
    let params = exec::init_parameters(&g, 0);
    let mut inp = inputs([0.0, 3.0, 2.0, -1.0]);
    inp.insert("y".into(), TensorF32::new(vec![1, 1], vec![1.0]).unwrap());
    let step = exec::gradient_cpu(&prepared, &params, &inp, 0).unwrap();
    // dL/dprod = 2·(0 − 1) = −2; dL/dx_0 = −2·(−6) = 12.
    assert_eq!(step.gradients["p"].data, vec![12.0, 0.0, 0.0, 0.0]);
}

#[test]
fn rejections_are_named() {
    let rejection = |g: ModelGraphV1| match graph::validate(&g) {
        Err(GraphError::Shape(m)) | Err(GraphError::Attribute(m)) => m,
        other => panic!("expected a rejection, got {other:?}"),
    };
    let cases: Vec<(ModelGraphV1, &str)> = vec![
        // Along the batch: every one of these would mix examples.
        (simple("ReduceSum", &[4], r#"{"axis":0}"#, &[4]), "batch"),
        (simple("ReduceMax", &[4], r#"{"axis":0}"#, &[4]), "batch"),
        (simple("CumSum", &[4], r#"{"axis":0}"#, &[4]), "batch"),
        (simple("Flip", &[4], r#"{"axes":[0]}"#, &[4]), "batch"),
        (
            simple("IndexSelect", &[4], r#"{"axis":0,"index":[0]}"#, &[4]),
            "batch",
        ),
        (simple("Pad", &[4], r#"{"pads":[1,0,0,0]}"#, &[4]), "batch"),
        // Indices.
        (
            simple("IndexSelect", &[4], r#"{"axis":1,"index":[4]}"#, &[1]),
            "outside axis 1",
        ),
        (
            simple("IndexSelect", &[4], r#"{"axis":1,"index":[-1]}"#, &[1]),
            "outside axis 1",
        ),
        (
            simple("IndexSelect", &[4], r#"{"axis":1,"index":[]}"#, &[1]),
            "indices",
        ),
        (
            simple("Flip", &[4, 3], r#"{"axes":[1,-2]}"#, &[4, 3]),
            "twice",
        ),
        // Padding.
        (
            simple("Pad", &[4], r#"{"pads":[0,0,0,-1]}"#, &[3]),
            "negative",
        ),
        (
            simple("Pad", &[4], r#"{"pads":[0,4,0,0],"mode":"reflect"}"#, &[8]),
            "reflection",
        ),
        (
            simple("Pad", &[4], r#"{"pads":[0,1,0,1],"mode":"wrap"}"#, &[6]),
            "unknown mode",
        ),
        (
            simple("Pad", &[4], r#"{"pads":[0,1]}"#, &[5]),
            "begins, then ends",
        ),
        (
            simple(
                "Pad",
                &[4],
                r#"{"pads":[0,1,0,1],"mode":"edge","value":1}"#,
                &[6],
            ),
            "constant mode",
        ),
        // Resizing.
        (simple("Resize", &[4], r#"{"size":[8]}"#, &[8]), "[N, C, L]"),
        (
            simple("Resize", &[2, 4], r#"{"size":[8,8]}"#, &[2, 8]),
            "spatial axes",
        ),
        (
            simple("Resize", &[2, 4], r#"{"size":[0]}"#, &[2, 0]),
            "outside",
        ),
        (
            simple("Resize", &[2, 4], r#"{"size":[8],"mode":"cubic"}"#, &[2, 8]),
            "unknown mode",
        ),
        (
            simple(
                "Resize",
                &[2, 4],
                r#"{"size":[8],"align_corners":1}"#,
                &[2, 8],
            ),
            "linear mode",
        ),
        // Reductions.
        (
            simple("ReduceMin", &[4], r#"{"axis":3}"#, &[4]),
            "outside the rank",
        ),
        (
            simple("ReduceSum", &[4], r#"{"axis":1,"keepdims":2}"#, &[1]),
            "keepdims",
        ),
    ];
    for (g, expected) in cases {
        let m = rejection(g);
        assert!(m.contains(expected), "expected '{expected}' in: {m}");
    }
    // Missing mandatory attributes.
    for (op, attrs, missing) in [
        ("IndexSelect", r#"{"axis":1}"#, "'index'"),
        ("IndexSelect", r#"{"index":[0]}"#, "'axis'"),
        ("Pad", "{}", "'pads'"),
        ("Resize", "{}", "'size'"),
        ("Flip", "{}", "'axes'"),
        ("CumSum", "{}", "'axis'"),
        ("ReduceSum", "{}", "'axis'"),
    ] {
        let e = graph::validate(&simple(op, &[4], attrs, &[4])).unwrap_err();
        let m = e.to_string();
        assert!(m.contains(missing) && m.contains("required"), "{op}: {m}");
    }
    // Transposed convolution.
    let conv = |attrs: &str, w: Vec<u64>| {
        build(&Case {
            label: "reject",
            input: vec![4, 3, 3],
            params: vec![("w", w)],
            nodes: format!(
                r#"{{"op":"ConvTranspose2d","inputs":["h","w"],"output":"pred","attrs":{attrs}}}"#
            ),
            output: vec![3, 5, 5],
            reference: Box::new(|h, _| h.clone()),
        })
    };
    for (attrs, w, expected) in [
        (
            r#"{"output_padding":[1,1]}"#,
            vec![4, 3, 3, 3],
            "output_padding",
        ),
        (r#"{}"#, vec![5, 3, 3, 3], "input channels"),
        (r#"{"groups":3}"#, vec![4, 3, 3, 3], "groups"),
        (r#"{"stride":[0,1]}"#, vec![4, 3, 3, 3], "stride"),
        (
            r#"{"padding":[4,4]}"#,
            vec![4, 3, 3, 3],
            "removes the whole output",
        ),
    ] {
        let m = rejection(conv(attrs, w));
        assert!(m.contains(expected), "expected '{expected}' in: {m}");
    }
    // IndexAdd: src must have the shape of x, with the index length along the axis.
    let g = build(&Case {
        label: "reject",
        input: vec![6, 4],
        params: vec![],
        nodes: r#"{"op":"IndexAdd","inputs":["h","h"],"output":"pred","attrs":{"axis":1,"index":[0,1]}}"#
            .into(),
        output: vec![6, 4],
        reference: Box::new(|h, _| h.clone()),
    });
    assert!(rejection(g).contains("src must be"));
}

/// Cost and memory follow the operator: FLOPs per example, and what automatic differentiation
/// keeps. The memory estimate stays an upper bound on what the executor measures.
#[test]
fn cost_and_memory_estimators() {
    let report = |g: ModelGraphV1| graph::validate(&g).unwrap();
    let node =
        |r: &graph::GraphReport| r.nodes.iter().find(|c| c.output == "pred").unwrap().clone();
    // Transposed convolution: 2 · (input elements) · (C_out/g · KH · KW).
    let r = report(build(&Case {
        label: "c",
        input: vec![4, 3, 5],
        params: vec![("w", vec![4, 3, 3, 2])],
        nodes: r#"{"op":"ConvTranspose2d","inputs":["h","w"],"output":"pred","attrs":{"stride":[2,1],"padding":[1,0],"output_padding":[1,0],"dilation":[1,2]}}"#.into(),
        output: vec![3, 6, 7],
        reference: Box::new(|h, _| h.clone()),
    }));
    assert_eq!(
        node(&r).training_flops_per_example,
        3 * 2 * (4 * 3 * 5) * (3 * 3 * 2)
    );
    assert_eq!(node(&r).activation_bytes_per_example, 4 * 3 * 6 * 7);
    // Copies cost no FLOP; a padded tensor keeps its fill.
    let r = report(simple("Pad", &[3, 4], r#"{"pads":[0,0,1,0,0,1]}"#, &[3, 6]));
    assert_eq!(node(&r).training_flops_per_example, 0);
    assert_eq!(node(&r).activation_bytes_per_example, 2 * 4 * 3 * 6);
    // A maximum keeps its mask and the difference: output + 2 · input.
    let r = report(simple("ReduceMax", &[5, 6], r#"{"axis":1}"#, &[1, 6]));
    assert_eq!(node(&r).training_flops_per_example, 3 * 3 * 30);
    assert_eq!(node(&r).activation_bytes_per_example, 4 * (6 + 60));
    // Bilinear: 3 FLOPs per output element and per axis; five steps per axis kept.
    let r = report(simple(
        "Resize",
        &[2, 3, 4],
        r#"{"size":[5,9],"mode":"linear"}"#,
        &[2, 5, 9],
    ));
    assert_eq!(node(&r).training_flops_per_example, 3 * 3 * 2 * 2 * 5 * 9);
    assert_eq!(
        node(&r).activation_bytes_per_example,
        4 * (5 * (2 * 3 * 9) + 5 * (2 * 5 * 9))
    );
    // Every new operator requires gg-ops/7; a Formula only when it uses a new function.
    for (op, attrs, output) in [
        ("CumSum", r#"{"axis":1}"#, vec![4]),
        ("Flip", r#"{"axes":[1]}"#, vec![4]),
        ("IndexSelect", r#"{"axis":1,"index":[1]}"#, vec![1]),
    ] {
        assert_eq!(
            graph::memory::required_profile(&simple(op, &[4], attrs, &output)),
            7
        );
    }
    let f = |e: &str| simple("Formula", &[4], &format!(r#"{{"expr":"{e}"}}"#), &[4]);
    assert_eq!(graph::memory::required_profile(&f("a * 2")), 6);
    assert_eq!(graph::memory::required_profile(&f("floor(a) + a")), 7);

    // The task memory estimate bounds the executor's measurement.
    for case in cases() {
        let g = build(&case);
        let prepared = Prepared::new(&g).unwrap();
        let params = exec::init_parameters(&g, 1);
        let inputs = BTreeMap::from([
            ("x".to_string(), random(&batch_shape(&case.input), 2, 1.0)),
            ("y".to_string(), random(&batch_shape(&case.output), 3, 1.0)),
        ]);
        let step = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
        let est = graph::memory::task_memory(
            &g,
            prepared.report(),
            N as u64,
            exec::ExecOptions::default(),
        );
        assert!(
            est.total() >= step.peak_activation_bytes + step.peak_parameter_bytes,
            "{}: estimate {} below the measurement {}",
            case.label,
            est.total(),
            step.peak_activation_bytes + step.peak_parameter_bytes
        );
    }
}

/// Real `transformers` speech models, captured by the PyTorch driver
/// (`gg-python/tools/make_torch_fixtures.py`): a SpeechT5 HiFi-GAN vocoder (transposed 1D
/// convolutions) and a Wav2Vec2 encoder (a weight-normalized positional convolution). The loss and
/// the gradient of every trained parameter, computed by PyTorch, travel with the fixture: the
/// executor must find the same ones. This is what says that such a model **trains** as in
/// PyTorch, not only that it evaluates the same function.
#[test]
fn captured_speech_models_train_like_pytorch() {
    use gg_protocol::tensor_file::{BundleF32, TensorLimits};

    let dir = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
    let bundle = |file: String| {
        BundleF32::decode(
            &std::fs::read(format!("{dir}/{file}")).unwrap(),
            &TensorLimits::default(),
        )
        .unwrap()
        .entries
    };
    // ConvNeXt: depthwise 7×7 convolutions, whose weight gradient the executor computes by its own
    // route (`exec/extra.rs`, `depthwise_conv2d`).
    for (name, profile) in [
        ("hifigan-mini", 7),
        ("wav2vec2-mini", 7),
        ("convnext-mini", 5),
    ] {
        let graph: ModelGraphV1 =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap())
                .unwrap();
        let torch = bundle(format!("{name}.torch.ggb"));
        let torch_grad = bundle(format!("{name}.grad.ggb"));
        let prepared = Prepared::new(&graph).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(graph::memory::required_profile(&graph), profile, "{name}");
        let params: BTreeMap<String, TensorF32> = graph
            .parameters
            .iter()
            .map(|p| (p.name.clone(), torch[&p.name].clone()))
            .collect();
        let inputs = BTreeMap::from([
            (graph.inputs[0].name.clone(), torch["__x"].clone()),
            ("y".to_string(), torch["__y"].clone()),
        ]);
        let step = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
        let expected_loss = torch["__loss"].data[0];
        assert!(
            (step.loss - expected_loss).abs() <= 1e-5 * expected_loss.abs().max(1.0),
            "{name}: loss {} versus {expected_loss} in PyTorch",
            step.loss
        );
        let mut worst = 0f64;
        assert_eq!(step.gradients.len(), torch_grad.len(), "{name}");
        for (p, g) in &step.gradients {
            let r = &torch_grad[p];
            assert_eq!(g.shape, r.shape, "{name}: {p}");
            let scale = r.data.iter().fold(1e-3f64, |m, v| m.max(v.abs() as f64));
            let gap = g
                .data
                .iter()
                .zip(&r.data)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0f64, f64::max)
                / scale;
            worst = worst.max(gap);
            assert!(
                gap <= 1e-5,
                "{name}: gradient of {p} departs from PyTorch's by {gap:e} (relative)"
            );
        }
        eprintln!(
            "{name}: {} nodes, loss {} (PyTorch {expected_loss}), worst relative gradient gap {worst:.1e}",
            graph.nodes.len(),
            step.loss
        );
        let again = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
        for (k, t) in &step.gradients {
            assert_eq!(t, &again.gradients[k], "{name}: {k} not reproducible");
        }
    }
}
