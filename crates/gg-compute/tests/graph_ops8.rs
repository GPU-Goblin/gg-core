//! Profile `gg-ops/8` (ADR-0029): integer tensors computed inside the graph. `ArgMax`, `ArgMin`,
//! `ArgTopK`, `ArgSort`, `Cast`, `GatherElements`, `ScatterElements`, and `Gather` over tables of
//! rank 3. Each brings its oracle, its named rejections, its determinism test, its cost and its
//! memory estimate (ADR-0016).
//!
//! Two kinds of oracle. The **indices** are compared exactly with a reference written here (first
//! index among ties, stable sort, a NaN counting as the largest). The **values** that flow through
//! them (the values of `topk`, `sort`, `max(dim)`, a scatter, an expert's weights) are compared
//! with an f64 definition, and their derivative with central finite differences of it: the
//! gradient must reach the selected positions, and nothing reaches an index.

mod common;

use std::collections::BTreeMap;

use common::*;
use gg_compute::graph::exec::{self, Prepared};
use gg_compute::graph::{self, GraphError, ModelGraphV1};
use gg_protocol::tensor_file::TensorF32;

/// Stable order of a line: descending or not, ties by index, NaN largest.
fn order_of(line: &[f64], descending: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..line.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (line[a], line[b]);
        let c = match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => x.partial_cmp(&y).unwrap(),
        };
        (if descending { c.reverse() } else { c }).then(a.cmp(&b))
    });
    idx
}

/// Applies `f` to every line of `x` along its last axis; `f` returns the new line.
fn per_line(x: &T64, out_len: usize, f: impl Fn(&[f64]) -> Vec<f64>) -> T64 {
    let n = *x.shape.last().unwrap();
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = out_len;
    let mut out = Vec::new();
    for line in x.data.chunks(n) {
        let v = f(line);
        assert_eq!(v.len(), out_len);
        out.extend(v);
    }
    T64 { shape, data: out }
}

fn topk(line: &[f64], k: usize, largest: bool) -> Vec<f64> {
    order_of(line, largest)[..k]
        .iter()
        .map(|i| line[*i])
        .collect()
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f64::MIN, f64::max);
    let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            label: "topk values (ArgTopK + GatherElements)",
            input: vec![3, 7],
            params: vec![],
            nodes: r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":2,"k":3}},
                      {"op":"GatherElements","inputs":["h","i"],"output":"pred","attrs":{"axis":2}}"#
                .into(),
            output: vec![3, 3],
            reference: Box::new(|h, _| per_line(h, 3, |l| topk(l, 3, true))),
        },
        Case {
            label: "smallest two along a middle axis",
            input: vec![5, 4],
            params: vec![],
            nodes: r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":1,"k":2,"largest":0}},
                      {"op":"GatherElements","inputs":["h","i"],"output":"pred","attrs":{"axis":1}}"#
                .into(),
            output: vec![2, 4],
            reference: Box::new(|h, _| {
                let (n, a, b) = (h.shape[0], h.shape[1], h.shape[2]);
                let mut out = T64::zeros(&[n, 2, b]);
                for e in 0..n {
                    for j in 0..b {
                        let line: Vec<f64> = (0..a).map(|i| h.get(&[e, i, j])).collect();
                        for (r, v) in topk(&line, 2, false).into_iter().enumerate() {
                            let o = out.offset(&[e, r, j]);
                            out.data[o] = v;
                        }
                    }
                }
                out
            }),
        },
        Case {
            label: "sort values (ArgSort + GatherElements)",
            input: vec![2, 6],
            params: vec![],
            nodes: r#"{"op":"ArgSort","inputs":["h"],"output":"i","attrs":{"axis":-1}},
                      {"op":"GatherElements","inputs":["h","i"],"output":"pred","attrs":{"axis":2}}"#
                .into(),
            output: vec![2, 6],
            reference: Box::new(|h, _| {
                per_line(h, 6, |l| order_of(l, false).iter().map(|i| l[*i]).collect())
            }),
        },
        Case {
            label: "max(dim) values (ArgMax + GatherElements)",
            input: vec![3, 5],
            params: vec![],
            nodes: r#"{"op":"ArgMax","inputs":["h"],"output":"i","attrs":{"axis":2}},
                      {"op":"GatherElements","inputs":["h","i"],"output":"pred","attrs":{"axis":2}}"#
                .into(),
            output: vec![3, 1],
            reference: Box::new(|h, _| {
                per_line(h, 1, |l| vec![l[order_of(l, true)[0]]])
            }),
        },
        Case {
            label: "argmin as f32 (ArgMin, Cast) next to a sum",
            input: vec![3, 5],
            params: vec![],
            nodes: r#"{"op":"ArgMin","inputs":["h"],"output":"i","attrs":{"axis":2,"keepdims":0}},
                      {"op":"Cast","inputs":["i"],"output":"f","attrs":{"to":"f32"}},
                      {"op":"ReduceSum","inputs":["h"],"output":"s","attrs":{"axis":2,"keepdims":0}},
                      {"op":"Add","inputs":["f","s"],"output":"pred"}"#
                .into(),
            output: vec![3],
            reference: Box::new(|h, _| {
                let t = per_line(h, 1, |l| {
                    vec![order_of(l, false)[0] as f64 + l.iter().sum::<f64>()]
                });
                T64 {
                    shape: t.shape[..2].to_vec(),
                    data: t.data,
                }
            }),
        },
        Case {
            label: "scatter_add at a computed index",
            input: vec![4, 6],
            params: vec![("q", vec![1, 4, 2])],
            nodes: r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":2,"k":2}},
                      {"op":"GatherElements","inputs":["h","i"],"output":"t","attrs":{"axis":2}},
                      {"op":"Mul","inputs":["t","q"],"output":"s"},
                      {"op":"ScatterElements","inputs":["h","i","s"],"output":"pred","attrs":{"axis":2}}"#
                .into(),
            output: vec![4, 6],
            reference: Box::new(|h, p| {
                let q = &p["q"];
                let mut out = h.clone();
                let n = 6;
                for (li, line) in h.data.chunks(n).enumerate() {
                    let r = li % 4;
                    for (j, i) in order_of(line, true)[..2].iter().enumerate() {
                        out.data[li * n + i] += line[*i] * q.data[r * 2 + j];
                    }
                }
                out
            }),
        },
        Case {
            // The gating of a mixture of experts: softmax of the two best scores, written back at
            // their positions over the scores themselves.
            label: "scatter assignment of distinct positions (MoE gating)",
            input: vec![5, 4],
            params: vec![],
            nodes: r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":2,"k":2}},
                      {"op":"GatherElements","inputs":["h","i"],"output":"t","attrs":{"axis":2}},
                      {"op":"Softmax","inputs":["t"],"output":"w","attrs":{"axis":2}},
                      {"op":"ScatterElements","inputs":["h","i","w"],"output":"pred","attrs":{"axis":2,"reduction":"none"}}"#
                .into(),
            output: vec![5, 4],
            reference: Box::new(|h, _| {
                per_line(h, 4, |l| {
                    let o = order_of(l, true);
                    let w = softmax(&[l[o[0]], l[o[1]]]);
                    let mut v = l.to_vec();
                    v[o[0]] = w[0];
                    v[o[1]] = w[1];
                    v
                })
            }),
        },
        Case {
            label: "integer shape operators (Slice, Concat of a sort)",
            input: vec![6],
            params: vec![],
            nodes: r#"{"op":"ArgSort","inputs":["h"],"output":"i","attrs":{"axis":1,"descending":1}},
                      {"op":"Slice","inputs":["i"],"output":"a","attrs":{"axis":1,"start":0,"end":3}},
                      {"op":"Slice","inputs":["i"],"output":"b","attrs":{"axis":1,"start":3,"end":6}},
                      {"op":"Concat","inputs":["b","a"],"output":"c","attrs":{"axis":1}},
                      {"op":"GatherElements","inputs":["h","c"],"output":"pred","attrs":{"axis":1}}"#
                .into(),
            output: vec![6],
            reference: Box::new(|h, _| {
                per_line(h, 6, |l| {
                    let o = order_of(l, true);
                    o[3..].iter().chain(&o[..3]).map(|i| l[*i]).collect()
                })
            }),
        },
        Case {
            // Top-1 routing to experts whose weights are a table [experts, out, in]: the rows of
            // the chosen experts are gathered, then applied (transformers' batched_mm experts).
            label: "routed experts (Gather of a rank-3 table)",
            input: vec![4],
            params: vec![("r", vec![4, 3]), ("w", vec![3, 2, 4])],
            nodes: r#"{"op":"MatMul","inputs":["h","r"],"output":"logits"},
                      {"op":"ArgTopK","inputs":["logits"],"output":"i","attrs":{"axis":1,"k":1}},
                      {"op":"Reshape","inputs":["i"],"output":"e","attrs":{"shape":[-1]}},
                      {"op":"Gather","inputs":["w","e"],"output":"we"},
                      {"op":"Reshape","inputs":["h"],"output":"h3","attrs":{"shape":[-1,4,1]}},
                      {"op":"MatMul","inputs":["we","h3"],"output":"o"},
                      {"op":"Reshape","inputs":["o"],"output":"pred","attrs":{"shape":[-1,2]}}"#
                .into(),
            output: vec![2],
            reference: Box::new(|h, p| {
                let (r, w) = (&p["r"], &p["w"]);
                let n = h.shape[0];
                let mut out = T64::zeros(&[n, 2]);
                for e in 0..n {
                    let x: Vec<f64> = (0..4).map(|k| h.data[e * 4 + k]).collect();
                    let logits: Vec<f64> = (0..3)
                        .map(|j| (0..4).map(|k| x[k] * r.data[k * 3 + j]).sum())
                        .collect();
                    let best = order_of(&logits, true)[0];
                    for o in 0..2 {
                        out.data[e * 2 + o] =
                            (0..4).map(|k| w.data[(best * 2 + o) * 4 + k] * x[k]).sum();
                    }
                }
                out
            }),
        },
    ]
}

#[test]
fn each_operator_matches_its_definition_in_value_and_derivative() {
    let mut lines = Vec::new();
    for (k, case) in cases().iter().enumerate() {
        let (value, grad) = run(case, 40 + 3 * k as u64);
        lines.push(format!("| {:<55} | {value:.1e} | {grad:.1e} |", case.label));
    }
    eprintln!("| Case | value, relative gap | gradient, relative gap |");
    for l in lines {
        eprintln!("{l}");
    }
}

/// A one-node graph around `x` (an input `[N, n]`) and a trained zero `p`: `h = formula(x, p)`,
/// then the integer operator, then `Cast` to read it.
fn index_graph(n: usize, formula: &str, node: &str, out: &str) -> ModelGraphV1 {
    let json = format!(
        r#"{{"format":"gg-graph/v1","name":"i","inputs":[{{"name":"x","shape":[0,{n}]}}],
          "parameters":[{{"name":"p","shape":[{n}],"init":"zeros"}}],
          "nodes":[{{"op":"Formula","inputs":["x","p"],"output":"h","attrs":{{"expr":"{formula}"}}}},
                   {node},
                   {{"op":"Cast","inputs":["i"],"output":"pred","attrs":{{"to":"f32"}}}}],
          "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[{out}]}}}}}}}}"#
    );
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("{e}\n{json}"))
}

fn indices(g: &ModelGraphV1, x: &[f32]) -> Vec<f32> {
    let prepared = Prepared::new(g).unwrap();
    let params = exec::init_parameters(g, 0);
    let t = TensorF32::new(vec![1, x.len() as u64], x.to_vec()).unwrap();
    exec::forward_cpu(
        &prepared,
        &params,
        &BTreeMap::from([("x".to_string(), t)]),
        0,
        "pred",
    )
    .unwrap()
    .data
}

/// Ties, NaNs and order: the rules of ADR-0029, value by value.
#[test]
fn ties_and_nans_follow_the_rules() {
    let arg = |op: &str, extra: &str| {
        format!(r#"{{"op":"{op}","inputs":["h"],"output":"i","attrs":{{"axis":1{extra}}}}}"#)
    };
    // First index among ties.
    let g = index_graph(6, "a + b", &arg("ArgMax", r#","keepdims":0"#), "0");
    assert_eq!(indices(&g, &[1.0, 3.0, 3.0, 2.0, 3.0, 0.0]), vec![1.0]);
    let g = index_graph(4, "a + b", &arg("ArgMin", r#","keepdims":0"#), "0");
    assert_eq!(indices(&g, &[2.0, 0.0, 1.0, 0.0]), vec![1.0]);
    // A NaN is the largest: the first NaN wins, for ArgMax as for ArgMin (PyTorch).
    let nan = index_graph(4, "sqrt(a + b)", &arg("ArgMax", r#","keepdims":0"#), "0");
    assert_eq!(indices(&nan, &[1.0, -4.0, 5.0, -1.0]), vec![1.0]);
    let nan = index_graph(4, "sqrt(a + b)", &arg("ArgMin", r#","keepdims":0"#), "0");
    assert_eq!(indices(&nan, &[1.0, -4.0, 5.0, -1.0]), vec![1.0]);
    // Top-k: lower index first among ties; NaN first when the largest are asked for.
    let x = [2.0, 5.0, 5.0, 1.0, 5.0];
    let g = |k, largest| {
        index_graph(
            5,
            "a + b",
            &arg("ArgTopK", &format!(r#","k":{k},"largest":{largest}"#)),
            &format!("0,{k}"),
        )
    };
    assert_eq!(indices(&g(3, 1), &x), vec![1.0, 2.0, 4.0]);
    assert_eq!(indices(&g(2, 0), &x), vec![3.0, 0.0]);
    let nan = index_graph(4, "sqrt(a + b)", &arg("ArgTopK", r#","k":2"#), "0,2");
    assert_eq!(indices(&nan, &[4.0, -1.0, 9.0, -2.0]), vec![1.0, 3.0]);
    // Stable sort, both directions.
    let g = |d| {
        index_graph(
            4,
            "a + b",
            &arg("ArgSort", &format!(r#","descending":{d}"#)),
            "0,4",
        )
    };
    assert_eq!(
        indices(&g(0), &[2.0, 1.0, 2.0, 1.0]),
        vec![1.0, 3.0, 0.0, 2.0]
    );
    assert_eq!(
        indices(&g(1), &[2.0, 1.0, 2.0, 1.0]),
        vec![0.0, 2.0, 1.0, 3.0]
    );
    // The host sort, against the reference, on many lines with repeated values.
    let dims = [3usize, 5, 4];
    let values: Vec<f32> = (0..60).map(|i| ((i * 7) % 5) as f32).collect();
    for axis in 0..3 {
        for desc in [false, true] {
            let got = exec::line_order(&values, &dims, axis, desc, dims[axis]);
            let inner: usize = dims[axis + 1..].iter().product();
            let outer: usize = dims[..axis].iter().product();
            for o in 0..outer {
                for i in 0..inner {
                    let line: Vec<f64> = (0..dims[axis])
                        .map(|j| values[(o * dims[axis] + j) * inner + i] as f64)
                        .collect();
                    let want = order_of(&line, desc);
                    for (j, w) in want.iter().enumerate() {
                        assert_eq!(got[(o * dims[axis] + j) * inner + i], *w as i64);
                    }
                }
            }
        }
    }
}

/// The derivative of an index is zero, and it reaches only the selected positions of a gather.
#[test]
fn indices_carry_no_gradient_and_values_carry_it_to_the_selected_positions() {
    let json = r#"{"format":"gg-graph/v1","name":"g","inputs":[{"name":"x","shape":[0,5]}],
      "parameters":[{"name":"p","shape":[1,5],"init":"zeros"}],
      "nodes":[{"op":"Add","inputs":["x","p"],"output":"h"},
               {"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":1,"k":2}},
               {"op":"Cast","inputs":["i"],"output":"c","attrs":{"to":"f32"}},
               {"op":"GatherElements","inputs":["h","i"],"output":"v","attrs":{"axis":1}},
               {"op":"Add","inputs":["v","c"],"output":"pred"}],
      "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,2]}}}}"#;
    let g: ModelGraphV1 = serde_json::from_str(json).unwrap();
    let prepared = Prepared::new(&g).unwrap();
    let params = exec::init_parameters(&g, 0);
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            TensorF32::new(vec![1, 5], vec![0.5, 3.0, -1.0, 2.0, 0.0]).unwrap(),
        ),
        (
            "y".to_string(),
            TensorF32::new(vec![1, 2], vec![0.0, 0.0]).unwrap(),
        ),
    ]);
    let step = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
    // pred = [3 + 1, 2 + 3] = [4, 5]; dL/dpred = pred (mean of 2 squares: 2·pred/2).
    assert_eq!(step.loss, (16.0 + 25.0) / 2.0);
    assert_eq!(step.gradients["p"].data, vec![0.0, 4.0, 0.0, 5.0, 0.0]);
}

fn simple(
    op_nodes: &str,
    input: &[u64],
    output: &[u64],
    params: Vec<(&'static str, Vec<u64>)>,
) -> ModelGraphV1 {
    build(&Case {
        label: "reject",
        input: input.to_vec(),
        params,
        nodes: op_nodes.to_string(),
        output: output.to_vec(),
        reference: Box::new(|h, _| h.clone()),
    })
}

#[test]
fn rejections_are_named() {
    let rejection = |g: ModelGraphV1| match graph::validate(&g) {
        Err(GraphError::Shape(m)) | Err(GraphError::Attribute(m)) | Err(GraphError::Dtype(m)) => m,
        other => panic!("expected a rejection, got {other:?}"),
    };
    let topk = r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":1,"k":2}}"#;
    let cases: Vec<(ModelGraphV1, &str)> = vec![
        // An integer into arithmetic.
        (
            simple(
                &format!(r#"{topk},{{"op":"Add","inputs":["i","h"],"output":"pred"}}"#),
                &[4],
                &[4],
                vec![],
            ),
            "integer tensor",
        ),
        // The loss compares floats.
        (
            simple(
                &format!(
                    r#"{topk},{{"op":"Reshape","inputs":["i"],"output":"pred","attrs":{{"shape":[-1,2]}}}}"#
                ),
                &[4],
                &[2],
                vec![],
            ),
            "loss",
        ),
        // Along the batch.
        (
            simple(
                r#"{"op":"ArgMax","inputs":["h"],"output":"i","attrs":{"axis":0}},{"op":"Cast","inputs":["i"],"output":"pred","attrs":{"to":"f32"}}"#,
                &[4],
                &[4],
                vec![],
            ),
            "batch",
        ),
        // k outside the axis.
        (
            simple(
                r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":1,"k":5}},{"op":"Cast","inputs":["i"],"output":"pred","attrs":{"to":"f32"}}"#,
                &[4],
                &[5],
                vec![],
            ),
            "k = 5",
        ),
        (
            simple(
                r#"{"op":"ArgSort","inputs":["h"],"output":"i","attrs":{"axis":1,"descending":2}},{"op":"Cast","inputs":["i"],"output":"pred","attrs":{"to":"f32"}}"#,
                &[4],
                &[4],
                vec![],
            ),
            "descending",
        ),
        (
            simple(
                &format!(
                    r#"{topk},{{"op":"Cast","inputs":["i"],"output":"pred","attrs":{{"to":"f16"}}}}"#
                ),
                &[4],
                &[2],
                vec![],
            ),
            "f32",
        ),
        // A computed index whose range does not fit the axis it indexes.
        (
            simple(
                &format!(
                    r#"{topk},{{"op":"Slice","inputs":["h"],"output":"s","attrs":{{"axis":1,"start":0,"end":1}}}},{{"op":"GatherElements","inputs":["s","i"],"output":"pred","attrs":{{"axis":1}}}}"#
                ),
                &[4],
                &[2],
                vec![],
            ),
            "takes values in [0, 4)",
        ),
        // An assignment whose positions are not known to be distinct.
        (
            simple(
                r#"{"op":"ArgMax","inputs":["h"],"output":"i","attrs":{"axis":1}},{"op":"GatherElements","inputs":["h","i"],"output":"v","attrs":{"axis":1}},{"op":"ScatterElements","inputs":["h","i","v"],"output":"pred","attrs":{"axis":1,"reduction":"none"}}"#,
                &[4],
                &[4],
                vec![],
            ),
            "distinct",
        ),
        (
            simple(
                &format!(
                    r#"{topk},{{"op":"GatherElements","inputs":["h","i"],"output":"v","attrs":{{"axis":1}}}},{{"op":"ScatterElements","inputs":["h","i","v"],"output":"pred","attrs":{{"axis":1,"reduction":"mul"}}}}"#
                ),
                &[4],
                &[4],
                vec![],
            ),
            "reduction",
        ),
        // Shapes.
        (
            simple(
                &format!(
                    r#"{topk},{{"op":"GatherElements","inputs":["h","i"],"output":"v","attrs":{{"axis":1}}}},{{"op":"Slice","inputs":["v"],"output":"w","attrs":{{"axis":1,"start":0,"end":1}}}},{{"op":"ScatterElements","inputs":["h","i","w"],"output":"pred","attrs":{{"axis":1}}}}"#
                ),
                &[4],
                &[4],
                vec![],
            ),
            "shape of the index",
        ),
        (
            simple(
                &format!(
                    r#"{topk},{{"op":"GatherElements","inputs":["t","i"],"output":"pred","attrs":{{"axis":1}}}}"#
                ),
                &[4],
                &[2],
                vec![("t", vec![1, 4])],
            ),
            "batch axis first, or neither",
        ),
        // A table whose first axis is the batch.
        (
            simple(
                &format!(r#"{topk},{{"op":"Gather","inputs":["h","i"],"output":"pred"}}"#),
                &[4],
                &[2, 4],
                vec![],
            ),
            "mix",
        ),
    ];
    for (g, expected) in cases {
        let m = rejection(g);
        assert!(m.contains(expected), "expected '{expected}' in: {m}");
    }
}

/// An index that comes from the data is checked when the micro-batch is loaded: a named error,
/// never a read the GPU would clamp.
#[test]
fn an_index_from_the_data_is_checked_at_loading() {
    let json = r#"{"format":"gg-graph/v1","name":"e","inputs":[{"name":"ids","shape":[0,3],"dtype":"i64"}],
      "parameters":[{"name":"emb","shape":[6,4],"init":"xavier_uniform"},{"name":"w","shape":[4,5],"init":"xavier_uniform"}],
      "nodes":[{"op":"Gather","inputs":["emb","ids"],"output":"h"},{"op":"MatMul","inputs":["h","w"],"output":"s"},
               {"op":"Reshape","inputs":["s"],"output":"pred","attrs":{"shape":[-1,5]}}],
      "loss":{"cross_entropy":{"prediction":"pred","target":{"name":"y","shape":[0,3],"dtype":"i64"}}}}"#;
    let g: ModelGraphV1 = serde_json::from_str(json).unwrap();
    let prepared = Prepared::new(&g).unwrap();
    let params = exec::init_parameters(&g, 0);
    let t = |v: Vec<f32>| TensorF32::new(vec![1, 3], v).unwrap();
    let run = |ids: Vec<f32>, y: Vec<f32>| {
        exec::gradient_cpu(
            &prepared,
            &params,
            &BTreeMap::from([("ids".to_string(), t(ids)), ("y".to_string(), t(y))]),
            0,
        )
    };
    assert!(run(vec![0.0, 5.0, 2.0], vec![4.0, 0.0, 1.0]).is_ok());
    let e = run(vec![0.0, 6.0, 2.0], vec![4.0, 0.0, 1.0])
        .unwrap_err()
        .to_string();
    assert!(e.contains("'ids'") && e.contains("outside [0, 6)"), "{e}");
    let e = run(vec![0.0, 1.0, -1.0], vec![4.0, 0.0, 1.0])
        .unwrap_err()
        .to_string();
    assert!(e.contains("outside [0, 6)"), "{e}");
    let e = run(vec![0.0, 1.0, 2.0], vec![5.0, 0.0, 1.0])
        .unwrap_err()
        .to_string();
    assert!(e.contains("'y'") && e.contains("outside [0, 5)"), "{e}");
}

/// An integer tensor never crosses a stage boundary; the partition does not cut there.
#[test]
fn an_integer_tensor_stays_inside_its_stage() {
    let case = &cases()[8];
    let g = build(case);
    let report = graph::validate(&g).unwrap();
    assert!(report.integer_tensors.contains_key("i"));
    let pos = report.order.iter().position(|o| o == "i").unwrap();
    let n = report.order.len() as u32;
    let e = graph::stages::stages(&g, &report, &[0, pos as u32 + 1, n]).unwrap_err();
    assert!(e.contains("integer tensor"), "{e}");
    // A capacity that forces several stages: every bound falls where no integer crosses.
    let total = graph::stages::stage_memory(
        &g,
        &report,
        &graph::stages::stages(&g, &report, &[0, n]).unwrap()[0],
        2,
    );
    let bounds = graph::stages::partition(&g, &report, 2, total * 2 / 3).unwrap();
    assert!(bounds.len() > 2, "{bounds:?}");
    graph::stages::stages(&g, &report, &bounds).unwrap();
}

#[test]
fn cost_memory_determinism_and_profile() {
    let report = |g: &ModelGraphV1| graph::validate(g).unwrap();
    let node =
        |r: &graph::GraphReport, o: &str| r.nodes.iter().find(|c| c.output == o).unwrap().clone();
    let r = report(&build(&cases()[0]));
    // ArgTopK over 7: 7·⌈log₂ 7⌉ = 21 per line, 3 lines; an index counts 8 bytes.
    assert_eq!(node(&r, "i").training_flops_per_example, 3 * 3 * 7 * 3);
    assert_eq!(node(&r, "i").activation_bytes_per_example, 8 * 3 * 3);
    // The gather keeps its index: 3·3 values + 3·3 indices of 8 bytes.
    assert_eq!(node(&r, "pred").activation_bytes_per_example, 4 * 9 + 8 * 9);
    let r = report(&build(&cases()[3]));
    assert_eq!(node(&r, "i").training_flops_per_example, 3 * 3 * 15);
    assert_eq!(
        r.integer_tensors["i"],
        graph::IntRange {
            bound: Some(5),
            distinct_axis: None
        }
    );
    for case in cases() {
        let g = build(&case);
        assert_eq!(graph::memory::required_profile(&g), 8, "{}", case.label);
        // The memory estimate bounds the executor's measurement.
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
        // Segmented execution (recomputation) keeps integers across segment bounds: same bytes.
        let seg = exec::gradient_cpu_with(
            &prepared,
            &params,
            &inputs,
            0,
            exec::ExecOptions {
                segments: 3,
                offload_parameters: false,
            },
        )
        .unwrap();
        for (k, t) in &step.gradients {
            assert_eq!(
                t, &seg.gradients[k],
                "{}: {k} differs with segments",
                case.label
            );
        }
    }
    // Descriptions without integers stay at their level.
    let plain = simple(
        r#"{"op":"ReduceSum","inputs":["h"],"output":"pred","attrs":{"axis":1}}"#,
        &[4],
        &[1],
        vec![],
    );
    assert_eq!(graph::memory::required_profile(&plain), 7);
}

/// Mixtures of experts captured by the PyTorch driver (`gg-python/tools/make_torch_fixtures.py`): a
/// Mixtral through transformers' `batched_mm` experts (top-2 routing, the chosen experts' weights
/// gathered per token) and the corpus' hand-written top-2 gating (a scatter of the softmax of the
/// two best scores). The loss and the gradient of every trained parameter, computed by PyTorch,
/// travel with the fixture: the executor must find the same ones — the router's included, which
/// receives its gradient through the values of the top-k only.
#[test]
fn captured_mixtures_of_experts_train_like_pytorch() {
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
    for name in ["mixtral-mini", "moe-top2-mini"] {
        let g: ModelGraphV1 =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap())
                .unwrap();
        let torch = bundle(format!("{name}.torch.ggb"));
        let torch_grad = bundle(format!("{name}.grad.ggb"));
        let prepared = Prepared::new(&g).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(graph::memory::required_profile(&g), 8, "{name}");
        let params: BTreeMap<String, TensorF32> = g
            .parameters
            .iter()
            .map(|p| (p.name.clone(), torch[&p.name].clone()))
            .collect();
        let inputs = BTreeMap::from([
            (g.inputs[0].name.clone(), torch["__x"].clone()),
            ("y".to_string(), torch["__y"].clone()),
        ]);
        let step = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
        let expected_loss = torch["__loss"].data[0];
        assert!(
            (step.loss - expected_loss).abs() <= 1e-5 * expected_loss.abs().max(1.0),
            "{name}: loss {} versus {expected_loss} in PyTorch",
            step.loss
        );
        assert_eq!(step.gradients.len(), torch_grad.len(), "{name}");
        let mut worst = 0f64;
        for (p, grad) in &step.gradients {
            let r = &torch_grad[p];
            assert_eq!(grad.shape, r.shape, "{name}: {p}");
            let scale = r.data.iter().fold(1e-3f64, |m, v| m.max(v.abs() as f64));
            let gap = grad
                .data
                .iter()
                .zip(&r.data)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0f64, f64::max)
                / scale;
            worst = worst.max(gap);
            assert!(
                gap <= 1e-5,
                "{name}: gradient of {p} departs from PyTorch's by {gap:e}"
            );
        }
        eprintln!(
            "{name}: {} nodes, loss {} (PyTorch {expected_loss}), worst relative gradient gap {worst:.1e}",
            g.nodes.len(),
            step.loss
        );
        let again = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
        for (k, t) in &step.gradients {
            assert_eq!(t, &again.gradients[k], "{name}: {k} not reproducible");
        }
    }
}

/// `Gelu` with `approximate = "tanh"` computes the tanh form (PyTorch's
/// `gelu(approximate="tanh")`, used by Gemma and GPT-2). Before `gg-ops/8` the attribute was
/// accepted and the exact form computed instead: found by executing the corpus at other batch
/// sizes (Gemma departed from PyTorch by 1.01e-5).
#[test]
fn gelu_tanh_is_the_tanh_form() {
    let tanh_gelu = |x: f64| {
        0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    };
    let case = Case {
        label: "gelu tanh",
        input: vec![3, 8],
        params: vec![],
        nodes: r#"{"op":"Gelu","inputs":["h"],"output":"pred","attrs":{"approximate":"tanh"}}"#
            .into(),
        output: vec![3, 8],
        reference: Box::new(move |h, _| T64 {
            shape: h.shape.clone(),
            data: h.data.iter().map(|v| tanh_gelu(*v * 1.0)).collect(),
        }),
    };
    let (value, grad) = run(&case, 77);
    eprintln!("gelu tanh: value {value:.1e}, gradient {grad:.1e}");
    let bad = simple(
        r#"{"op":"Gelu","inputs":["h"],"output":"pred","attrs":{"approximate":"fast"}}"#,
        &[4],
        &[4],
        vec![],
    );
    assert!(
        graph::validate(&bad)
            .unwrap_err()
            .to_string()
            .contains("approximate")
    );
    assert_eq!(graph::memory::required_profile(&build(&case)), 8);
}
