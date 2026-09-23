//! Execution of a validated graph with Burn (ADR-0016, slice 3): the executor is checked against
//! the already admitted `mlp.grad.v1` kernel, then against finite differences on the operators no
//! existing kernel covers (convolution, pooling, attention, normalizations).

use std::collections::BTreeMap;

use burn::backend::flex::FlexDevice;
use burn::backend::{Autodiff, Flex};
use gg_compute::graph::exec::{self, ExecError, Prepared};
use gg_compute::graph::templates::{TransformerDims, transformer};
use gg_compute::graph::{self, ModelGraphV1};
use gg_compute::mlp::Weights;
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

type B = Autodiff<Flex>;

fn device() -> FlexDevice {
    FlexDevice
}

fn parse(json: &str) -> ModelGraphV1 {
    serde_json::from_str(json).expect("readable description")
}

fn random(shape: &[u64], seed: u64, scale: f32) -> TensorF32 {
    let mut d = Drbg::new("test/graph-exec", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(
        shape.to_vec(),
        (0..n).map(|_| d.next_sym_f32() * scale).collect(),
    )
    .unwrap()
}

fn classes(n: u64, count: u64, seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/graph-exec/classes", seed);
    TensorF32::new(
        vec![n],
        (0..n).map(|_| d.next_index(count as u32) as f32).collect(),
    )
    .unwrap()
}

/// One class per token, for language model skeletons.
fn token_classes(n: u64, seq: u64, count: u64, seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/graph-exec/token-classes", seed);
    TensorF32::new(
        vec![n, seq],
        (0..n * seq)
            .map(|_| d.next_index(count as u32) as f32)
            .collect(),
    )
    .unwrap()
}

fn run(
    graph: &ModelGraphV1,
    params: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    seed: u64,
) -> exec::StepResult {
    let prepared = Prepared::new(graph).expect("validated graph");
    exec::gradient::<B>(&device(), &prepared, params, inputs, seed).expect("execution")
}

/// Checks the components with the largest gradient against central finite differences.
fn check_finite_differences(
    graph: &ModelGraphV1,
    params: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    h: f32,
    abs: f32,
    rel: f32,
) {
    let base = run(graph, params, inputs, 7);
    let mut checked = 0;
    for (name, g) in &base.gradients {
        let mut order: Vec<usize> = (0..g.data.len()).collect();
        order.sort_by(|a, b| g.data[*b].abs().total_cmp(&g.data[*a].abs()));
        for &i in order.iter().take(3) {
            let mut plus = params.clone();
            plus.get_mut(name).unwrap().data[i] += h;
            let mut minus = params.clone();
            minus.get_mut(name).unwrap().data[i] -= h;
            let lp = run(graph, &plus, inputs, 7).loss;
            let lm = run(graph, &minus, inputs, 7).loss;
            let fd = (lp - lm) / (2.0 * h);
            let ad = g.data[i];
            assert!(
                (fd - ad).abs() <= abs + rel * ad.abs(),
                "{name}[{i}]: finite differences {fd}, autodiff {ad}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 6, "too few components checked: {checked}");
}

#[test]
fn the_perceptron_gradient_matches_the_admitted_kernel() {
    // relu(x·w1 + b1)·w2 + b2, mean squared error: exactly `mlp.grad.v1`.
    let (n, d_in, d_h, d_out) = (16u64, 6u64, 10u64, 3u64);
    let weights = Weights {
        w1: random(&[d_in, d_h], 1, 0.5),
        b1: random(&[d_h], 2, 0.1),
        w2: random(&[d_h, d_out], 3, 0.5),
        b2: random(&[d_out], 4, 0.1),
    };
    let x = random(&[n, d_in], 5, 1.0);
    let y = random(&[n, d_out], 6, 1.0);
    let (reference, ref_loss) =
        gg_compute::grad::gradient::<B>(&device(), &weights, &x, &y).expect("admitted kernel");

    let graph = parse(&format!(
        r#"{{
      "format": "gg-graph/v1", "name": "mlp-like-the-kernel",
      "inputs": [{{"name": "x", "shape": [0, {d_in}]}}],
      "parameters": [
        {{"name": "w1", "shape": [{d_in}, {d_h}], "init": "zeros"}},
        {{"name": "b1", "shape": [{d_h}], "init": "zeros"}},
        {{"name": "w2", "shape": [{d_h}, {d_out}], "init": "zeros"}},
        {{"name": "b2", "shape": [{d_out}], "init": "zeros"}}
      ],
      "nodes": [
        {{"op": "MatMul", "inputs": ["x", "w1"], "output": "h"}},
        {{"op": "Add", "inputs": ["h", "b1"], "output": "hb"}},
        {{"op": "Relu", "inputs": ["hb"], "output": "a"}},
        {{"op": "MatMul", "inputs": ["a", "w2"], "output": "o"}},
        {{"op": "Add", "inputs": ["o", "b2"], "output": "pred"}}
      ],
      "loss": {{"mean_squared_error": {{"prediction": "pred", "target": {{"name": "y", "shape": [0, {d_out}]}}}}}}
    }}"#
    ));
    let params = BTreeMap::from([
        ("w1".to_string(), weights.w1.clone()),
        ("b1".to_string(), weights.b1.clone()),
        ("w2".to_string(), weights.w2.clone()),
        ("b2".to_string(), weights.b2.clone()),
    ]);
    let inputs = BTreeMap::from([("x".to_string(), x), ("y".to_string(), y)]);
    let step = run(&graph, &params, &inputs, 0);

    assert!(
        (step.loss - ref_loss).abs() <= 1e-6,
        "loss {} versus {ref_loss}",
        step.loss
    );
    for (name, reference) in [
        ("w1", &reference.w1),
        ("b1", &reference.b1),
        ("w2", &reference.w2),
        ("b2", &reference.b2),
    ] {
        let got = &step.gradients[name];
        assert_eq!(got.shape, reference.shape, "{name}");
        let max = got
            .data
            .iter()
            .zip(&reference.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max <= 1e-6, "gradient of {name}: max deviation {max}");
    }
}

#[test]
fn two_runs_are_identical_byte_for_byte() {
    let graph = transformer(TransformerDims {
        layers: 1,
        dim: 8,
        heads: 2,
        seq: 4,
        vocab: 11,
        ffn: 16,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    })
    .unwrap();
    let params = exec::init_parameters(&graph, 42);
    let n = 3;
    let mut ids = Drbg::new("test/ids", 1);
    let inputs = BTreeMap::from([
        (
            "ids".to_string(),
            TensorF32::new(
                vec![n, 4],
                (0..n * 4).map(|_| ids.next_index(11) as f32).collect(),
            )
            .unwrap(),
        ),
        ("y".to_string(), token_classes(n, 4, 11, 2)),
    ]);
    let a = run(&graph, &params, &inputs, 9);
    let b = run(&graph, &params, &inputs, 9);
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    for (k, g) in &a.gradients {
        assert_eq!(g.encode(), b.gradients[k].encode(), "{k}");
    }
}

#[test]
fn finite_differences_on_a_transformer_block() {
    // Gather, RMSNorm, rank-3 by rank-2 MatMul, Reshape, Transpose, rank-4 MatMul,
    // Softmax, Gelu, Add, cross-entropy.
    let graph = transformer(TransformerDims {
        layers: 1,
        dim: 8,
        heads: 2,
        seq: 4,
        vocab: 11,
        ffn: 16,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    })
    .unwrap();
    let mut params = exec::init_parameters(&graph, 3);
    // Slightly larger weights so that the loss visibly depends on each of them.
    for t in params.values_mut() {
        for v in &mut t.data {
            *v *= 4.0;
        }
    }
    let n = 3;
    let mut ids = Drbg::new("test/ids", 4);
    let inputs = BTreeMap::from([
        (
            "ids".to_string(),
            TensorF32::new(
                vec![n, 4],
                (0..n * 4).map(|_| ids.next_index(11) as f32).collect(),
            )
            .unwrap(),
        ),
        ("y".to_string(), token_classes(n, 4, 11, 5)),
    ]);
    check_finite_differences(&graph, &params, &inputs, 1e-2, 2e-2, 5e-2);
}

#[test]
fn finite_differences_on_a_cnn() {
    // Conv2d with bias and padding, MaxPool2d, BatchNorm2d, AvgPool2d, Flatten, MatMul, Relu.
    let graph = parse(
        r#"{
      "format": "gg-graph/v1", "name": "tiny-cnn",
      "inputs": [{"name": "x", "shape": [0, 2, 6, 6]}],
      "parameters": [
        {"name": "k1", "shape": [3, 2, 3, 3], "init": "xavier_uniform"},
        {"name": "c1", "shape": [3], "init": "zeros"},
        {"name": "gamma", "shape": [3], "init": "ones"},
        {"name": "beta", "shape": [3], "init": "zeros"},
        {"name": "k2", "shape": [4, 3, 2, 2], "init": "xavier_uniform"},
        {"name": "wd", "shape": [4, 5], "init": "xavier_uniform"},
        {"name": "bd", "shape": [5], "init": "zeros"}
      ],
      "nodes": [
        {"op": "Conv2d", "inputs": ["x", "k1", "c1"], "output": "conv1", "attrs": {"padding": [1, 1]}},
        {"op": "BatchNorm2d", "inputs": ["conv1", "gamma", "beta"], "output": "bn"},
        {"op": "Relu", "inputs": ["bn"], "output": "r1"},
        {"op": "MaxPool2d", "inputs": ["r1"], "output": "p1", "attrs": {"kernel": [2, 2]}},
        {"op": "Conv2d", "inputs": ["p1", "k2"], "output": "conv2", "attrs": {"stride": [1, 1]}},
        {"op": "Tanh", "inputs": ["conv2"], "output": "t2"},
        {"op": "AvgPool2d", "inputs": ["t2"], "output": "p2", "attrs": {"kernel": [2, 2]}},
        {"op": "Flatten", "inputs": ["p2"], "output": "f", "attrs": {"axis": 1}},
        {"op": "MatMul", "inputs": ["f", "wd"], "output": "h"},
        {"op": "Add", "inputs": ["h", "bd"], "output": "logits"}
      ],
      "loss": {"cross_entropy": {"prediction": "logits", "target": {"name": "y", "shape": [0], "dtype": "i64"}}}
    }"#,
    );
    let report = graph::validate(&graph).unwrap();
    assert_eq!(report.shapes["p2"], "[N, 4, 1, 1]");
    let mut params = exec::init_parameters(&graph, 8);
    for t in params.values_mut() {
        for v in &mut t.data {
            *v *= 3.0;
        }
    }
    let n = 4;
    let inputs = BTreeMap::from([
        ("x".to_string(), random(&[n, 2, 6, 6], 9, 1.0)),
        ("y".to_string(), classes(n, 5, 10)),
    ]);
    check_finite_differences(&graph, &params, &inputs, 1e-2, 2e-2, 5e-2);
}

#[test]
fn normalizations_concat_dropout_and_arithmetic() {
    let graph = parse(
        r#"{
      "format": "gg-graph/v1", "name": "mix",
      "inputs": [{"name": "x", "shape": [0, 6]}],
      "parameters": [
        {"name": "g", "shape": [6], "init": "ones"},
        {"name": "b", "shape": [6], "init": "zeros"},
        {"name": "w", "shape": [12, 4], "init": "xavier_uniform"},
        {"name": "s", "shape": [1, 4], "init": {"normal": {"std": 0.5}}}
      ],
      "nodes": [
        {"op": "LayerNorm", "inputs": ["x", "g", "b"], "output": "ln", "attrs": {"epsilon": 0.001}},
        {"op": "Sigmoid", "inputs": ["x"], "output": "sg"},
        {"op": "Mul", "inputs": ["ln", "sg"], "output": "m"},
        {"op": "Sub", "inputs": ["m", "x"], "output": "d"},
        {"op": "Concat", "inputs": ["d", "ln"], "output": "c", "attrs": {"axis": 1}},
        {"op": "Dropout", "inputs": ["c"], "output": "dr", "attrs": {"rate": 0.5}},
        {"op": "MatMul", "inputs": ["dr", "w"], "output": "h"},
        {"op": "Div", "inputs": ["h", "s"], "output": "pred"}
      ],
      "loss": {"mean_squared_error": {"prediction": "pred", "target": {"name": "y", "shape": [0, 4]}}}
    }"#,
    );
    let mut params = exec::init_parameters(&graph, 11);
    // The divisor must not come close to zero.
    for v in &mut params.get_mut("s").unwrap().data {
        *v = 1.0 + v.abs();
    }
    let n = 5;
    let inputs = BTreeMap::from([
        ("x".to_string(), random(&[n, 6], 12, 1.0)),
        ("y".to_string(), random(&[n, 4], 13, 1.0)),
    ]);
    let a = run(&graph, &params, &inputs, 100);
    let b = run(&graph, &params, &inputs, 100);
    let c = run(&graph, &params, &inputs, 101);
    assert!(a.loss.is_finite());
    assert_eq!(
        a.gradients["w"].encode(),
        b.gradients["w"].encode(),
        "same seed, same mask"
    );
    assert_ne!(
        a.gradients["w"].encode(),
        c.gradients["w"].encode(),
        "different seed, different mask"
    );
    // Without Dropout, finite differences must hold on this mix.
    let mut no_dropout = graph.clone();
    for node in &mut no_dropout.nodes {
        if node.op == "Dropout" {
            node.attrs.insert("rate".into(), graph::Attr::Float(0.0));
        }
    }
    check_finite_differences(&no_dropout, &params, &inputs, 1e-2, 2e-2, 5e-2);
}

#[test]
fn initialization_is_reproducible_and_bounded() {
    let graph = transformer(TransformerDims {
        layers: 2,
        dim: 16,
        heads: 4,
        seq: 8,
        vocab: 50,
        ffn: 32,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    })
    .unwrap();
    let a = exec::init_parameters(&graph, 1);
    let b = exec::init_parameters(&graph, 1);
    let c = exec::init_parameters(&graph, 2);
    assert_eq!(a.len(), graph.parameters.len());
    for p in &graph.parameters {
        assert_eq!(a[&p.name].shape, p.shape);
        assert_eq!(a[&p.name].encode(), b[&p.name].encode(), "{}", p.name);
        assert_ne!(a[&p.name].encode(), c[&p.name].encode(), "{}", p.name);
    }
    // Xavier: bounded by 1/sqrt(fan_in); wk is [16, 16] so |v| ≤ 0.25.
    assert!(a["l0.wk"].data.iter().all(|v| v.abs() <= 0.25));
    // Normal with standard deviation 0.02 on the embedding: the empirical standard deviation is close.
    let e = &a["emb"].data;
    let mean = e.iter().sum::<f32>() / e.len() as f32;
    let std = (e.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / e.len() as f32).sqrt();
    assert!((std - 0.02).abs() < 0.004, "standard deviation {std}");
}

#[test]
fn errors_are_named() {
    let graph = parse(
        r#"{
      "format": "gg-graph/v1", "name": "small",
      "inputs": [{"name": "x", "shape": [0, 3]}],
      "parameters": [{"name": "w", "shape": [3, 2], "init": "ones"}],
      "nodes": [{"op": "MatMul", "inputs": ["x", "w"], "output": "pred"}],
      "loss": {"mean_squared_error": {"prediction": "pred", "target": {"name": "y", "shape": [0, 2]}}}
    }"#,
    );
    let prepared = Prepared::new(&graph).unwrap();
    let params = exec::init_parameters(&graph, 0);
    let ok_inputs = BTreeMap::from([
        ("x".to_string(), random(&[4, 3], 1, 1.0)),
        ("y".to_string(), random(&[4, 2], 2, 1.0)),
    ]);

    let mut missing = ok_inputs.clone();
    missing.remove("y");
    let err = exec::gradient::<B>(&device(), &prepared, &params, &missing, 0).unwrap_err();
    assert_eq!(err, ExecError::Missing("y".into()));

    let mut wrong = ok_inputs.clone();
    wrong.insert("y".to_string(), random(&[5, 2], 2, 1.0));
    let err = exec::gradient::<B>(&device(), &prepared, &params, &wrong, 0).unwrap_err();
    assert!(matches!(err, ExecError::Shape(_)), "{err:?}");
    assert!(err.to_string().contains("batch of 5"), "{err}");

    let mut bad_param = params.clone();
    bad_param.insert("w".to_string(), random(&[2, 3], 3, 1.0));
    let err = exec::gradient::<B>(&device(), &prepared, &bad_param, &ok_inputs, 0).unwrap_err();
    assert!(err.to_string().contains("'w'"), "{err}");

    let mut nan = ok_inputs.clone();
    nan.get_mut("x").unwrap().data[0] = f32::NAN;
    let err = exec::gradient::<B>(&device(), &prepared, &params, &nan, 0).unwrap_err();
    assert!(matches!(err, ExecError::Numeric(_)), "{err:?}");
}
