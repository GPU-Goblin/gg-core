//! Validation of models described by a graph (ADR-0016).
//!
//! The graphs are written in JSON, as a researcher would send them, not built in Rust: what is
//! tested here is the actual submission path.

use gg_compute::graph::{self, GraphError, ModelGraphV1};

fn parse(json: &str) -> ModelGraphV1 {
    serde_json::from_str(json).expect("readable JSON description")
}

/// Tabular perceptron: 64 inputs, two hidden layers, 10 classes.
fn mlp_json() -> String {
    r#"{
      "format": "gg-graph/v1",
      "name": "tabular-mlp",
      "inputs": [{"name": "x", "shape": [0, 64]}],
      "parameters": [
        {"name": "w1", "shape": [64, 256], "init": "xavier_uniform"},
        {"name": "b1", "shape": [1, 256], "init": "zeros"},
        {"name": "w2", "shape": [256, 128], "init": "xavier_uniform"},
        {"name": "b2", "shape": [1, 128], "init": "zeros"},
        {"name": "w3", "shape": [128, 10], "init": "xavier_uniform"},
        {"name": "b3", "shape": [1, 10], "init": "zeros"}
      ],
      "nodes": [
        {"op": "MatMul", "inputs": ["x", "w1"], "output": "h1"},
        {"op": "Add", "inputs": ["h1", "b1"], "output": "h1b"},
        {"op": "Relu", "inputs": ["h1b"], "output": "a1"},
        {"op": "MatMul", "inputs": ["a1", "w2"], "output": "h2"},
        {"op": "Add", "inputs": ["h2", "b2"], "output": "h2b"},
        {"op": "Relu", "inputs": ["h2b"], "output": "a2"},
        {"op": "MatMul", "inputs": ["a2", "w3"], "output": "h3"},
        {"op": "Add", "inputs": ["h3", "b3"], "output": "logits"}
      ],
      "loss": {"cross_entropy": {"prediction": "logits",
        "target": {"name": "y", "shape": [0], "dtype": "i64"}}}
    }"#
    .to_string()
}

#[test]
fn mlp_is_valid_and_its_cost_is_computed() {
    let report = graph::validate(&parse(&mlp_json())).expect("MLP accepted");
    // 64×256 + 256 + 256×128 + 128 + 128×10 + 10 = 50,826.
    assert_eq!(report.parameters, 50_826);
    assert_eq!(report.parameter_bytes, 203_304);
    // Matrix products: 2×(64×256 + 256×128 + 128×10) = 100,352 FLOPs per example.
    assert!(report.forward_flops_per_example > 100_000);
    assert_eq!(
        report.training_flops_per_example,
        report.forward_flops_per_example * 3
    );
    assert_eq!(report.order.len(), 8);
    assert_eq!(report.shapes["logits"], "[N, 10]");
    assert!(report.ops_used.contains(&"Relu".to_string()));
    assert_eq!(report.profile, graph::PROFILE_VERSION);
    // Memory for a batch of 128, excluding optimizer state.
    assert!(report.memory_bytes(128) > report.parameter_bytes * 2);
}

#[test]
fn execution_order_is_inferred_whatever_the_written_order() {
    let mut reversed = parse(&mlp_json());
    reversed.nodes.reverse();
    let report = graph::validate(&reversed).expect("the written order does not matter");
    assert_eq!(report.order.first().unwrap(), "h1");
    assert_eq!(report.order.last().unwrap(), "logits");
}

#[test]
fn vision_cnn_is_valid() {
    let json = r#"{
      "format": "gg-graph/v1",
      "name": "cnn-cifar",
      "inputs": [{"name": "x", "shape": [0, 3, 32, 32]}],
      "parameters": [
        {"name": "k1", "shape": [32, 3, 3, 3], "init": "xavier_uniform"},
        {"name": "k2", "shape": [64, 32, 3, 3], "init": "xavier_uniform"},
        {"name": "wd", "shape": [4096, 10], "init": "xavier_uniform"},
        {"name": "bd", "shape": [1, 10], "init": "zeros"}
      ],
      "nodes": [
        {"op": "Conv2d", "inputs": ["x", "k1"], "output": "c1", "attrs": {"padding": [1, 1]}},
        {"op": "Relu", "inputs": ["c1"], "output": "r1"},
        {"op": "MaxPool2d", "inputs": ["r1"], "output": "p1", "attrs": {"kernel": [2, 2]}},
        {"op": "Conv2d", "inputs": ["p1", "k2"], "output": "c2", "attrs": {"padding": [1, 1]}},
        {"op": "Relu", "inputs": ["c2"], "output": "r2"},
        {"op": "MaxPool2d", "inputs": ["r2"], "output": "p2", "attrs": {"kernel": [2, 2]}},
        {"op": "Flatten", "inputs": ["p2"], "output": "f", "attrs": {"axis": 1}},
        {"op": "MatMul", "inputs": ["f", "wd"], "output": "h"},
        {"op": "Add", "inputs": ["h", "bd"], "output": "logits"}
      ],
      "loss": {"cross_entropy": {"prediction": "logits",
        "target": {"name": "y", "shape": [0], "dtype": "i64"}}}
    }"#;
    let report = graph::validate(&parse(json)).expect("CNN accepted");
    assert_eq!(report.shapes["c1"], "[N, 32, 32, 32]");
    assert_eq!(report.shapes["p1"], "[N, 32, 16, 16]");
    assert_eq!(report.shapes["p2"], "[N, 64, 8, 8]");
    assert_eq!(report.shapes["f"], "[N, 4096]");
    // Convolution dominates: far more FLOPs than parameters.
    assert!(report.forward_flops_per_example > 10 * report.parameters);
}

#[test]
fn transformer_block_is_valid() {
    // Single-head attention over a sequence of 128 tokens, width 256.
    let json = r#"{
      "format": "gg-graph/v1",
      "name": "transformer-block",
      "inputs": [{"name": "ids", "shape": [0, 128], "dtype": "i64"}],
      "parameters": [
        {"name": "emb", "shape": [32000, 256], "init": {"normal": {"std": 0.02}}},
        {"name": "wq", "shape": [256, 256], "init": "xavier_uniform"},
        {"name": "wk", "shape": [256, 256], "init": "xavier_uniform"},
        {"name": "wv", "shape": [256, 256], "init": "xavier_uniform"},
        {"name": "wo", "shape": [256, 256], "init": "xavier_uniform"},
        {"name": "wu", "shape": [256, 1024], "init": "xavier_uniform"},
        {"name": "wd", "shape": [1024, 256], "init": "xavier_uniform"},
        {"name": "wlm", "shape": [256, 32000], "init": "xavier_uniform"}
      ],
      "nodes": [
        {"op": "Gather", "inputs": ["emb", "ids"], "output": "h0"},
        {"op": "RMSNorm", "inputs": ["h0"], "output": "n0"},
        {"op": "MatMul", "inputs": ["n0", "wq"], "output": "q"},
        {"op": "MatMul", "inputs": ["n0", "wk"], "output": "k"},
        {"op": "MatMul", "inputs": ["n0", "wv"], "output": "v"},
        {"op": "Transpose", "inputs": ["k"], "output": "kt", "attrs": {"perm": [0, 2, 1]}},
        {"op": "MatMul", "inputs": ["q", "kt"], "output": "scores"},
        {"op": "Softmax", "inputs": ["scores"], "output": "weights", "attrs": {"axis": 2}},
        {"op": "MatMul", "inputs": ["weights", "v"], "output": "ctx"},
        {"op": "MatMul", "inputs": ["ctx", "wo"], "output": "attn"},
        {"op": "Add", "inputs": ["h0", "attn"], "output": "h1"},
        {"op": "RMSNorm", "inputs": ["h1"], "output": "n1"},
        {"op": "MatMul", "inputs": ["n1", "wu"], "output": "up"},
        {"op": "Gelu", "inputs": ["up"], "output": "act"},
        {"op": "MatMul", "inputs": ["act", "wd"], "output": "down"},
        {"op": "Add", "inputs": ["h1", "down"], "output": "h2"},
        {"op": "MatMul", "inputs": ["h2", "wlm"], "output": "seqlogits"},
        {"op": "Reshape", "inputs": ["seqlogits"], "output": "logits", "attrs": {"shape": [-1, 32000]}}
      ],
      "loss": {"cross_entropy": {"prediction": "logits",
        "target": {"name": "y", "shape": [0, 128], "dtype": "i64"}}}
    }"#;
    let report = graph::validate(&parse(json)).expect("transformer block accepted");
    assert_eq!(report.shapes["h0"], "[N, 128, 256]");
    assert_eq!(report.shapes["scores"], "[N, 128, 128]");
    assert_eq!(report.shapes["logits"], "[N, 32000]");
    // Tolerance operators are flagged: the validator will not be able to require bit-exact results.
    assert!(report.tolerance_ops.contains(&"Softmax".to_string()));
    assert!(report.parameters > 16_000_000);
}

#[test]
fn time_series_is_valid() {
    // 48-step window, 7 channels, regression of one future step.
    let json = r#"{
      "format": "gg-graph/v1",
      "name": "series-horizon-1",
      "inputs": [{"name": "x", "shape": [0, 7, 48]}],
      "parameters": [
        {"name": "k1", "shape": [32, 7, 5, 1], "init": "xavier_uniform"},
        {"name": "wo", "shape": [1408, 1], "init": "xavier_uniform"}
      ],
      "nodes": [
        {"op": "Reshape", "inputs": ["x"], "output": "x4", "attrs": {"shape": [-1, 7, 48, 1]}},
        {"op": "Conv2d", "inputs": ["x4", "k1"], "output": "c", "attrs": {"stride": [1, 1]}},
        {"op": "Tanh", "inputs": ["c"], "output": "a"},
        {"op": "Flatten", "inputs": ["a"], "output": "f", "attrs": {"axis": 1}},
        {"op": "MatMul", "inputs": ["f", "wo"], "output": "yhat"}
      ],
      "loss": {"mean_squared_error": {"prediction": "yhat",
        "target": {"name": "y", "shape": [0, 1]}}}
    }"#;
    let report = graph::validate(&parse(json)).expect("time series accepted");
    assert_eq!(report.shapes["c"], "[N, 32, 44, 1]");
    assert_eq!(report.shapes["f"], "[N, 1408]");
    assert_eq!(report.shapes["yhat"], "[N, 1]");
}

#[test]
fn operator_outside_profile_is_named() {
    let json = mlp_json().replace(
        r#""op": "Relu", "inputs": ["h1b"]"#,
        r#""op": "Selu", "inputs": ["h1b"]"#,
    );
    let err = graph::validate(&parse(&json)).expect_err("Selu is not in the profile");
    match &err {
        GraphError::UnsupportedOp { op, node, .. } => {
            assert_eq!(op, "Selu");
            assert_eq!(node, "a1");
        }
        other => panic!("erreur inattendue : {other:?}"),
    }
    let message = err.to_string();
    assert!(message.contains("Selu"), "{message}");
    assert!(message.contains(graph::PROFILE_VERSION), "{message}");
}

#[test]
fn close_operator_is_suggested() {
    let json = mlp_json().replace(r#""op": "MatMul""#, r#""op": "matmul""#);
    let err = graph::validate(&parse(&json)).expect_err("case matters");
    let message = err.to_string();
    assert!(message.contains("MatMul"), "suggestion expected: {message}");
}

#[test]
fn incompatible_shapes_are_rejected() {
    let json = mlp_json().replace(
        r#"{"name": "w2", "shape": [256, 128]"#,
        r#"{"name": "w2", "shape": [512, 128]"#,
    );
    let err = graph::validate(&parse(&json)).expect_err("256 versus 512");
    let message = err.to_string();
    assert!(message.contains("MatMul"), "{message}");
    assert!(
        message.contains("256") && message.contains("512"),
        "{message}"
    );
}

#[test]
fn unknown_tensor_is_rejected() {
    let json = mlp_json().replace(r#""inputs": ["a1", "w2"]"#, r#""inputs": ["a0", "w2"]"#);
    let err = graph::validate(&parse(&json)).expect_err("a0 does not exist");
    assert_eq!(err, GraphError::Unknown("a0".into()));
    assert!(err.to_string().contains("a0"));
}

#[test]
fn cycle_is_rejected() {
    let json = r#"{
      "format": "gg-graph/v1",
      "name": "cycle",
      "inputs": [{"name": "x", "shape": [0, 4]}],
      "parameters": [{"name": "w", "shape": [4, 4], "init": "zeros"}],
      "nodes": [
        {"op": "Add", "inputs": ["x", "b"], "output": "a"},
        {"op": "Add", "inputs": ["a", "x"], "output": "b"},
        {"op": "MatMul", "inputs": ["b", "w"], "output": "out"}
      ],
      "loss": {"mean_squared_error": {"prediction": "out",
        "target": {"name": "y", "shape": [0, 4]}}}
    }"#;
    let err = graph::validate(&parse(json)).expect_err("a depends on b, which depends on a");
    match err {
        GraphError::Cycle(names) => assert!(names.contains(&"a".to_string())),
        other => panic!("erreur inattendue : {other:?}"),
    }
}

#[test]
fn unknown_or_missing_attribute_is_rejected() {
    let missing = mlp_json().replace(
        r#"{"op": "Relu", "inputs": ["h1b"], "output": "a1"}"#,
        r#"{"op": "Reshape", "inputs": ["h1b"], "output": "a1"}"#,
    );
    let err = graph::validate(&parse(&missing)).expect_err("Reshape requires shape");
    assert!(err.to_string().contains("shape"), "{err}");

    let unknown = mlp_json().replace(
        r#"{"op": "Relu", "inputs": ["h1b"], "output": "a1"}"#,
        r#"{"op": "Relu", "inputs": ["h1b"], "output": "a1", "attrs": {"slope": 0.1}}"#,
    );
    let err = graph::validate(&parse(&unknown)).expect_err("Relu takes no attributes");
    assert!(err.to_string().contains("slope"), "{err}");
}

#[test]
fn inconsistent_loss_is_rejected() {
    let json = mlp_json().replace(
        r#"{"cross_entropy": {"prediction": "logits",
        "target": {"name": "y", "shape": [0], "dtype": "i64"}}}"#,
        r#"{"mean_squared_error": {"prediction": "logits",
        "target": {"name": "y", "shape": [0, 3]}}}"#,
    );
    let err = graph::validate(&parse(&json)).expect_err("10 classes versus 3");
    assert!(err.to_string().contains("loss"), "{err}");

    let absent = mlp_json().replace(r#""prediction": "logits""#, r#""prediction": "output""#);
    let err = graph::validate(&parse(&absent)).expect_err("the prediction must exist");
    assert_eq!(err, GraphError::Unknown("output".into()));
}

#[test]
fn bounds_are_enforced() {
    let too_large = r#"{
      "format": "gg-graph/v1",
      "name": "too-large",
      "inputs": [{"name": "x", "shape": [0, 1048576]}],
      "parameters": [{"name": "w", "shape": [1048576, 1048576], "init": "zeros"}],
      "nodes": [{"op": "MatMul", "inputs": ["x", "w"], "output": "out"}],
      "loss": {"mean_squared_error": {"prediction": "out",
        "target": {"name": "y", "shape": [0, 1048576]}}}
    }"#;
    // 1.1 trillion parameters: beyond the maximum.
    let err = graph::validate(&parse(too_large)).expect_err("parameter bound");
    assert!(matches!(err, GraphError::Bound(_)), "{err:?}");

    let unknown_format = mlp_json().replace(r#""format": "gg-graph/v1""#, r#""format": "onnx/17""#);
    let err = graph::validate(&parse(&unknown_format)).expect_err("unknown format");
    assert!(err.to_string().contains("gg-graph/v1"), "{err}");
}

#[test]
fn published_profile_is_complete_and_stable() {
    let profile = graph::profile();
    // gg-ops/4 adds no operator (ADR-0022); gg-ops/5 adds ReduceMean and Rsqrt (ADR-0023);
    // gg-ops/7 adds eleven (ADR-0027), gg-ops/8 seven for integer tensors (ADR-0029).
    assert_eq!(profile.len(), 45, "the gg-ops/8 profile has 45 operators");
    assert_eq!(graph::PROFILE_VERSION, "gg-ops/8");
    for name in [
        "ArgMax",
        "ArgMin",
        "ArgTopK",
        "ArgSort",
        "Cast",
        "GatherElements",
        "ScatterElements",
        "ConvTranspose2d",
        "Pad",
        "Resize",
        "Flip",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "CumSum",
        "IndexSelect",
        "IndexAdd",
    ] {
        assert!(profile.iter().any(|o| o.name == name), "{name} missing");
    }
    assert!(profile.iter().any(|o| o.name == "ReduceMean"));
    assert!(profile.iter().any(|o| o.name == "Rsqrt"));
    assert!(profile.iter().any(|o| o.name == "CausalMask"));
    assert!(profile.iter().any(|o| o.name == "Slice"));
    for spec in &profile {
        assert!(!spec.summary.is_empty(), "{} has no summary", spec.name);
        assert!(spec.arity.0 <= spec.arity.1);
        for required in spec.required {
            assert!(
                spec.attributes.contains(required),
                "{}: '{required}' required but missing from the attributes",
                spec.name
            );
        }
    }
    // The profile is sent to clients as is.
    let json = serde_json::to_string(&profile).expect("serializable profile");
    assert!(json.contains("MatMul") && json.contains("Conv2d"));
}

/// A model constant (the 1/√d factor of an attention) is declared as a frozen parameter with a
/// constant initialization: the description computes the right function without a checkpoint.
#[test]
fn frozen_constant_is_initialized_to_its_value() {
    let json = r#"{
      "format": "gg-graph/v1", "name": "scale",
      "inputs": [{"name": "x", "shape": [0, 4]}],
      "parameters": [
        {"name": "w", "shape": [4, 4], "init": "xavier_uniform"},
        {"name": "scale", "shape": [1], "init": {"constant": {"value": 0.125}}, "trainable": false}
      ],
      "nodes": [
        {"op": "MatMul", "inputs": ["x", "w"], "output": "h"},
        {"op": "Mul", "inputs": ["h", "scale"], "output": "pred"}
      ],
      "loss": {"mean_squared_error": {"prediction": "pred", "target": {"name": "y", "shape": [0, 4]}}}
    }"#;
    let model = parse(json);
    let report = graph::validate(&model).expect("valid");
    assert_eq!(
        report.trainable_parameters, 16,
        "the constant is not trained"
    );
    let params = gg_compute::graph::exec::init_parameters(&model, 3);
    assert_eq!(params["scale"].data, vec![0.125]);

    // JSON cannot express a non-finite value: the case is built in memory, as a caller that
    // generates the description programmatically would.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad = model.clone();
        bad.parameters[1].init = gg_compute::graph::Init::Constant { value };
        let err = graph::validate(&bad).expect_err("non-finite constant rejected");
        assert!(err.to_string().contains("not finite"), "{err}");
    }
    let negative_std = json.replace(
        r#"{"constant": {"value": 0.125}}"#,
        r#"{"normal": {"std": -1.0}}"#,
    );
    let err =
        graph::validate(&parse(&negative_std)).expect_err("negative standard deviation rejected");
    assert!(err.to_string().contains("standard deviation"), "{err}");
}
