//! Description functions (ADR-0025): a named subgraph, called like an operator, then inlined by
//! the server. The expected proof: the inlined description computes **exactly** what the same
//! description written flat computes, byte for byte, for both value and gradient.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, Prepared};
use gg_compute::graph::functions::inline;
use gg_compute::graph::{self, GraphError, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

fn random(shape: &[u64], seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/functions", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(shape.to_vec(), (0..n).map(|_| d.next_sym_f32()).collect()).unwrap()
}

const HEAD: &str = r#""format":"gg-graph/v1","name":"blocks","inputs":[{"name":"x","shape":[0,5,8]}],
  "parameters":[
    {"name":"w0","shape":[8,8],"init":"xavier_uniform"},{"name":"g0","shape":[8],"init":"ones"},
    {"name":"w1","shape":[8,8],"init":"xavier_uniform"},{"name":"g1","shape":[8],"init":"ones"},
    {"name":"w2","shape":[8,8],"init":"xavier_uniform"},{"name":"g2","shape":[8],"init":"ones"},
    {"name":"out","shape":[8,3],"init":"xavier_uniform"}],
  "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,5,3]}}}"#;

/// Three residual blocks `h + act(RMSNorm(h)·W)`, where `act` is itself a function (SiLU
/// written as a `Formula`).
fn with_functions() -> ModelGraphV1 {
    serde_json::from_str(&format!(
        r#"{{{HEAD},
  "functions":[
    {{"name":"act","inputs":["v"],"output":"s",
      "nodes":[{{"op":"Formula","inputs":["v"],"output":"s","attrs":{{"expr":"a * sigmoid(a)"}}}}]}},
    {{"name":"block","inputs":["h","w","g"],"output":"r",
      "nodes":[{{"op":"RMSNorm","inputs":["h","g"],"output":"n","attrs":{{"epsilon":1e-6}}}},
               {{"op":"MatMul","inputs":["n","w"],"output":"m"}},
               {{"op":"fn:act","inputs":["m"],"output":"a"}},
               {{"op":"Add","inputs":["h","a"],"output":"r"}}]}}],
  "nodes":[{{"op":"fn:block","inputs":["x","w0","g0"],"output":"h1"}},
           {{"op":"fn:block","inputs":["h1","w1","g1"],"output":"h2"}},
           {{"op":"fn:block","inputs":["h2","w2","g2"],"output":"h3"}},
           {{"op":"MatMul","inputs":["h3","out"],"output":"pred"}}]}}"#
    ))
    .unwrap()
}

/// The same thing, written flat by hand.
fn flat_by_hand() -> ModelGraphV1 {
    let mut nodes = String::new();
    let mut prev = "x".to_string();
    for i in 0..3 {
        let h = format!("h{}", i + 1);
        nodes.push_str(&format!(
            r#"{{"op":"RMSNorm","inputs":["{prev}","g{i}"],"output":"n{i}","attrs":{{"epsilon":1e-6}}}},
               {{"op":"MatMul","inputs":["n{i}","w{i}"],"output":"m{i}"}},
               {{"op":"Formula","inputs":["m{i}"],"output":"a{i}","attrs":{{"expr":"a * sigmoid(a)"}}}},
               {{"op":"Add","inputs":["{prev}","a{i}"],"output":"{h}"}},"#
        ));
        prev = h;
    }
    serde_json::from_str(&format!(
        r#"{{{HEAD},"nodes":[{nodes}{{"op":"MatMul","inputs":["h3","out"],"output":"pred"}}]}}"#
    ))
    .unwrap()
}

#[test]
fn a_description_with_functions_computes_exactly_the_same_as_flat() {
    let g = with_functions();
    // Not inlined, it is rejected with a message naming what to do.
    let e = graph::validate(&g).unwrap_err();
    assert!(e.to_string().contains("inlined"), "{e}");

    let flat = inline(&g).unwrap();
    assert!(flat.functions.is_empty());
    assert_eq!(flat.nodes.len(), 13);
    // Inner names are prefixed by the call's output; the function's output keeps the call's
    // name; nested calls nest.
    let names: Vec<&str> = flat.nodes.iter().map(|n| n.output.as_str()).collect();
    assert_eq!(&names[..4], &["h1/n", "h1/m", "h1/a", "h1"]);
    assert_eq!(flat.nodes[2].inputs, vec!["h1/m"]);
    assert_eq!(flat.nodes[4].inputs, vec!["h1", "g1"]);

    let by_hand = flat_by_hand();
    let ra = graph::validate(&flat).unwrap();
    let rb = graph::validate(&by_hand).unwrap();
    assert_eq!(ra.training_flops_per_example, rb.training_flops_per_example);
    assert_eq!(
        ra.activation_bytes_per_example,
        rb.activation_bytes_per_example
    );
    assert_eq!(ra.parameter_bytes, rb.parameter_bytes);

    let params = exec::init_parameters(&flat, 4);
    assert_eq!(params, exec::init_parameters(&by_hand, 4));
    let inputs = BTreeMap::from([
        ("x".to_string(), random(&[4, 5, 8], 1)),
        ("y".to_string(), random(&[4, 5, 3], 2)),
    ]);
    let a = exec::gradient_cpu(&Prepared::new(&flat).unwrap(), &params, &inputs, 0).unwrap();
    let b = exec::gradient_cpu(&Prepared::new(&by_hand).unwrap(), &params, &inputs, 0).unwrap();
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    assert_eq!(a.gradients, b.gradients);
}

#[test]
fn inlining_is_deterministic_and_leaves_descriptions_without_functions_unchanged() {
    let g = with_functions();
    let x = serde_json::to_vec(&inline(&g).unwrap()).unwrap();
    let y = serde_json::to_vec(&inline(&g).unwrap()).unwrap();
    assert_eq!(x, y);
    // Without functions, the field does not appear: earlier descriptions keep their bytes.
    let flat = flat_by_hand();
    let text = serde_json::to_string(&flat).unwrap();
    assert!(!text.contains("functions"));
    assert_eq!(inline(&flat).unwrap(), flat);
}

fn rejection(functions: &str, nodes: &str) -> String {
    let g: ModelGraphV1 = serde_json::from_str(&format!(
        r#"{{{HEAD},"functions":{functions},"nodes":{nodes}}}"#
    ))
    .unwrap();
    match inline(&g).and_then(|f| graph::validate(&f).map(|_| ())) {
        Err(e) => e.to_string(),
        Ok(()) => panic!("expected a rejection"),
    }
}

#[test]
fn rejections_are_named() {
    let id = r#"{"name":"id","inputs":["v"],"output":"o","nodes":[{"op":"Formula","inputs":["v"],"output":"o","attrs":{"expr":"a*1"}}]}"#;
    let call = |op: &str, ins: &str| {
        format!(
            r#"[{{"op":"{op}","inputs":{ins},"output":"h"}},{{"op":"MatMul","inputs":["h","w0"],"output":"k"}},
               {{"op":"MatMul","inputs":["k","out"],"output":"pred"}}]"#
        )
    };
    // Unknown function.
    let m = rejection(&format!("[{id}]"), &call("fn:other", r#"["x"]"#));
    assert!(m.contains("'other' is not declared"), "{m}");
    // Number of inputs.
    let m = rejection(&format!("[{id}]"), &call("fn:id", r#"["x","w0"]"#));
    assert!(m.contains("expects 1 inputs (v), 2 given"), "{m}");
    // Open function: reads a global parameter without receiving it.
    let open = r#"{"name":"f","inputs":["v"],"output":"o","nodes":[{"op":"MatMul","inputs":["v","w0"],"output":"o"}]}"#;
    let m = rejection(&format!("[{open}]"), &call("fn:f", r#"["x"]"#));
    assert!(m.contains("'w0' is neither one of its inputs"), "{m}");
    // Recursion: a function does not call itself.
    let rec = r#"{"name":"r","inputs":["v"],"output":"o","nodes":[{"op":"fn:r","inputs":["v"],"output":"o"}]}"#;
    let m = rejection(&format!("[{rec}]"), &call("fn:r", r#"["x"]"#));
    assert!(m.contains("never itself"), "{m}");
    // Output not produced.
    let empty = r#"{"name":"v","inputs":["v"],"output":"z","nodes":[{"op":"Relu","inputs":["v"],"output":"o"}]}"#;
    let m = rejection(&format!("[{empty}]"), &call("fn:v", r#"["x"]"#));
    assert!(m.contains("output 'z'"), "{m}");
    // Attribute on a call.
    let m = rejection(
        &format!("[{id}]"),
        r#"[{"op":"fn:id","inputs":["x"],"output":"h","attrs":{"axis":1}},{"op":"MatMul","inputs":["h","out"],"output":"pred"}]"#,
    );
    assert!(m.contains("takes no attributes"), "{m}");
    // Declared twice.
    let m = rejection(&format!("[{id},{id}]"), &call("fn:id", r#"["x"]"#));
    assert!(m.contains("declared twice"), "{m}");
    // An operator outside the profile in the body is rejected at validation, by name.
    let off_profile = r#"{"name":"h","inputs":["v"],"output":"o","nodes":[{"op":"Cosh","inputs":["v"],"output":"o"}]}"#;
    let m = rejection(&format!("[{off_profile}]"), &call("fn:h", r#"["x"]"#));
    assert!(m.contains("'Cosh' (node 'h')"), "{m}");
}

/// Nested calls that double at each level quickly reach the node bound: inlining stops as soon
/// as it is exceeded, without building the rest.
#[test]
fn inlining_is_bounded() {
    let mut functions = vec![
        r#"{"name":"f0","inputs":["v"],"output":"o","nodes":[{"op":"Relu","inputs":["v"],"output":"o"}]}"#.to_string(),
    ];
    for i in 1..20 {
        functions.push(format!(
            r#"{{"name":"f{i}","inputs":["v"],"output":"o","nodes":[{{"op":"fn:f{}","inputs":["v"],"output":"p"}},{{"op":"fn:f{}","inputs":["p"],"output":"o"}}]}}"#,
            i - 1,
            i - 1
        ));
    }
    let g: ModelGraphV1 = serde_json::from_str(&format!(
        r#"{{{HEAD},"functions":[{}],"nodes":[{{"op":"fn:f19","inputs":["x"],"output":"h"}},{{"op":"MatMul","inputs":["h","out"],"output":"pred"}}]}}"#,
        functions.join(",")
    ))
    .unwrap();
    match inline(&g) {
        Err(GraphError::Bound(m)) => assert!(m.contains("4096"), "{m}"),
        other => panic!("expected a bound error, got {other:?}"),
    }
}
