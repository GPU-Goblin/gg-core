//! Profile `gg-ops/6`: the `Formula` operator (ADR-0025), and the functions `gg-ops/7` adds to its
//! grammar (ADR-0027). Like every added operator, it brings
//! its oracle, its named rejections, its determinism test and its tolerance (ADR-0016).
//!
//! The oracle is twofold: for each expression, the value **and** the derivative are recomputed in
//! f64 from their mathematical definition, written here by hand, and compared with what the
//! executor returns (forward pass, then gradient by automatic differentiation).

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, Prepared};
use gg_compute::graph::{self, GraphError, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

const N: u64 = 6;
const D: u64 = 16;

fn random(shape: &[u64], seed: u64, scale: f32) -> TensorF32 {
    let mut d = Drbg::new("test/formula", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(
        shape.to_vec(),
        (0..n).map(|_| d.next_sym_f32() * scale).collect(),
    )
    .unwrap()
}

/// `pred = Formula(x, w)` with `x` an input `[N, D]` and `w` a broadcast parameter `[D]`.
fn graph_of(expr: &str) -> ModelGraphV1 {
    let json = format!(
        r#"{{"format":"gg-graph/v1","name":"f","inputs":[{{"name":"x","shape":[0,{D}]}}],
          "parameters":[{{"name":"w","shape":[{D}],"init":{{"normal":{{"std":1.0}}}}}}],
          "nodes":[{{"op":"Formula","inputs":["x","w"],"output":"pred","attrs":{{"expr":{}}}}}],
          "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,{D}]}}}}}}}}"#,
        serde_json::to_string(expr).unwrap()
    );
    serde_json::from_str(&json).unwrap()
}

type F = fn(f64, f64) -> f64;

/// Expression, value `f(a, b)`, derivative `∂f/∂b`, and the domain of `a` (scale, positive or not).
struct Case {
    expr: &'static str,
    f: F,
    df_db: F,
    positive: bool,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn erf(x: f64) -> f64 {
    // Taylor series, accurate to 1e-12 for |x| ≤ 3 (the test inputs).
    let mut sum = 0.0;
    let mut term = x;
    let mut n = 0.0;
    while term.abs() > 1e-17 || n < 5.0 {
        sum += term / (2.0 * n + 1.0);
        n += 1.0;
        term *= -x * x / n;
        if n > 200.0 {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            expr: "a * b + 0.5",
            f: |a, b| a * b + 0.5,
            df_db: |a, _| a,
            positive: false,
        },
        Case {
            expr: "(a - b) / (1 + b * b)",
            f: |a, b| (a - b) / (1.0 + b * b),
            df_db: |a, b| {
                let q = 1.0 + b * b;
                (-q - (a - b) * 2.0 * b) / (q * q)
            },
            positive: false,
        },
        Case {
            expr: "-b^3 + a",
            f: |a, b| -b * b * b + a,
            df_db: |_, b| -3.0 * b * b,
            positive: false,
        },
        Case {
            expr: "a * exp(b)",
            f: |a, b| a * b.exp(),
            df_db: |a, b| a * b.exp(),
            positive: false,
        },
        Case {
            expr: "log(a) * b",
            f: |a, b| a.ln() * b,
            df_db: |a, _| a.ln(),
            positive: true,
        },
        Case {
            expr: "a * sigmoid(b)",
            f: |a, b| a * sigmoid(b),
            df_db: |a, b| a * sigmoid(b) * (1.0 - sigmoid(b)),
            positive: false,
        },
        Case {
            expr: "softplus(a * b)",
            f: |a, b| (1.0 + (a * b).exp()).ln(),
            df_db: |a, b| a * sigmoid(a * b),
            positive: false,
        },
        Case {
            expr: "tanh(b) * a + sin(b) - cos(a * b)",
            f: |a, b| b.tanh() * a + b.sin() - (a * b).cos(),
            df_db: |a, b| a * (1.0 - b.tanh().powi(2)) + b.cos() + a * (a * b).sin(),
            positive: false,
        },
        Case {
            expr: "erf(b) + a",
            f: |a, b| erf(b) + a,
            df_db: |_, b| 2.0 / std::f64::consts::PI.sqrt() * (-b * b).exp(),
            positive: false,
        },
        Case {
            expr: "sqrt(a) * b + rsqrt(a) + log1p(a) * b",
            f: |a, b| a.sqrt() * b + 1.0 / a.sqrt() + a.ln_1p() * b,
            df_db: |a, _| a.sqrt() + a.ln_1p(),
            positive: true,
        },
        Case {
            expr: "a^0.5 * b + a^-2",
            f: |a, b| a.sqrt() * b + 1.0 / (a * a),
            df_db: |a, _| a.sqrt(),
            positive: true,
        },
        // gg-ops/7 (ADR-0027): inverse trigonometric and hyperbolic functions, atan2, rounding.
        Case {
            expr: "tan(b * 0.5) * a + atan(a * b)",
            f: |a, b| (b * 0.5).tan() * a + (a * b).atan(),
            df_db: |a, b| 0.5 * a / (b * 0.5).cos().powi(2) + a / (1.0 + (a * b).powi(2)),
            positive: false,
        },
        Case {
            expr: "asin(sigmoid(b) - 0.5) * a + acos(tanh(b) * 0.9)",
            f: |a, b| (sigmoid(b) - 0.5).asin() * a + (b.tanh() * 0.9).acos(),
            df_db: |a, b| {
                let s = sigmoid(b);
                let u = s - 0.5;
                let t = b.tanh() * 0.9;
                a * s * (1.0 - s) / (1.0 - u * u).sqrt()
                    - 0.9 * (1.0 - b.tanh().powi(2)) / (1.0 - t * t).sqrt()
            },
            positive: false,
        },
        Case {
            expr: "sinh(b) * a + cosh(a * b)",
            f: |a, b| b.sinh() * a + (a * b).cosh(),
            df_db: |a, b| b.cosh() * a + a * (a * b).sinh(),
            positive: false,
        },
        Case {
            expr: "asinh(a * b) + acosh(b * b + 1.5) + atanh(tanh(b) * 0.8)",
            f: |a, b| (a * b).asinh() + (b * b + 1.5).acosh() + (b.tanh() * 0.8).atanh(),
            df_db: |a, b| {
                let u = b * b + 1.5;
                let t = b.tanh() * 0.8;
                a / (1.0 + (a * b).powi(2)).sqrt()
                    + 2.0 * b / (u * u - 1.0).sqrt()
                    + 0.8 * (1.0 - b.tanh().powi(2)) / (1.0 - t * t)
            },
            positive: false,
        },
        Case {
            // `a` positive keeps (a, b) away from the origin, where atan2 is not differentiable.
            expr: "atan2(b, a) + atan2(2, b) * a",
            f: |a, b| b.atan2(a) + 2f64.atan2(b) * a,
            df_db: |a, b| a / (a * a + b * b) - 2.0 * a / (4.0 + b * b),
            positive: true,
        },
        Case {
            // Piecewise-constant functions: their derivative is zero, and the derivative with
            // respect to `b` is carried only by what multiplies them.
            expr: "floor(a * 3) * b + ceil(b) + round(a) * b + trunc(a * b) + sign(a - 0.5) * b",
            f: |a, b| {
                (a * 3.0).floor() * b
                    + b.ceil()
                    + a.round_ties_even() * b
                    + (a * b).trunc()
                    + (a - 0.5).signum() * b
            },
            df_db: |a, _| (a * 3.0).floor() + a.round_ties_even() + (a - 0.5).signum(),
            positive: false,
        },
        Case {
            expr: "max(a, b) + min(b, 0.25) + relu(b) * abs(a)",
            f: |a, b| a.max(b) + b.min(0.25) + b.max(0.0) * a.abs(),
            df_db: |a, b| {
                (if b > a { 1.0 } else { 0.0 })
                    + (if b < 0.25 { 1.0 } else { 0.0 })
                    + (if b > 0.0 { a.abs() } else { 0.0 })
            },
            positive: false,
        },
    ]
}

#[test]
fn each_expression_matches_its_definition_in_value_and_derivative() {
    for (k, case) in cases().iter().enumerate() {
        let g = graph_of(case.expr);
        let prepared = Prepared::new(&g).unwrap_or_else(|e| panic!("{}: {e}", case.expr));
        let params = exec::init_parameters(&g, 7);
        let mut x = random(&[N, D], 100 + k as u64, 2.0);
        if case.positive {
            for v in &mut x.data {
                *v = v.abs() + 0.1;
            }
        }
        let y = random(&[N, D], 200 + k as u64, 1.0);
        let inputs = BTreeMap::from([("x".to_string(), x.clone()), ("y".to_string(), y.clone())]);
        let w = &params["w"].data;

        // Value.
        let got = exec::forward_cpu(&prepared, &params, &inputs, 0, "pred").unwrap();
        assert_eq!(got.shape, vec![N, D]);
        let mut expected_grad = vec![0f64; D as usize];
        let mut loss = 0f64;
        let count = (N * D) as f64;
        let mut worst = 0f64;
        for i in 0..N as usize {
            for j in 0..D as usize {
                let (a, b) = (x.data[i * D as usize + j] as f64, w[j] as f64);
                let v = (case.f)(a, b);
                let g = got.data[i * D as usize + j] as f64;
                worst = worst.max((g - v).abs() / v.abs().max(1.0));
                let r = v - y.data[i * D as usize + j] as f64;
                loss += r * r / count;
                expected_grad[j] += 2.0 * r / count * (case.df_db)(a, b);
            }
        }
        assert!(
            worst <= 2e-6,
            "{}: value, relative error {worst:e}",
            case.expr
        );

        // Derivative: the gradient of the loss with respect to `w`.
        let step = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
        assert!(
            ((step.loss as f64) - loss).abs() <= 1e-5 * loss.max(1.0),
            "{}: loss {} versus {loss}",
            case.expr,
            step.loss
        );
        let grad = &step.gradients["w"].data;
        let scale = expected_grad.iter().fold(1e-3f64, |m, v| m.max(v.abs()));
        for (j, (g, e)) in grad.iter().zip(&expected_grad).enumerate() {
            assert!(
                ((*g as f64) - e).abs() <= 1e-5 * scale,
                "{}: gradient[{j}] {g} versus {e}",
                case.expr
            );
        }
    }
}

/// An integer power of a negative base is exact (repeated products), where `powf` would
/// return `NaN`; `softplus` stays finite for large arguments.
#[test]
fn integer_powers_and_softplus_are_stable() {
    let g = graph_of("b^3 + softplus(a * 100) * 0 + a * 0");
    let prepared = Prepared::new(&g).unwrap();
    let mut params = exec::init_parameters(&g, 1);
    params.get_mut("w").unwrap().data.fill(-2.0);
    let x = random(&[N, D], 3, 50.0);
    let got = exec::forward_cpu(
        &prepared,
        &params,
        &BTreeMap::from([("x".to_string(), x)]),
        0,
        "pred",
    )
    .unwrap();
    assert!(got.data.iter().all(|v| *v == -8.0), "{:?}", &got.data[..4]);

    let g = graph_of("softplus(a * b)");
    let prepared = Prepared::new(&g).unwrap();
    let mut params = exec::init_parameters(&g, 1);
    params.get_mut("w").unwrap().data.fill(1.0);
    let x = TensorF32::new(vec![1, D], vec![500.0; D as usize]).unwrap();
    let got = exec::forward_cpu(
        &prepared,
        &params,
        &BTreeMap::from([("x".to_string(), x)]),
        0,
        "pred",
    )
    .unwrap();
    assert!(got.data.iter().all(|v| *v == 500.0), "{:?}", &got.data[..4]);
}

/// `round` is half to even, as PyTorch, on every backend; `sign(0) = 0`; the derivative of a
/// piecewise-constant function is exactly zero.
#[test]
fn rounding_is_half_to_even_and_has_a_zero_derivative() {
    let values = [
        -2.5f32,
        -1.5,
        -0.5,
        0.5,
        1.5,
        2.5,
        3.5,
        0.499_999_97,
        -7.25,
        0.0,
        1e7 + 1.0,
    ];
    let n = values.len() as u64;
    let json = |expr: &str| {
        format!(
            r#"{{"format":"gg-graph/v1","name":"r","inputs":[{{"name":"x","shape":[0,{n}]}}],
              "parameters":[{{"name":"w","shape":[{n}],"init":"zeros"}}],
              "nodes":[{{"op":"Formula","inputs":["x","w"],"output":"pred","attrs":{{"expr":"{expr}"}}}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,{n}]}}}}}}}}"#
        )
    };
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            TensorF32::new(vec![1, n], values.to_vec()).unwrap(),
        ),
        (
            "y".to_string(),
            TensorF32::new(vec![1, n], vec![0.5; n as usize]).unwrap(),
        ),
    ]);
    for (expr, f) in [
        ("round(a + b)", f32::round_ties_even as fn(f32) -> f32),
        ("floor(a + b)", f32::floor),
        ("ceil(a + b)", f32::ceil),
        ("trunc(a + b)", f32::trunc),
        (
            "sign(a + b)",
            |v: f32| if v == 0.0 { 0.0 } else { v.signum() },
        ),
    ] {
        let g: ModelGraphV1 = serde_json::from_str(&json(expr)).unwrap();
        let prepared = Prepared::new(&g).unwrap();
        let params = exec::init_parameters(&g, 0);
        let got = exec::forward_cpu(&prepared, &params, &inputs, 0, "pred").unwrap();
        let want: Vec<f32> = values.iter().map(|v| f(*v)).collect();
        assert_eq!(got.data, want, "{expr}");
        let step = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
        assert!(
            step.gradients["w"].data.iter().all(|v| *v == 0.0),
            "{expr}: {:?}",
            step.gradients["w"].data
        );
    }
}

/// Two runs give the same bytes, for both value and gradient.
#[test]
fn bit_for_bit_deterministic() {
    let g = graph_of("tanh(a * b) * softplus(b) + erf(a) / (1 + b^2)");
    let prepared = Prepared::new(&g).unwrap();
    let params = exec::init_parameters(&g, 3);
    let inputs = BTreeMap::from([
        ("x".to_string(), random(&[N, D], 5, 2.0)),
        ("y".to_string(), random(&[N, D], 6, 1.0)),
    ]);
    let a = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
    let b = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    assert_eq!(a.gradients["w"].data, b.gradients["w"].data);
}

/// A formula is equivalent to the profile operators it replaces: `a·sigmoid(a)` (SiLU) versus
/// `Mul(x, Sigmoid(x))`, up to rounding.
#[test]
fn a_formula_equals_the_composition_of_operators() {
    let head = format!(
        r#""format":"gg-graph/v1","name":"s","inputs":[{{"name":"x","shape":[0,{D}]}}],
        "parameters":[{{"name":"w","shape":[{D},{D}],"init":"xavier_uniform"}}],
        "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,{D}]}}}}}}"#
    );
    let formula: ModelGraphV1 = serde_json::from_str(&format!(
        r#"{{{head},"nodes":[{{"op":"MatMul","inputs":["x","w"],"output":"h"}},
            {{"op":"Formula","inputs":["h"],"output":"pred","attrs":{{"expr":"a * sigmoid(a)"}}}}]}}"#
    ))
    .unwrap();
    let ops: ModelGraphV1 = serde_json::from_str(&format!(
        r#"{{{head},"nodes":[{{"op":"MatMul","inputs":["x","w"],"output":"h"}},
            {{"op":"Sigmoid","inputs":["h"],"output":"s"}},
            {{"op":"Mul","inputs":["h","s"],"output":"pred"}}]}}"#
    ))
    .unwrap();
    let params = exec::init_parameters(&ops, 9);
    let inputs = BTreeMap::from([
        ("x".to_string(), random(&[N, D], 10, 1.0)),
        ("y".to_string(), random(&[N, D], 11, 1.0)),
    ]);
    let a = exec::gradient_cpu(&Prepared::new(&formula).unwrap(), &params, &inputs, 0).unwrap();
    let b = exec::gradient_cpu(&Prepared::new(&ops).unwrap(), &params, &inputs, 0).unwrap();
    assert!((a.loss - b.loss).abs() <= 1e-7);
    let worst = a.gradients["w"]
        .data
        .iter()
        .zip(&b.gradients["w"].data)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(worst <= 1e-6, "difference {worst:e}");
}

#[test]
fn rejections_are_named() {
    let rejection = |expr: &str| match graph::validate(&graph_of(expr)) {
        Err(GraphError::Shape(m)) => m,
        other => panic!("{expr}: expected a rejection, got {other:?}"),
    };
    assert!(rejection("a + b + c").contains("only has 2"));
    assert!(rejection("a * 2").contains("never used"));
    assert!(rejection("system(a, b)").contains("neither an input"));
    assert!(rejection("a +").contains("incomplete"));
    // The attribute is mandatory and must be text.
    let mut g = graph_of("a * b");
    g.nodes[0].attrs.remove("expr");
    let e = graph::validate(&g).unwrap_err();
    assert!(e.to_string().contains("expr"), "{e}");
    let mut g = graph_of("a * b");
    g.nodes[0].attrs.insert("expr".into(), graph::Attr::Int(3));
    assert!(
        graph::validate(&g)
            .unwrap_err()
            .to_string()
            .contains("string")
    );
    // The input shapes must broadcast.
    let mut g = graph_of("a * b");
    g.parameters[0].shape = vec![D + 1];
    assert!(
        graph::validate(&g)
            .unwrap_err()
            .to_string()
            .contains("do not broadcast")
    );
}

/// The cost follows from the expression: FLOPs per element, and one activation kept per step.
#[test]
fn cost_and_memory_follow_the_expression() {
    let simple = graph::validate(&graph_of("a * b")).unwrap();
    let long = graph::validate(&graph_of("tanh(a * b) + exp(b) * a")).unwrap();
    assert_eq!(simple.forward_flops_per_example, D);
    // Mul, tanh, exp, Mul, Add: 1 + 4 + 4 + 1 + 1 per element.
    assert_eq!(long.forward_flops_per_example, 11 * D);
    assert_eq!(simple.activation_bytes_per_example, 4 * D);
    assert_eq!(long.activation_bytes_per_example, 5 * 4 * D);
    assert_eq!(graph::memory::required_profile(&graph_of("a * b")), 6);
}
