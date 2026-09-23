//! Profile `gg-ops/5`: `ReduceMean` and `Rsqrt` (ADR-0023). Each added operator brings its
//! oracle, its named rejections, its determinism test and its tolerance (ADR-0016). The strongest
//! oracle is the composition they serve: Hugging Face's hand-written RMSNorm
//! (`x·rsqrt(mean(x²) + ε)·g`) must equal the profile's `RMSNorm` operator, in value **and** in
//! gradient.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, Prepared};
use gg_compute::graph::{self, GraphError, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

fn random(shape: &[u64], seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/reduce", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(shape.to_vec(), (0..n).map(|_| d.next_sym_f32()).collect()).unwrap()
}

fn one_op(op: &str, attrs: &str, input: &[u64], output: &[u64]) -> ModelGraphV1 {
    let dims = |s: &[u64]| {
        std::iter::once("0".to_string())
            .chain(s[1..].iter().map(u64::to_string))
            .collect::<Vec<_>>()
            .join(",")
    };
    let json = format!(
        r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[{}]}}],
          "parameters":[{{"name":"w","shape":[1],"init":"ones"}}],
          "nodes":[{{"op":"Mul","inputs":["x","w"],"output":"xs"}},{{"op":"{op}","inputs":["xs"],"output":"out"{attrs}}}],
          "loss":{{"mean_squared_error":{{"prediction":"out","target":{{"name":"y","shape":[{}]}}}}}}}}"#,
        dims(input),
        dims(output)
    );
    serde_json::from_str(&json).unwrap()
}

fn forward(graph: &ModelGraphV1, x: &TensorF32) -> TensorF32 {
    let prepared = Prepared::new(graph).expect("valid");
    let params = exec::init_parameters(graph, 0);
    exec::forward_cpu(
        &prepared,
        &params,
        &BTreeMap::from([("x".to_string(), x.clone())]),
        0,
        "out",
    )
    .unwrap()
}

#[test]
fn reducemean_matches_its_definition() {
    let shape = [3u64, 4, 5, 6];
    let x = random(&shape, 1);
    let strides = [4 * 5 * 6, 5 * 6, 6, 1];
    for axis in [1usize, 2, 3] {
        for keep in [1, 0] {
            let mut out: Vec<u64> = shape.to_vec();
            if keep == 1 {
                out[axis] = 1;
            } else {
                out.remove(axis);
            }
            // Negative axis for the last one: same result.
            let written = if axis == 3 { -1 } else { axis as i64 };
            let g = one_op(
                "ReduceMean",
                &format!(r#","attrs":{{"axis":{written},"keepdims":{keep}}}"#),
                &shape,
                &out,
            );
            let got = forward(&g, &x);
            assert_eq!(got.shape, out, "axis {axis}, keepdims {keep}");
            // Naive oracle in f64.
            let mut worst = 0f64;
            let mut k = 0;
            for i0 in 0..shape[0] {
                for i1 in 0..if axis == 1 { 1 } else { shape[1] } {
                    for i2 in 0..if axis == 2 { 1 } else { shape[2] } {
                        for i3 in 0..if axis == 3 { 1 } else { shape[3] } {
                            let mut sum = 0f64;
                            for r in 0..shape[axis] {
                                let mut idx = [i0, i1, i2, i3];
                                idx[axis] = r;
                                let flat: u64 = idx.iter().zip(strides).map(|(a, s)| a * s).sum();
                                sum += x.data[flat as usize] as f64;
                            }
                            let expected = sum / shape[axis] as f64;
                            worst = worst.max((got.data[k] as f64 - expected).abs());
                            k += 1;
                        }
                    }
                }
            }
            assert_eq!(k, got.data.len());
            assert!(worst <= 1e-6, "axis {axis}: difference {worst:e}");
        }
    }
}

#[test]
fn rsqrt_matches_its_definition() {
    let mut x = random(&[4, 64], 2);
    for v in &mut x.data {
        *v = v.abs() * 100.0 + 1e-3;
    }
    let got = forward(&one_op("Rsqrt", "", &[4, 64], &[4, 64]), &x);
    let worst = got
        .data
        .iter()
        .zip(&x.data)
        .map(|(g, v)| ((*g as f64) * (*v as f64).sqrt() - 1.0).abs())
        .fold(0f64, f64::max);
    assert!(worst <= 3e-7, "relative error {worst:e}");
}

#[test]
fn rejections_are_named() {
    let shape_err = |g: ModelGraphV1| match graph::validate(&g) {
        Err(GraphError::Shape(m)) | Err(GraphError::Attribute(m)) => m,
        other => panic!("expected a rejection, got {other:?}"),
    };
    // Mean over the batch: it would mix examples.
    let m = shape_err(one_op(
        "ReduceMean",
        r#","attrs":{"axis":0}"#,
        &[3, 4],
        &[1, 4],
    ));
    assert!(m.contains("batch"), "{m}");
    let m = shape_err(one_op(
        "ReduceMean",
        r#","attrs":{"axis":5}"#,
        &[3, 4],
        &[3, 1],
    ));
    assert!(m.contains("outside the rank"), "{m}");
    let m = shape_err(one_op(
        "ReduceMean",
        r#","attrs":{"axis":1,"keepdims":2}"#,
        &[3, 4],
        &[3, 1],
    ));
    assert!(m.contains("keepdims"), "{m}");
    // The axis is mandatory.
    let e = graph::validate(&one_op("ReduceMean", "", &[3, 4], &[3, 1])).unwrap_err();
    assert!(e.to_string().contains("axis"), "{e}");
}

/// The hand-written RMSNorm equals the RMSNorm operator: value and gradient.
#[test]
fn composed_rmsnorm_equals_the_rmsnorm_operator() {
    let eps = 1e-6;
    let composed = format!(
        r#"{{"format":"gg-graph/v1","name":"hf","inputs":[{{"name":"x","shape":[0,5,16]}}],
          "parameters":[{{"name":"w1","shape":[16,16],"init":"xavier_uniform"}},{{"name":"g","shape":[16],"init":{{"normal":{{"std":1.0}}}}}},
                        {{"name":"eps","shape":[1],"init":{{"constant":{{"value":{eps}}}}},"trainable":false}},{{"name":"w2","shape":[16,4],"init":"xavier_uniform"}}],
          "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"h"}},
                   {{"op":"Mul","inputs":["h","h"],"output":"h2"}},
                   {{"op":"ReduceMean","inputs":["h2"],"output":"var","attrs":{{"axis":2}}}},
                   {{"op":"Add","inputs":["var","eps"],"output":"ve"}},
                   {{"op":"Rsqrt","inputs":["ve"],"output":"r"}},
                   {{"op":"Mul","inputs":["h","r"],"output":"hn"}},
                   {{"op":"Mul","inputs":["g","hn"],"output":"n"}},
                   {{"op":"MatMul","inputs":["n","w2"],"output":"pred"}}],
          "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,5,4]}}}}}}}}"#
    );
    let fused = format!(
        r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[0,5,16]}}],
          "parameters":[{{"name":"w1","shape":[16,16],"init":"xavier_uniform"}},{{"name":"g","shape":[16],"init":{{"normal":{{"std":1.0}}}}}},{{"name":"w2","shape":[16,4],"init":"xavier_uniform"}}],
          "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"h"}},
                   {{"op":"RMSNorm","inputs":["h","g"],"output":"n","attrs":{{"axis":-1,"epsilon":{eps}}}}},
                   {{"op":"MatMul","inputs":["n","w2"],"output":"pred"}}],
          "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,5,4]}}}}}}}}"#
    );
    let composed: ModelGraphV1 = serde_json::from_str(&composed).unwrap();
    let fused: ModelGraphV1 = serde_json::from_str(&fused).unwrap();
    let inputs = BTreeMap::from([
        ("x".to_string(), random(&[6, 5, 16], 3)),
        ("y".to_string(), random(&[6, 5, 4], 4)),
    ]);
    let pc = exec::init_parameters(&composed, 5);
    let mut pf = exec::init_parameters(&fused, 5);
    // Same starting values for the shared parameters.
    for k in ["w1", "g", "w2"] {
        pf.insert(k.into(), pc[k].clone());
    }
    let a = exec::gradient_cpu(&Prepared::new(&composed).unwrap(), &pc, &inputs, 0).unwrap();
    let b = exec::gradient_cpu(&Prepared::new(&fused).unwrap(), &pf, &inputs, 0).unwrap();
    assert!(
        (a.loss - b.loss).abs() <= 1e-6,
        "loss {} versus {}",
        a.loss,
        b.loss
    );
    for k in ["w1", "g", "w2"] {
        let worst = a.gradients[k]
            .data
            .iter()
            .zip(&b.gradients[k].data)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("gradient of {k}: difference {worst:e}");
        assert!(worst <= 1e-5, "{k}: difference {worst:e}");
    }
    // Determinism: two runs, same bytes.
    let again = exec::gradient_cpu(&Prepared::new(&composed).unwrap(), &pc, &inputs, 0).unwrap();
    for (k, t) in &a.gradients {
        assert_eq!(t, &again.gradients[k], "{k} not reproducible");
    }
}
