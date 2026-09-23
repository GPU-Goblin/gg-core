//! Shared harness of the operator tests (`graph_ops7.rs`, `graph_ops8.rs`): a graph `h = x + p`
//! followed by the nodes under test, an f64 definition of what they compute, and the check of
//! value, loss, gradient (against central finite differences of that definition) and bit-for-bit
//! determinism.
#![allow(dead_code)]

use std::collections::BTreeMap;

use gg_compute::graph::ModelGraphV1;
use gg_compute::graph::exec::{self, Prepared};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

/// Batch of the tests.
pub const N: usize = 2;

/// A tensor in f64, row-major.
#[derive(Clone, Debug)]
pub struct T64 {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl T64 {
    pub fn zeros(shape: &[usize]) -> Self {
        T64 {
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        }
    }
    pub fn of(t: &TensorF32) -> Self {
        T64 {
            shape: t.shape.iter().map(|d| *d as usize).collect(),
            data: t.data.iter().map(|v| *v as f64).collect(),
        }
    }
    pub fn offset(&self, idx: &[usize]) -> usize {
        idx.iter()
            .zip(&self.shape)
            .fold(0, |acc, (i, s)| acc * s + i)
    }
    pub fn get(&self, idx: &[usize]) -> f64 {
        self.data[self.offset(idx)]
    }
    pub fn add_at(&mut self, idx: &[usize], v: f64) {
        let o = self.offset(idx);
        self.data[o] += v;
    }
}

/// Calls `f` on every multi-index of `shape`, in row-major order.
pub fn each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    let total: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..total {
        f(&idx);
        for a in (0..shape.len()).rev() {
            idx[a] += 1;
            if idx[a] < shape[a] {
                break;
            }
            idx[a] = 0;
        }
    }
}

pub fn random(shape: &[u64], seed: u64, scale: f32) -> TensorF32 {
    let mut d = Drbg::new("test/ops7", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(
        shape.to_vec(),
        (0..n).map(|_| d.next_sym_f32() * scale).collect(),
    )
    .unwrap()
}

pub type Reference = dyn Fn(&T64, &BTreeMap<String, T64>) -> T64;

/// A test graph: `h = x + p` (`x` an input `[N, …]`, `p` a trained parameter `[1, …]`), then the
/// nodes under test, which read `h` and the extra parameters and produce `pred`.
pub struct Case {
    pub label: &'static str,
    /// Shape of `x` without the batch.
    pub input: Vec<u64>,
    /// Extra trained parameters: name and shape.
    pub params: Vec<(&'static str, Vec<u64>)>,
    /// Nodes, as JSON, after `h`.
    pub nodes: String,
    /// Shape of `pred` without the batch.
    pub output: Vec<u64>,
    /// `pred` from `h` and the parameters, in f64.
    pub reference: Box<Reference>,
}

pub fn dims(batch: bool, s: &[u64]) -> String {
    let mut v: Vec<String> = s.iter().map(u64::to_string).collect();
    v.insert(0, if batch { "0".into() } else { "1".into() });
    v.join(",")
}

pub fn build(case: &Case) -> ModelGraphV1 {
    let mut params = vec![format!(
        r#"{{"name":"p","shape":[{}],"init":{{"normal":{{"std":1.0}}}}}}"#,
        dims(false, &case.input)
    )];
    for (name, shape) in &case.params {
        let s: Vec<String> = shape.iter().map(u64::to_string).collect();
        params.push(format!(
            r#"{{"name":"{name}","shape":[{}],"init":{{"normal":{{"std":0.5}}}}}}"#,
            s.join(",")
        ));
    }
    let json = format!(
        r#"{{"format":"gg-graph/v1","name":"{}","inputs":[{{"name":"x","shape":[{}]}}],
          "parameters":[{}],
          "nodes":[{{"op":"Add","inputs":["x","p"],"output":"h"}},{}],
          "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[{}]}}}}}}}}"#,
        case.label
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>(),
        dims(true, &case.input),
        params.join(","),
        case.nodes,
        dims(true, &case.output)
    );
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("{}: {e}\n{json}", case.label))
}

pub fn batch_shape(s: &[u64]) -> Vec<u64> {
    let mut v = s.to_vec();
    v.insert(0, N as u64);
    v
}

/// `h = x + p`, in f64, `p` broadcast over the batch.
pub fn h_of(x: &TensorF32, p: &TensorF32) -> T64 {
    let x = T64::of(x);
    let per = p.data.len();
    T64 {
        shape: x.shape.clone(),
        data: x
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| v + p.data[i % per] as f64)
            .collect(),
    }
}

pub fn loss_of(pred: &T64, y: &T64) -> f64 {
    pred.data
        .iter()
        .zip(&y.data)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / pred.data.len() as f64
}

/// Runs a case: value, loss, gradient of every parameter against finite differences of the f64
/// definition, and bit-for-bit determinism. Returns the worst relative deviations (value,
/// gradient).
pub fn run(case: &Case, seed: u64) -> (f64, f64) {
    let graph = build(case);
    let prepared = Prepared::new(&graph).unwrap_or_else(|e| panic!("{}: {e}", case.label));
    let report = prepared.report();
    assert_eq!(
        report.shapes["pred"],
        format!(
            "[N, {}]",
            case.output
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        "{}: inferred shape",
        case.label
    );
    let params = exec::init_parameters(&graph, seed);
    let x = random(&batch_shape(&case.input), seed + 1, 1.0);
    let y = random(&batch_shape(&case.output), seed + 2, 1.0);
    let inputs = BTreeMap::from([("x".to_string(), x.clone()), ("y".to_string(), y.clone())]);

    let extra = |params: &BTreeMap<String, TensorF32>| -> BTreeMap<String, T64> {
        params
            .iter()
            .filter(|(k, _)| k.as_str() != "p")
            .map(|(k, v)| (k.clone(), T64::of(v)))
            .collect()
    };
    let expected = (case.reference)(&h_of(&x, &params["p"]), &extra(&params));
    let got = exec::forward_cpu(&prepared, &params, &inputs, 0, "pred").unwrap();
    assert_eq!(
        got.shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
        expected.shape,
        "{}: executed shape",
        case.label
    );
    let scale = expected.data.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    let value_gap = got
        .data
        .iter()
        .zip(&expected.data)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0, f64::max)
        / scale;
    assert!(
        value_gap <= 2e-6,
        "{}: value, relative deviation {value_gap:e}",
        case.label
    );

    // Loss and gradients against central differences of the f64 definition.
    let y64 = T64::of(&y);
    let step = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
    let loss = |params: &BTreeMap<String, TensorF32>, name: &str, k: usize, delta: f64| {
        let mut e = extra(params);
        let mut hx = h_of(&x, &params["p"]);
        if name == "p" {
            let per = params["p"].data.len();
            for (i, v) in hx.data.iter_mut().enumerate() {
                if i % per == k {
                    *v += delta;
                }
            }
        } else {
            e.get_mut(name).unwrap().data[k] += delta;
        }
        loss_of(&(case.reference)(&hx, &e), &y64)
    };
    let l0 = loss(&params, "p", 0, 0.0);
    assert!(
        (step.loss as f64 - l0).abs() <= 1e-5 * l0.max(1.0),
        "{}: loss {} versus {l0}",
        case.label,
        step.loss
    );
    let mut grad_gap = 0f64;
    for (name, g) in &step.gradients {
        let h = 1e-5;
        let fd: Vec<f64> = (0..g.data.len())
            .map(|k| (loss(&params, name, k, h) - loss(&params, name, k, -h)) / (2.0 * h))
            .collect();
        let gscale = fd.iter().fold(1e-3f64, |m, v| m.max(v.abs()));
        for (k, (a, b)) in g.data.iter().zip(&fd).enumerate() {
            let d = (*a as f64 - b).abs() / gscale;
            grad_gap = grad_gap.max(d);
            assert!(
                d <= 2e-5,
                "{}: gradient of {name}[{k}] {a} versus {b} (relative {d:e})",
                case.label
            );
        }
    }

    // Determinism: two runs, same bytes.
    let again = exec::gradient_cpu(&prepared, &params, &inputs, 0).unwrap();
    assert_eq!(step.loss.to_bits(), again.loss.to_bits(), "{}", case.label);
    for (k, t) in &step.gradients {
        assert_eq!(
            t, &again.gradients[k],
            "{}: {k} not reproducible",
            case.label
        );
    }
    (value_gap, grad_gap)
}
