//! Position encoding (profile `gg-ops/3`, ADR-0021): the `Slice` operator, the generated position
//! tables, the RoPE-style rotation composed of profile operators, and the measurement that
//! justifies it all — a task that depends on order.

use std::collections::BTreeMap;

use gg_compute::grad::CoordOptim;
use gg_compute::graph::exec::{self, Prepared, position_table};
use gg_compute::graph::task::{NamedOptimizerState, apply_step_named};
use gg_compute::graph::templates::{Positions, TransformerDims, rotary, transformer};
use gg_compute::graph::{
    self, FORMAT, GraphError, Init, Loss, ModelGraphV1, ParamSpec, Rotary, TensorSpec,
};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

fn random(shape: &[u64], seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/positions", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(shape.to_vec(), (0..n).map(|_| d.next_sym_f32()).collect()).unwrap()
}

fn slice_graph(shape: &[u64], axis: i64, start: i64, end: i64) -> ModelGraphV1 {
    let dims: Vec<String> = shape
        .iter()
        .enumerate()
        .map(|(i, d)| if i == 0 { "0".into() } else { d.to_string() })
        .collect();
    let mut out = shape.to_vec();
    out[0] = 0;
    let a = if axis < 0 {
        axis + shape.len() as i64
    } else {
        axis
    } as usize;
    out[a] = (end - start) as u64;
    let outs: Vec<String> = out.iter().map(|d| d.to_string()).collect();
    serde_json::from_str(&format!(
        r#"{{"format":"gg-graph/v1","name":"slice",
        "inputs":[{{"name":"x","shape":[{}]}}],
        "parameters":[{{"name":"w","shape":[1],"init":"ones"}}],
        "nodes":[{{"op":"Mul","inputs":["x","w"],"output":"s"}},
                 {{"op":"Slice","inputs":["s"],"output":"out","attrs":{{"axis":{axis},"start":{start},"end":{end}}}}}],
        "loss":{{"mean_squared_error":{{"prediction":"out","target":{{"name":"y","shape":[{}]}}}}}}}}"#,
        dims.join(","),
        outs.join(",")
    ))
    .unwrap()
}

#[test]
fn slice_matches_its_definition_to_the_byte() {
    let shape = [3u64, 4, 6, 5];
    let x = random(&shape, 1);
    let ones = BTreeMap::from([("w".to_string(), TensorF32::new(vec![1], vec![1.0]).unwrap())]);
    for (axis, start, end) in [
        (1i64, 1i64, 3i64),
        (2, 0, 6),
        (2, 2, 5),
        (3, 4, 5),
        (-1, 0, 2),
    ] {
        let g = slice_graph(&shape, axis, start, end);
        let p = Prepared::new(&g).unwrap_or_else(|e| panic!("axis {axis}: {e}"));
        let got = exec::forward_cpu(
            &p,
            &ones,
            &BTreeMap::from([("x".to_string(), x.clone())]),
            0,
            "out",
        )
        .unwrap();
        let a = if axis < 0 { axis + 4 } else { axis } as usize;
        // Naive oracle: walk the output and read back the corresponding input element.
        let mut expected = Vec::new();
        let strides = [
            shape[1] * shape[2] * shape[3],
            shape[2] * shape[3],
            shape[3],
            1,
        ];
        let mut idx = [0u64; 4];
        let mut out_shape = shape;
        out_shape[a] = (end - start) as u64;
        for i in 0..out_shape.iter().product::<u64>() {
            let mut r = i;
            for k in (0..4).rev() {
                idx[k] = r % out_shape[k];
                r /= out_shape[k];
            }
            let mut src = idx;
            src[a] += start as u64;
            let flat: u64 = src.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
            expected.push(x.data[flat as usize]);
        }
        assert_eq!(got.shape, out_shape.to_vec(), "axis {axis}");
        assert_eq!(
            got.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "axis {axis}, [{start}, {end})"
        );
    }
}

#[test]
fn malformed_slice_is_rejected_by_name() {
    let refuse = |axis: i64, start: i64, end: i64, needle: &str| {
        let json = format!(
            r#"{{"format":"gg-graph/v1","name":"t","inputs":[{{"name":"x","shape":[0,4,6]}}],
            "parameters":[{{"name":"w","shape":[1],"init":"ones"}}],
            "nodes":[{{"op":"Mul","inputs":["x","w"],"output":"s"}},{{"op":"Slice","inputs":["s"],"output":"out","attrs":{{"axis":{axis},"start":{start},"end":{end}}}}}],
            "loss":{{"mean_squared_error":{{"prediction":"out","target":{{"name":"y","shape":[0,4,2]}}}}}}}}"#
        );
        let g: ModelGraphV1 = serde_json::from_str(&json).unwrap();
        match graph::validate(&g) {
            Err(GraphError::Shape(m)) => {
                assert!(m.contains(needle), "\"{m}\" does not mention {needle}")
            }
            other => panic!("expected a rejection for axis {axis} [{start}, {end}): {other:?}"),
        }
    };
    refuse(0, 0, 2, "batch");
    refuse(2, 4, 8, "outside");
    refuse(2, 3, 3, "outside");
    refuse(5, 0, 1, "rank");
}

/// The generated tables equal their formula, computed independently here in f64.
#[test]
fn position_tables_equal_their_formula() {
    let (t_len, d, base) = (16usize, 8usize, 10000.0f64);
    let shape = [1u64, 1, t_len as u64, d as u64];
    let omega = |i: usize| base.powf(-(2.0 * i as f64) / d as f64);
    for (init, f) in [
        (
            Init::RotaryCos(Rotary::from_base(base)),
            Box::new(|t: usize, j: usize| (t as f64 * omega(j % (d / 2))).cos())
                as Box<dyn Fn(usize, usize) -> f64>,
        ),
        (
            Init::RotarySin(Rotary::from_base(base)),
            Box::new(|t: usize, j: usize| (t as f64 * omega(j % (d / 2))).sin()),
        ),
        (
            Init::Sinusoidal { base },
            Box::new(|t: usize, j: usize| {
                let a = t as f64 * omega(j / 2);
                if j.is_multiple_of(2) {
                    a.sin()
                } else {
                    a.cos()
                }
            }),
        ),
    ] {
        let table = position_table(&init, &shape);
        assert_eq!(table.len(), t_len * d);
        let mut worst = 0.0f64;
        for t in 0..t_len {
            for j in 0..d {
                worst = worst.max((table[t * d + j] as f64 - f(t, j)).abs());
            }
        }
        assert!(
            worst <= 1e-5,
            "{init:?}: difference {worst} from the formula"
        );
        // Position 0: no rotation.
        if let Init::RotaryCos(_) = init {
            assert!(table[..d].iter().all(|v| *v == 1.0));
        }
    }
}

#[test]
fn malformed_table_is_rejected() {
    let make = |shape: Vec<u64>, init: Init| ModelGraphV1 {
        format: FORMAT.into(),
        name: "t".into(),
        inputs: vec![TensorSpec {
            name: "x".into(),
            shape: vec![0, 4],
            dtype: "f32".into(),
        }],
        parameters: vec![
            ParamSpec {
                name: "w".into(),
                shape: vec![4, 4],
                init: Init::Zeros,
                trainable: true,
            },
            ParamSpec {
                name: "tab".into(),
                shape,
                init,
                trainable: false,
            },
        ],
        nodes: vec![graph::Node {
            op: "MatMul".into(),
            inputs: vec!["x".into(), "w".into()],
            output: "pred".into(),
            attrs: BTreeMap::new(),
        }],
        functions: Vec::new(),
        loss: Loss::MeanSquaredError {
            prediction: "pred".into(),
            target: TensorSpec {
                name: "y".into(),
                shape: vec![0, 4],
                dtype: "f32".into(),
            },
        },
    };
    assert!(
        graph::validate(&make(
            vec![1, 16, 8],
            Init::RotaryCos(Rotary::from_base(10000.0))
        ))
        .is_ok()
    );
    for (shape, init, needle) in [
        (
            vec![2, 16, 8],
            Init::RotarySin(Rotary::from_base(10000.0)),
            "[1, …, 1, T, d]",
        ),
        (vec![16, 7], Init::Sinusoidal { base: 10000.0 }, "d even"),
        (vec![16, 8], Init::RotaryCos(Rotary::from_base(1.0)), "base"),
        (
            vec![16, 8],
            Init::RotaryCos(Rotary::from_base(f64::NAN)),
            "base",
        ),
    ] {
        let err = graph::validate(&make(shape, init)).expect_err("malformed table");
        assert!(err.to_string().contains(needle), "{err}");
    }
}

/// The property that defines RoPE: after rotation, the dot product between position m and
/// position n depends only on n − m. This is the oracle for the Slice + Concat + Mul + Add composition.
#[test]
fn rope_only_sees_the_distance_between_positions() {
    let (t_len, d) = (16u64, 8u64);
    let table = [1, 1, t_len, d];
    let graph = ModelGraphV1 {
        format: FORMAT.into(),
        name: "rope".into(),
        inputs: vec![TensorSpec {
            name: "x".into(),
            shape: vec![0, 1, t_len, d],
            dtype: "f32".into(),
        }],
        parameters: vec![
            ParamSpec {
                name: "rope.cos".into(),
                shape: table.to_vec(),
                init: Init::RotaryCos(Rotary::from_base(10000.0)),
                trainable: false,
            },
            ParamSpec {
                name: "rope.sin".into(),
                shape: table.to_vec(),
                init: Init::RotarySin(Rotary::from_base(10000.0)),
                trainable: false,
            },
            ParamSpec {
                name: "rope.neg".into(),
                shape: vec![1],
                init: Init::Constant { value: -1.0 },
                trainable: false,
            },
            ParamSpec {
                name: "w".into(),
                shape: vec![1],
                init: Init::Ones,
                trainable: true,
            },
        ],
        nodes: {
            let mut nodes = Vec::new();
            // A neutral trainable factor: validation rejects a graph that learns nothing.
            nodes.push(graph::Node {
                op: "Mul".into(),
                inputs: vec!["x".into(), "w".into()],
                output: "xs".into(),
                attrs: BTreeMap::new(),
            });
            rotary(&mut nodes, "xs", "rot", d);
            nodes
        },
        functions: Vec::new(),
        loss: Loss::MeanSquaredError {
            prediction: "rot".into(),
            target: TensorSpec {
                name: "y".into(),
                shape: vec![0, 1, t_len, d],
                dtype: "f32".into(),
            },
        },
    };
    let prepared = Prepared::new(&graph).expect("valid composition");
    let params = exec::init_parameters(&graph, 0);
    let u = random(&[d], 1).data;
    let v = random(&[d], 2).data;
    let rotated_dot = |m: usize, n: usize| -> f64 {
        let mut x = vec![0f32; (t_len * d) as usize];
        x[m * d as usize..(m + 1) * d as usize].copy_from_slice(&u);
        x[n * d as usize..(n + 1) * d as usize].copy_from_slice(&v);
        let out = exec::forward_cpu(
            &prepared,
            &params,
            &BTreeMap::from([(
                "x".to_string(),
                TensorF32::new(vec![1, 1, t_len, d], x).unwrap(),
            )]),
            0,
            "rot",
        )
        .unwrap();
        let a = &out.data[m * d as usize..(m + 1) * d as usize];
        let b = &out.data[n * d as usize..(n + 1) * d as usize];
        a.iter().zip(b).map(|(p, q)| *p as f64 * *q as f64).sum()
    };
    for (m, n, shift) in [(2usize, 5usize, 5usize), (0, 3, 12), (4, 4, 9), (9, 1, 6)] {
        let here = rotated_dot(m, n);
        let there = rotated_dot(m + shift, n + shift);
        assert!(
            (here - there).abs() <= 1e-5,
            "positions ({m}, {n}) and ({}, {}): {here} versus {there}",
            m + shift,
            n + shift
        );
    }
    // And the rotation is not the identity: the same vector changes with its position.
    assert!((rotated_dot(1, 3) - rotated_dot(1, 6)).abs() > 1e-3);
}

#[test]
fn the_template_carries_its_positions() {
    let dims = |positions| TransformerDims {
        layers: 3,
        dim: 32,
        heads: 4,
        seq: 16,
        vocab: 50,
        ffn: 64,
        causal: true,
        positions,
    };
    let none = graph::validate(&transformer(dims(Positions::None)).unwrap()).unwrap();
    let learned = graph::validate(&transformer(dims(Positions::Learned)).unwrap()).unwrap();
    let rope = graph::validate(&transformer(dims(Positions::rotary(10000.0))).unwrap()).unwrap();
    // Learned: one trained vector per position, one addition node.
    assert_eq!(
        learned.trainable_parameters,
        none.trainable_parameters + 16 * 32
    );
    assert_eq!(learned.order.len(), none.order.len() + 1);
    // RoPE: no extra trained parameter; two tables and a −1, frozen; 15 nodes per layer.
    assert_eq!(rope.trainable_parameters, none.trainable_parameters);
    assert_eq!(rope.parameters, none.parameters + 2 * 16 * 8 + 1);
    assert_eq!(rope.order.len(), none.order.len() + 3 * 15);
    assert_eq!(
        dims(Positions::rotary(10000.0)).parameters(),
        rope.parameters
    );
    assert_eq!(
        dims(Positions::Learned).trainable_parameters(),
        learned.trainable_parameters
    );
}

/// The measurement that justifies the encoding: copy the token two positions earlier, on randomly
/// drawn tokens. The answer depends on order, not content.
#[test]
fn a_task_that_depends_on_order_needs_an_encoding() {
    let (seq, vocab, batch, steps) = (8u64, 16u64, 32u64, 300u64);
    let floor = (vocab as f32).ln();
    let mut result = Vec::new();
    for (label, positions) in [
        ("none", Positions::None),
        ("learned", Positions::Learned),
        ("RoPE", Positions::rotary(10000.0)),
    ] {
        let graph = transformer(TransformerDims {
            layers: 1,
            dim: 32,
            heads: 2,
            seq,
            vocab,
            ffn: 64,
            causal: true,
            positions,
        })
        .unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        let mut params = exec::init_parameters(&graph, 7);
        let mut state = NamedOptimizerState::default();
        let mut tail = Vec::new();
        for step in 0..steps {
            let mut d = Drbg::new("test/positions/copy", 5000 + step);
            let ids: Vec<f32> = (0..batch * seq)
                .map(|_| d.next_index(vocab as u32) as f32)
                .collect();
            // Target: the token two positions earlier; 0 for the first two.
            let mut y = vec![0f32; ids.len()];
            for b in 0..batch as usize {
                for t in 2..seq as usize {
                    y[b * seq as usize + t] = ids[b * seq as usize + t - 2];
                }
            }
            let inputs = BTreeMap::from([
                (
                    "ids".to_string(),
                    TensorF32::new(vec![batch, seq], ids).unwrap(),
                ),
                (
                    "y".to_string(),
                    TensorF32::new(vec![batch, seq], y).unwrap(),
                ),
            ]);
            let r = exec::gradient_cpu(&prepared, &params, &inputs, step).unwrap();
            if step >= steps - 40 {
                tail.push(r.loss);
            }
            apply_step_named(
                CoordOptim::Adam,
                0.01,
                &mut params,
                &r.gradients,
                &mut state,
            )
            .unwrap();
        }
        let last = tail.iter().sum::<f32>() / tail.len() as f32;
        eprintln!(
            "{label:>7}: loss over the last 40 steps {last:.3} (no-information floor ln V = {floor:.3})"
        );
        result.push(last);
    }
    let (none, learned, rope) = (result[0], result[1], result[2]);
    assert!(
        learned < 0.1,
        "learned positions: the task should be solved, loss {learned}"
    );
    assert!(
        rope < 0.6,
        "RoPE: the task should be largely solved, loss {rope}"
    );
    assert!(
        none > rope + 0.5,
        "without an encoding the model should stay clearly behind: {none} versus {rope}"
    );
}
