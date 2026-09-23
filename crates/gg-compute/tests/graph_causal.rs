//! `CausalMask` (profile `gg-ops/2`, ADR-0020). What these tests establish, in order:
//!
//! 1. the operator does exactly what its definition says (naive oracle, byte for byte);
//! 2. **no information flows back from the future**: changing a token changes nothing at the
//!    positions before it, byte for byte, in direct as well as segmented execution; the same test
//!    without a mask sees the leak, which proves it is able to detect one;
//! 3. malformed descriptions are rejected with the cause named;
//! 4. in training, the mask prevents a model from "cheating": on tokens drawn at random, hence
//!    unpredictable, a model without a mask still learns to "predict" by reading the future, while
//!    a masked model cannot.

use std::collections::BTreeMap;

use gg_compute::grad::CoordOptim;
use gg_compute::graph::exec::{self, ExecOptions, Prepared};
use gg_compute::graph::task::{NamedOptimizerState, apply_step_named};
use gg_compute::graph::templates::{TransformerDims, transformer};
use gg_compute::graph::{self, CAUSAL_FILL, GraphError, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

fn random(shape: &[u64], seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/causal", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(
        shape.to_vec(),
        (0..n).map(|_| d.next_sym_f32() * 3.0).collect(),
    )
    .unwrap()
}

fn tokens(n: u64, seq: u64, vocab: u64, seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/causal/tokens", seed);
    TensorF32::new(
        vec![n, seq],
        (0..n * seq)
            .map(|_| d.next_index(vocab as u32) as f32)
            .collect(),
    )
    .unwrap()
}

/// A graph reduced to `CausalMask` on an input of the given shape.
fn mask_only(shape: &[u64]) -> ModelGraphV1 {
    let dims: Vec<String> = shape
        .iter()
        .enumerate()
        .map(|(i, d)| if i == 0 { "0".into() } else { d.to_string() })
        .collect();
    serde_json::from_str(&format!(
        r#"{{"format":"gg-graph/v1","name":"mask",
        "inputs":[{{"name":"x","shape":[{d}]}}],
        "parameters":[{{"name":"w","shape":[1],"init":"ones"}}],
        "nodes":[{{"op":"Mul","inputs":["x","w"],"output":"s"}},
                 {{"op":"CausalMask","inputs":["s"],"output":"out"}}],
        "loss":{{"mean_squared_error":{{"prediction":"out","target":{{"name":"y","shape":[{d}]}}}}}}}}"#,
        d = dims.join(",")
    ))
    .unwrap()
}

#[test]
fn the_operator_follows_its_definition_byte_for_byte() {
    // Rank 3 and 4: the first axis is the batch, the last two the score matrix. A rank of 2 would
    // put the batch inside the matrix, which validation rejects (see below).
    for shape in [vec![3u64, 5, 5], vec![3, 6, 6], vec![2, 3, 4, 4]] {
        let graph = mask_only(&shape);
        let prepared = Prepared::new(&graph).expect("valid");
        let x = random(&shape, 1);
        let params =
            BTreeMap::from([("w".to_string(), TensorF32::new(vec![1], vec![1.0]).unwrap())]);
        let got = exec::forward_cpu(
            &prepared,
            &params,
            &BTreeMap::from([("x".to_string(), x.clone())]),
            0,
            "out",
        )
        .expect("execution");
        assert_eq!(got.shape, shape);
        // Naive oracle: on the last two axes, (i, j) with j > i receives the mask value.
        let t = *shape.last().unwrap() as usize;
        for (k, (a, b)) in got.data.iter().zip(&x.data).enumerate() {
            let (i, j) = ((k / t) % t, k % t);
            let expected = if j > i { CAUSAL_FILL } else { *b };
            assert_eq!(
                a.to_bits(),
                expected.to_bits(),
                "shape {shape:?}, element {k} (i={i}, j={j})"
            );
        }
    }
}

fn causal_lm(causal: bool) -> ModelGraphV1 {
    transformer(TransformerDims {
        layers: 2,
        dim: 16,
        heads: 2,
        seq: 6,
        vocab: 11,
        ffn: 32,
        causal,
        positions: gg_compute::graph::templates::Positions::None,
    })
    .unwrap()
}

/// Positions `[0, t)` of the logits of the batch's first sequence: what a position "knows".
fn prefix_rows(logits: &TensorF32, seq: usize, vocab: usize, upto: usize) -> Vec<u32> {
    logits.data[..upto * vocab]
        .iter()
        .map(|v| v.to_bits())
        .chain(std::iter::once(seq as u32))
        .collect()
}

#[test]
fn no_information_flows_back_from_the_future() {
    let (seq, vocab) = (6usize, 11usize);
    for causal in [true, false] {
        let graph = causal_lm(causal);
        let prepared = Prepared::new(&graph).unwrap();
        let params = exec::init_parameters(&graph, 5);
        let ids = tokens(3, seq as u64, vocab as u64, 1);
        let run = |ids: &TensorF32| {
            exec::forward_cpu(
                &prepared,
                &params,
                &BTreeMap::from([("ids".to_string(), ids.clone())]),
                0,
                "seqlogits",
            )
            .unwrap()
        };
        let base = run(&ids);
        assert_eq!(base.shape, vec![3, seq as u64, vocab as u64]);
        let mut leaked = false;
        for j in 1..seq {
            // Change the token at position j of the first sequence.
            let mut changed = ids.clone();
            changed.data[j] = ((changed.data[j] as usize + 1) % vocab) as f32;
            let after = run(&changed);
            let same_prefix =
                prefix_rows(&base, seq, vocab, j) == prefix_rows(&after, seq, vocab, j);
            // Position j itself, however, must see the change.
            let at_j = |t: &TensorF32| t.data[j * vocab..(j + 1) * vocab].to_vec();
            assert_ne!(
                at_j(&base),
                at_j(&after),
                "position {j} ignores its own token"
            );
            // The other sequences of the batch are never affected.
            let other = |t: &TensorF32| {
                t.data[seq * vocab..]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                other(&base),
                other(&after),
                "leak between sequences of the batch"
            );
            if causal {
                assert!(
                    same_prefix,
                    "causal mask: changing token {j} modifies an earlier position"
                );
            } else if !same_prefix {
                leaked = true;
            }
        }
        if !causal {
            assert!(
                leaked,
                "without a mask, the test must see the leak: otherwise it would prove nothing"
            );
        }
    }
}

/// The same guarantee in segmented execution with offloading: segment-wise recomputation rebuilds
/// each segment, mask included. The gradient of a position depends only on the past.
#[test]
fn the_mask_survives_segmented_execution() {
    let graph = causal_lm(true);
    let prepared = Prepared::new(&graph).unwrap();
    let params = exec::init_parameters(&graph, 5);
    let batch = BTreeMap::from([
        ("ids".to_string(), tokens(3, 6, 11, 1)),
        ("y".to_string(), tokens(3, 6, 11, 2)),
    ]);
    let direct = exec::gradient_cpu(&prepared, &params, &batch, 0).unwrap();
    for segments in [2usize, 5, 17, usize::MAX] {
        let split = exec::gradient_cpu_with(
            &prepared,
            &params,
            &batch,
            0,
            ExecOptions {
                segments,
                offload_parameters: true,
            },
        )
        .unwrap();
        let ulps = (split.loss - direct.loss).abs() / (f32::EPSILON * direct.loss.abs());
        assert!(ulps <= 4.0, "{segments} segments: {ulps} ulp");
        for (name, g) in &direct.gradients {
            let other = &split.gradients[name];
            let max = g
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = g.data.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            assert!(
                max / scale <= 1e-4,
                "{segments} segments, {name}: {}",
                max / scale
            );
        }
    }
}

#[test]
fn malformed_descriptions_are_rejected() {
    let reject = |json: &str, needle: &str| {
        let graph: ModelGraphV1 = serde_json::from_str(json).unwrap();
        match graph::validate(&graph) {
            Err(GraphError::Shape(m)) => {
                assert!(m.contains(needle), "\"{m}\" does not mention {needle}")
            }
            other => panic!("rejection expected, got {other:?}"),
        }
    };
    let base = |shape: &str, target: &str| {
        format!(
            r#"{{"format":"gg-graph/v1","name":"m","inputs":[{{"name":"x","shape":{shape}}}],
            "parameters":[{{"name":"w","shape":[1],"init":"ones"}}],
            "nodes":[{{"op":"Mul","inputs":["x","w"],"output":"s"}},{{"op":"CausalMask","inputs":["s"],"output":"out"}}],
            "loss":{{"mean_squared_error":{{"prediction":"out","target":{{"name":"y","shape":{target}}}}}}}}}"#
        )
    };
    // A vector has no score matrix.
    reject(&base("[0]", "[0]"), "[.., T, T]");
    // Queries and keys of different lengths.
    reject(&base("[0, 4, 5]", "[0, 4, 5]"), "square");
    // The batch axis cannot carry the sequence.
    reject(&base("[0, 4]", "[0, 4]"), "batch");
}

/// Declared cost: one comparison per element, and no parameter.
#[test]
fn the_mask_costs_no_parameter_and_no_significant_compute() {
    let causal = graph::validate(&causal_lm(true)).unwrap();
    let encoder = graph::validate(&causal_lm(false)).unwrap();
    assert_eq!(causal.parameters, encoder.parameters);
    assert!(causal.ops_used.contains(&"CausalMask".to_string()));
    let extra = causal.forward_flops_per_example - encoder.forward_flops_per_example;
    // Two layers, 2 heads, 6×6 scores: one operation per score element.
    assert_eq!(extra, 2 * 2 * 6 * 6);
}

/// Why the mask exists. Tokens are drawn at random at every step: the next one is
/// **unpredictable**, and the best possible loss is ln(V). A model without a mask nevertheless goes
/// below this floor, because it sees all the tokens of the sequence, the future included. A masked
/// model stays at the floor: it has no way to cheat.
#[test]
fn without_a_mask_the_model_cheats_with_a_mask_it_cannot() {
    let (seq, vocab, batch, steps) = (8u64, 32u64, 32u64, 250u64);
    let floor = (vocab as f32).ln();
    let mut finals = Vec::new();
    for causal in [false, true] {
        let graph = transformer(TransformerDims {
            layers: 1,
            dim: 32,
            heads: 2,
            seq,
            vocab,
            ffn: 64,
            causal,
            positions: gg_compute::graph::templates::Positions::None,
        })
        .unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        let mut params = exec::init_parameters(&graph, 7);
        let mut state = NamedOptimizerState::default();
        let mut tail = Vec::new();
        for step in 0..steps {
            // seq + 1 fresh tokens; the input is [0, seq), the target is [1, seq]: the next token.
            let raw = tokens(batch, seq + 1, vocab, 1000 + step);
            let row = (seq + 1) as usize;
            let mut ids = Vec::with_capacity((batch * seq) as usize);
            let mut next = Vec::with_capacity((batch * seq) as usize);
            for b in 0..batch as usize {
                ids.extend_from_slice(&raw.data[b * row..b * row + seq as usize]);
                next.extend_from_slice(&raw.data[b * row + 1..(b + 1) * row]);
            }
            let inputs = BTreeMap::from([
                (
                    "ids".to_string(),
                    TensorF32::new(vec![batch, seq], ids).unwrap(),
                ),
                (
                    "y".to_string(),
                    TensorF32::new(vec![batch, seq], next).unwrap(),
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
            "{}: mean loss over the last 40 steps {last:.3} (floor ln V = {floor:.3})",
            if causal { "with mask" } else { "without mask" }
        );
        finals.push(last);
    }
    let (cheater, honest) = (finals[0], finals[1]);
    assert!(
        cheater < floor - 0.5,
        "without a mask, the model should go well below ln V by reading the future: {cheater}"
    );
    assert!(
        honest > floor - 0.1,
        "with a mask, the model cannot do better than ln V on random tokens: {honest}"
    );
}
