//! D3 (gg-server/docs/ROADMAP.md): a matrix larger than one machine is tiled in the description, and its tiles
//! are spread across stages, hence across machines. What must hold: the output and the gradient
//! (tiles joined back) are those of the original model, and a model in which a single matrix
//! exceeds one machine's memory becomes splittable into stages.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, Prepared};
use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
use gg_compute::graph::tiling::{self, Tiled};
use gg_compute::graph::{self, ModelGraphV1, stages};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

fn lm() -> ModelGraphV1 {
    transformer(TransformerDims {
        layers: 2,
        dim: 32,
        heads: 4,
        seq: 8,
        vocab: 50,
        ffn: 64,
        causal: true,
        positions: Positions::rotary(10000.0),
    })
    .unwrap()
}

fn hf_llama() -> ModelGraphV1 {
    serde_json::from_str(include_str!("fixtures/llama-hf-mini.json")).unwrap()
}

fn random_params(graph: &ModelGraphV1) -> BTreeMap<String, TensorF32> {
    let mut p = exec::init_parameters(graph, 3);
    let mut d = Drbg::new("test/tiling-auto", 1);
    for spec in graph.parameters.iter().filter(|s| s.trainable) {
        for v in &mut p.get_mut(&spec.name).unwrap().data {
            *v = 0.1 * d.next_sym_f32()
                + if spec.name.ends_with("norm.weight") {
                    1.0
                } else {
                    0.0
                };
        }
    }
    p
}

fn batch(graph: &ModelGraphV1, vocab: u32) -> BTreeMap<String, TensorF32> {
    let mut d = Drbg::new("test/tiling-auto/batch", 2);
    graph
        .inputs
        .iter()
        .chain(std::iter::once(graph.loss.target()))
        .map(|s| {
            let shape: Vec<u64> = s
                .shape
                .iter()
                .map(|x| if *x == 0 { 4 } else { *x })
                .collect();
            let n: u64 = shape.iter().product();
            (
                s.name.clone(),
                TensorF32::new(shape, (0..n).map(|_| d.next_index(vocab) as f32).collect())
                    .unwrap(),
            )
        })
        .collect()
}

fn max_diff(a: &TensorF32, b: &TensorF32) -> f32 {
    assert_eq!(a.shape, b.shape);
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn check_same_function(label: &str, graph: &ModelGraphV1, cuts: &[(&str, usize)], vocab: u32) {
    let params = random_params(graph);
    let data = batch(graph, vocab);
    let mut g = graph.clone();
    let mut done: Vec<Tiled> = Vec::new();
    for (param, tiles) in cuts {
        let report = graph::validate(&g).unwrap();
        let (next, t) =
            tiling::tile(&g, &report, param, *tiles).unwrap_or_else(|e| panic!("{label}: {e}"));
        graph::validate(&next).unwrap_or_else(|e| panic!("{label} after {param}: {e}"));
        g = next;
        done.push(t);
    }
    let split = tiling::split_weights(&params, &done).unwrap();
    assert_eq!(
        tiling::merge_weights(&split, &done).unwrap(),
        params,
        "{label}: weight round trip"
    );
    let a = exec::gradient_cpu(&Prepared::new(graph).unwrap(), &params, &data, 0).unwrap();
    let b = exec::gradient_cpu(&Prepared::new(&g).unwrap(), &split, &data, 0).unwrap();
    let merged = tiling::merge_weights(&b.gradients, &done).unwrap();
    let mut worst = 0f32;
    for (name, t) in &a.gradients {
        worst = worst.max(max_diff(&merged[name], t));
    }
    eprintln!(
        "{label}: loss {} vs {}, gradients within {worst:e}",
        b.loss, a.loss
    );
    assert!((a.loss - b.loss).abs() <= 1e-6, "{label}: loss");
    assert!(worst <= 1e-6, "{label}: gradients within {worst:e}");
}

#[test]
fn a_tiled_matrix_computes_the_same_function() {
    // Direct MatMul (language-model head, feed-forward network) and embedding.
    check_same_function(
        "transformer",
        &lm(),
        &[("wlm", 3), ("emb", 4), ("l0.wu", 2)],
        50,
    );
    // Captured Llama: nn.Linear (Transpose then MatMul) and embedding; 11 tiles (joined back
    // through two levels of Concat).
    check_same_function(
        "transformers Llama",
        &hf_llama(),
        &[
            ("lm.lm_head.weight", 11),
            ("lm.model.embed_tokens.weight", 3),
            ("lm.model.layers.0.mlp.down_proj.weight", 2),
        ],
        64,
    );
}

#[test]
fn a_matrix_too_large_for_one_machine_is_spread_across_stages() {
    // A vocabulary of 2,048 at a width of 64: the matrices, not the activations of a micro-batch,
    // fill a machine — the case of a real language model.
    let graph = transformer(TransformerDims {
        layers: 2,
        dim: 64,
        heads: 4,
        seq: 8,
        vocab: 2048,
        ffn: 128,
        causal: true,
        positions: Positions::rotary(10000.0),
    })
    .unwrap();
    let report = graph::validate(&graph).unwrap();
    // The largest matrix, plus its output, exceeds one machine's memory: no stage can carry it,
    // and stage partitioning fails, saying what to do.
    let biggest = graph
        .parameters
        .iter()
        .map(|p| p.shape.iter().product::<u64>() * 4)
        .max()
        .unwrap();
    let cap = biggest;
    let err = stages::partition(&graph, &report, 1, cap).unwrap_err();
    assert!(err.contains("tile its matrix"), "{err}");
    // Tiled into tiles of a quarter of that memory, it spreads out.
    let (tiled, done) = tiling::auto_tile(&graph, cap / 4).unwrap();
    assert!(!done.is_empty());
    let r2 = graph::validate(&tiled).unwrap();
    let bounds = stages::partition(&tiled, &r2, 1, cap).unwrap();
    let st = stages::stages(&tiled, &r2, &bounds).unwrap();
    // The tiles of one matrix sit in different stages.
    let t = &done[0];
    let holders: std::collections::BTreeSet<usize> = (0..t.tiles)
        .map(|i| {
            let name = tiling::tile_name(&t.param, i, t.tiles);
            st.iter().position(|s| s.params.contains(&name)).unwrap()
        })
        .collect();
    eprintln!(
        "'{}' in {} tiles over {} stages; {} stages in total",
        t.param,
        t.tiles,
        holders.len(),
        st.len()
    );
    assert!(holders.len() > 1, "the tiles are all in the same stage");
}

#[test]
fn refusals_are_named() {
    let graph = lm();
    let report = graph::validate(&graph).unwrap();
    for (param, tiles, needle) in [
        ("unknown", 2, "unknown"),
        ("rope.neg", 2, "matrices"),
        ("wlm", 1, "tiles"),
        ("wlm", 10_000, "tiles"),
    ] {
        let err = tiling::tile(&graph, &report, param, tiles).unwrap_err();
        assert!(err.contains(needle), "{param}: {err}");
    }
    // A tied weight (consumed twice) cannot be tiled.
    let mut tied = graph.clone();
    let n = tied
        .nodes
        .iter()
        .position(|n| n.op == "MatMul" && n.inputs[1] == "wlm")
        .unwrap();
    tied.nodes[n].inputs[1] = "l0.wu".into();
    tied.parameters.retain(|p| p.name != "wlm");
    if let Ok(r) = graph::validate(&tied) {
        assert!(
            tiling::tile(&tied, &r, "l0.wu", 2)
                .unwrap_err()
                .contains("tied weight")
        );
    }
}
