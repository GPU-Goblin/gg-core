//! Matrix tiling, expressed in the description itself.
//!
//! The question to settle: can an operator whose weight exceeds the card be written as several
//! operators over tiles of the weight, so that the ADR-0019 levers do the rest? If so, a very
//! large network trains on a single GPU without adding anything to the engine.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, ExecOptions, Prepared};
use gg_compute::graph::{self, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

const D_IN: u64 = 64;
const D_OUT: u64 = 256;
const N: u64 = 8;

fn random(shape: &[u64], seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/tiling", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(
        shape.to_vec(),
        (0..n).map(|_| d.next_sym_f32() * 0.3).collect(),
    )
    .unwrap()
}

/// `y = relu(x·w)` then squared error, with `w` as a single tensor.
fn whole() -> ModelGraphV1 {
    serde_json::from_str(&format!(
        r#"{{
      "format": "gg-graph/v1", "name": "whole",
      "inputs": [{{"name": "x", "shape": [0, {D_IN}]}}],
      "parameters": [{{"name": "w", "shape": [{D_IN}, {D_OUT}], "init": "zeros"}}],
      "nodes": [
        {{"op": "MatMul", "inputs": ["x", "w"], "output": "h"}},
        {{"op": "Relu", "inputs": ["h"], "output": "pred"}}
      ],
      "loss": {{"mean_squared_error": {{"prediction": "pred", "target": {{"name": "y", "shape": [0, {D_OUT}]}}}}}}
    }}"#
    ))
    .unwrap()
}

/// The same computation, with `w` split into `tiles` column tiles joined back by Concat.
fn tiled(tiles: u64) -> ModelGraphV1 {
    let width = D_OUT / tiles;
    let params: Vec<String> = (0..tiles)
        .map(|i| format!(r#"{{"name": "w{i}", "shape": [{D_IN}, {width}], "init": "zeros"}}"#))
        .collect();
    let mut nodes: Vec<String> = (0..tiles)
        .map(|i| format!(r#"{{"op": "MatMul", "inputs": ["x", "w{i}"], "output": "h{i}"}}"#))
        .collect();
    // Concat accepts at most 8 inputs in the profile: beyond that, joins are chained in groups,
    // which changes neither the result nor the number of resident tiles.
    let mut parts: Vec<String> = (0..tiles).map(|i| format!("h{i}")).collect();
    let mut round = 0;
    while parts.len() > 1 {
        let mut next = Vec::new();
        for (g, chunk) in parts.chunks(8).enumerate() {
            if chunk.len() == 1 {
                next.push(chunk[0].clone());
                continue;
            }
            let out = if parts.len() <= 8 {
                "h".to_string()
            } else {
                format!("c{round}_{g}")
            };
            nodes.push(format!(
                r#"{{"op": "Concat", "inputs": [{}], "output": "{out}", "attrs": {{"axis": 1}}}}"#,
                chunk
                    .iter()
                    .map(|n| format!("\"{n}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            next.push(out);
        }
        parts = next;
        round += 1;
    }
    nodes.push(r#"{"op": "Relu", "inputs": ["h"], "output": "pred"}"#.to_string());
    serde_json::from_str(&format!(
        r#"{{
      "format": "gg-graph/v1", "name": "tiled",
      "inputs": [{{"name": "x", "shape": [0, {D_IN}]}}],
      "parameters": [{}],
      "nodes": [{}],
      "loss": {{"mean_squared_error": {{"prediction": "pred", "target": {{"name": "y", "shape": [0, {D_OUT}]}}}}}}
    }}"#,
        params.join(", "),
        nodes.join(", ")
    ))
    .unwrap()
}

fn inputs() -> BTreeMap<String, TensorF32> {
    BTreeMap::from([
        ("x".to_string(), random(&[N, D_IN], 1)),
        ("y".to_string(), random(&[N, D_OUT], 2)),
    ])
}

/// Splits `w` into column tiles, under the names `tiled` expects.
fn split_weight(w: &TensorF32, tiles: u64) -> BTreeMap<String, TensorF32> {
    let width = (D_OUT / tiles) as usize;
    let mut out = BTreeMap::new();
    for i in 0..tiles as usize {
        let mut data = Vec::with_capacity(D_IN as usize * width);
        for row in 0..D_IN as usize {
            let start = row * D_OUT as usize + i * width;
            data.extend_from_slice(&w.data[start..start + width]);
        }
        out.insert(
            format!("w{i}"),
            TensorF32::new(vec![D_IN, width as u64], data).unwrap(),
        );
    }
    out
}

fn run(
    graph: &ModelGraphV1,
    params: &BTreeMap<String, TensorF32>,
    options: ExecOptions,
) -> exec::StepResult {
    let prepared = Prepared::new(graph).expect("validated graph");
    exec::gradient_cpu_with(&prepared, params, &inputs(), 7, options).expect("execution")
}

#[test]
fn a_tiled_matrix_gives_the_same_gradient_for_less_memory() {
    let w = random(&[D_IN, D_OUT], 3);
    let whole_graph = whole();
    let whole_params = BTreeMap::from([("w".to_string(), w.clone())]);
    let reference = run(&whole_graph, &whole_params, ExecOptions::default());
    let total_bytes = D_IN * D_OUT * 4;

    eprintln!(
        "whole weight    : parameter peak {} B (total {total_bytes} B), loss {}",
        reference.peak_parameter_bytes, reference.loss
    );
    assert_eq!(reference.peak_parameter_bytes, total_bytes);

    for tiles in [2u64, 4, 8, 16, 32, 64] {
        let graph = tiled(tiles);
        let report = graph::validate(&graph).expect("tiling keeps the description valid");
        assert_eq!(report.parameters, D_IN * D_OUT, "{tiles} tiles");
        let params = split_weight(&w, tiles);
        // One node per segment: each tile is loaded only for the duration of its own product.
        // With larger segments, two tiles coexist and the gain levels off.
        let split = run(
            &graph,
            &params,
            ExecOptions {
                segments: usize::MAX,
                offload_parameters: true,
            },
        );
        eprintln!(
            "{tiles:>2} tiles        : parameter peak {} B, i.e. 1/{:.1} of the whole weight",
            split.peak_parameter_bytes,
            total_bytes as f64 / split.peak_parameter_bytes as f64
        );
        // Same loss, up to rounding.
        assert!(
            (split.loss - reference.loss).abs() <= 1e-6,
            "{tiles} tiles: loss {} vs {}",
            split.loss,
            reference.loss
        );
        // Same gradient: the joined tiles equal the gradient of the whole weight.
        let width = (D_OUT / tiles) as usize;
        for i in 0..tiles as usize {
            let g = &split.gradients[&format!("w{i}")];
            for row in 0..D_IN as usize {
                for col in 0..width {
                    let mine = g.data[row * width + col];
                    let theirs =
                        reference.gradients["w"].data[row * D_OUT as usize + i * width + col];
                    assert!(
                        (mine - theirs).abs() <= 1e-6,
                        "{tiles} tiles, w{i}[{row},{col}]: {mine} vs {theirs}"
                    );
                }
            }
        }
        // Above all: parameter memory drops to a single tile.
        assert!(
            split.peak_parameter_bytes <= total_bytes / tiles + 64,
            "{tiles} tiles: peak {} B, expected about {} B",
            split.peak_parameter_bytes,
            total_bytes / tiles
        );
    }
}

/// What tiling does not solve: the joined output exists in full at the time of the Concat.
/// The output width, not the weight, therefore becomes the floor.
#[test]
fn the_joined_output_remains_a_floor() {
    let w = random(&[D_IN, D_OUT], 3);
    let tiles = 8u64;
    let output_bytes = N * D_OUT * 4;
    let mut floor = u64::MAX;
    for t in [tiles, tiles * 2, tiles * 4] {
        let split = run(
            &tiled(t),
            &split_weight(&w, t),
            ExecOptions {
                segments: usize::MAX,
                offload_parameters: true,
            },
        );
        eprintln!(
            "{t:>2} tiles: activation peak {} B, parameter peak {} B (joined output alone: {output_bytes} B)",
            split.peak_activation_bytes, split.peak_parameter_bytes
        );
        assert!(
            split.peak_activation_bytes >= output_bytes,
            "joining materializes the whole output: at least {output_bytes} B expected"
        );
        floor = floor.min(split.peak_activation_bytes);
    }
    // Finer tiling lightens the parameters but never goes below the joined output.
    assert!(floor >= output_bytes);
}

/// What tiling costs in time. The sponsor's question was "even if it would be very slow": so we
/// need to know whether the cost comes from tiling itself or from the recomputation that goes
/// with it.
#[test]
#[ignore = "timing measurement, sensitive to machine load"]
fn time_cost_of_tiling() {
    use std::time::Instant;
    let w = random(&[D_IN, D_OUT], 3);
    let whole_params = BTreeMap::from([("w".to_string(), w.clone())]);
    let whole_graph = whole();
    let time = |f: &dyn Fn()| -> f64 {
        f();
        let start = Instant::now();
        for _ in 0..20 {
            f();
        }
        start.elapsed().as_secs_f64() / 20.0
    };

    let base = time(&|| {
        run(&whole_graph, &whole_params, ExecOptions::default());
    });
    let recompute = time(&|| {
        run(
            &whole_graph,
            &whole_params,
            ExecOptions {
                segments: usize::MAX,
                offload_parameters: false,
            },
        );
    });
    eprintln!("whole weight, all in memory: {:.4} ms", base * 1e3);
    eprintln!(
        "whole weight, recomputed by segments: {:.4} ms ({:.2}×)",
        recompute * 1e3,
        recompute / base
    );
    for tiles in [4u64, 16, 64] {
        let graph = tiled(tiles);
        let params = split_weight(&w, tiles);
        let t = time(&|| {
            run(
                &graph,
                &params,
                ExecOptions {
                    segments: usize::MAX,
                    offload_parameters: true,
                },
            );
        });
        eprintln!(
            "{tiles:>2} tiles, recomputation and offloading: {:.4} ms ({:.2}× the whole weight)",
            t * 1e3,
            t / base
        );
    }
}
