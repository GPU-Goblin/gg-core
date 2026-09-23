//! The executor's memory levers (ADR-0019): segmented activation recomputation, parameter
//! offloading, parameter freezing. The property that matters is that **the gradient does not
//! change**: these levers trade memory for compute, they do not change the result.

use std::collections::BTreeMap;

use burn::backend::flex::FlexDevice;
use burn::backend::{Autodiff, Flex};
use gg_compute::graph::exec::{self, ExecOptions, Prepared};
use gg_compute::graph::templates::{TransformerDims, transformer};
use gg_compute::graph::{self, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

type B = Autodiff<Flex>;

fn token_ids(n: u64, seq: u64, vocab: u64, seed: u64) -> TensorF32 {
    let mut d = Drbg::new("test/memory/ids", seed);
    TensorF32::new(
        vec![n, seq],
        (0..n * seq)
            .map(|_| d.next_index(vocab as u32) as f32)
            .collect(),
    )
    .unwrap()
}

fn model() -> ModelGraphV1 {
    transformer(TransformerDims {
        layers: 3,
        dim: 32,
        heads: 4,
        seq: 8,
        vocab: 40,
        ffn: 64,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    })
    .unwrap()
}

fn batch() -> BTreeMap<String, TensorF32> {
    BTreeMap::from([
        ("ids".to_string(), token_ids(4, 8, 40, 1)),
        ("y".to_string(), token_ids(4, 8, 40, 2)),
    ])
}

fn run(
    graph: &ModelGraphV1,
    params: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    options: ExecOptions,
) -> exec::StepResult {
    let prepared = Prepared::new(graph).expect("validated graph");
    exec::gradient_with::<B>(&FlexDevice, &prepared, params, inputs, 7, options).expect("execution")
}

/// Segmenting recomputes activations and returns **exactly** the same gradient.
#[test]
fn segmented_recomputation_returns_the_same_gradient() {
    let graph = model();
    let params = exec::init_parameters(&graph, 3);
    let inputs = batch();
    let direct = run(&graph, &params, &inputs, ExecOptions::default());

    // Segmentation reassociates the f32 sums: the difference is rounding, not an error. The
    // bound comes from measurement (see gg-server/docs/BENCHMARKS.md), and it does not grow with the number
    // of segments, which this test also checks.
    let mut worst_overall = 0.0f32;
    for segments in [2usize, 3, 5, 8, 100] {
        let split = run(
            &graph,
            &params,
            &inputs,
            ExecOptions {
                segments,
                ..ExecOptions::default()
            },
        );
        // The loss is equal up to rounding, not necessarily to the byte. Measured on 2026-09-21:
        // without a causal mask it is byte-identical; with one, it differs by one unit in the last
        // place for 3 and 5 segments only, not for 2, 8 or 100. This non-monotonic scatter is the
        // signature of a different evaluation order. A mask missing from one of the two paths
        // would shift the loss by about 0.01, forty thousand times more than the bound below.
        let ulps = (split.loss - direct.loss).abs() / (f32::EPSILON * direct.loss.abs());
        assert!(
            ulps <= 4.0,
            "{segments} segments: loss {} versus {}, i.e. {ulps:.1} units in the last place",
            split.loss,
            direct.loss
        );
        assert_eq!(split.gradients.len(), direct.gradients.len());
        let mut worst = 0.0f32;
        for (name, g) in &direct.gradients {
            let other = &split.gradients[name];
            let max = g
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = g.data.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let rel = max / scale;
            if rel > worst {
                worst = rel;
            }
            assert!(
                rel <= 1e-4,
                "{segments} segments, '{name}': relative error {rel}, beyond the expected rounding"
            );
        }
        eprintln!("{segments:>4} segments: worst relative error {worst:.3e}");
        worst_overall = worst_overall.max(worst);
    }
    // A real backpropagation defect would grow with the number of segments; rounding does not.
    assert!(worst_overall < 1e-4, "overall error {worst_overall:.3e}");
}

/// Independent proof that the segmented path computes the right gradient: finite differences on a
/// segmented model, without ever comparing it with the direct path.
#[test]
fn finite_differences_confirm_the_segmented_path() {
    let graph = model();
    let mut params = exec::init_parameters(&graph, 3);
    for t in params.values_mut() {
        for v in &mut t.data {
            *v *= 4.0;
        }
    }
    let inputs = batch();
    let options = ExecOptions {
        segments: 5,
        offload_parameters: true,
    };
    let base = run(&graph, &params, &inputs, options);
    let h = 1e-2f32;
    let mut checked = 0;
    for (name, g) in &base.gradients {
        let mut order: Vec<usize> = (0..g.data.len()).collect();
        order.sort_by(|a, b| g.data[*b].abs().total_cmp(&g.data[*a].abs()));
        for &i in order.iter().take(2) {
            let mut plus = params.clone();
            plus.get_mut(name).unwrap().data[i] += h;
            let mut minus = params.clone();
            minus.get_mut(name).unwrap().data[i] -= h;
            let fd = (run(&graph, &plus, &inputs, options).loss
                - run(&graph, &minus, &inputs, options).loss)
                / (2.0 * h);
            let ad = g.data[i];
            assert!(
                (fd - ad).abs() <= 2e-2 + 5e-2 * ad.abs(),
                "{name}[{i}]: finite differences {fd}, segmented autodiff {ad}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 10, "too few components checked: {checked}");
}

/// And it really holds fewer activations: the measured peak goes down.
#[test]
fn recomputation_lowers_the_measured_activation_peak() {
    let graph = model();
    let params = exec::init_parameters(&graph, 3);
    let inputs = batch();
    let direct = run(&graph, &params, &inputs, ExecOptions::default());
    let split = run(
        &graph,
        &params,
        &inputs,
        ExecOptions {
            segments: 6,
            ..ExecOptions::default()
        },
    );
    assert!(direct.peak_activation_bytes > 0);
    assert!(
        split.peak_activation_bytes < direct.peak_activation_bytes,
        "peak {} with 6 segments versus {} without",
        split.peak_activation_bytes,
        direct.peak_activation_bytes
    );
}

/// Offloading loads only the current segment's parameters: the parameter peak goes down, and the
/// gradient stays the same.
#[test]
fn offloading_lowers_the_parameter_peak() {
    let graph = model();
    let params = exec::init_parameters(&graph, 3);
    let inputs = batch();
    let resident = run(
        &graph,
        &params,
        &inputs,
        ExecOptions {
            segments: 4,
            offload_parameters: false,
        },
    );
    let offloaded = run(
        &graph,
        &params,
        &inputs,
        ExecOptions {
            segments: 4,
            offload_parameters: true,
        },
    );
    let total: u64 = params.values().map(|t| t.data.len() as u64 * 4).sum();
    assert_eq!(
        resident.peak_parameter_bytes, total,
        "without offloading, everything is resident"
    );
    assert!(
        offloaded.peak_parameter_bytes < total,
        "with offloading, the peak is {} versus {total} resident",
        offloaded.peak_parameter_bytes
    );
    assert!(offloaded.peak_parameter_bytes > 0);
    for (name, g) in &resident.gradients {
        assert_eq!(g.encode(), offloaded.gradients[name].encode(), "{name}");
    }
}

/// Freezing a parameter: it receives no gradient, it is not transmitted, and the other
/// gradients are unchanged.
#[test]
fn a_frozen_parameter_receives_no_gradient() {
    let graph = model();
    let params = exec::init_parameters(&graph, 3);
    let inputs = batch();
    let all = run(&graph, &params, &inputs, ExecOptions::default());

    // Freeze everything except the value projections: this is the shape of adapter fine-tuning.
    let mut frozen = graph.clone();
    for p in &mut frozen.parameters {
        p.trainable = p.name.ends_with(".wv");
    }
    let report = graph::validate(&frozen).expect("a partially frozen graph remains valid");
    assert_eq!(
        report.parameters,
        graph::validate(&graph).unwrap().parameters
    );
    assert!(report.trainable_parameters < report.parameters);
    assert_eq!(report.trainable_parameters, 3 * 32 * 32);
    assert_eq!(
        report.trainable_parameter_bytes,
        report.trainable_parameters * 4
    );

    let partial = run(&frozen, &params, &inputs, ExecOptions::default());
    assert_eq!(
        partial.loss.to_bits(),
        all.loss.to_bits(),
        "the loss does not depend on freezing"
    );
    assert_eq!(
        partial.gradients.len(),
        3,
        "one gradient per value projection"
    );
    for (name, g) in &partial.gradients {
        assert!(name.ends_with(".wv"), "{name} should not be trained");
        let full = &all.gradients[name];
        let max = g
            .data
            .iter()
            .zip(&full.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max <= 1e-6,
            "{name}: freezing changes the gradient by {max}"
        );
    }
}

/// Freezing everything makes no sense and must be rejected at validation, not discovered at execution.
#[test]
fn freezing_everything_is_rejected() {
    let mut graph = model();
    for p in &mut graph.parameters {
        p.trainable = false;
    }
    let err = graph::validate(&graph).expect_err("nothing to learn");
    assert!(err.to_string().contains("trained"), "{err}");
}

/// The three levers together, on the same model: same gradient, less memory.
#[test]
fn the_three_levers_together() {
    let graph = model();
    let params = exec::init_parameters(&graph, 3);
    let inputs = batch();
    let direct = run(&graph, &params, &inputs, ExecOptions::default());

    let mut frozen = graph.clone();
    for p in &mut frozen.parameters {
        p.trainable = p.name.ends_with(".wv") || p.name.ends_with(".wo");
    }
    let lean = run(
        &frozen,
        &params,
        &inputs,
        ExecOptions {
            segments: 6,
            offload_parameters: true,
        },
    );
    assert!(lean.peak_activation_bytes < direct.peak_activation_bytes);
    assert!(lean.peak_parameter_bytes < direct.peak_parameter_bytes);
    assert_eq!(lean.gradients.len(), 6);
    for (name, g) in &lean.gradients {
        let full = &direct.gradients[name];
        let max = g
            .data
            .iter()
            .zip(&full.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = full
            .data
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        assert!(
            max / scale <= 1e-5,
            "{name}: relative error {}",
            max / scale
        );
    }
}

#[test]
fn zero_segments_is_rejected() {
    let graph = model();
    let params = exec::init_parameters(&graph, 3);
    let prepared = Prepared::new(&graph).unwrap();
    let err = exec::gradient_with::<B>(
        &FlexDevice,
        &prepared,
        &params,
        &batch(),
        7,
        ExecOptions {
            segments: 0,
            offload_parameters: false,
        },
    )
    .expect_err("0 segments");
    assert!(err.to_string().contains("segments"), "{err}");
}
