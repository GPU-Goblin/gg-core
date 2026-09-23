//! A task's memory estimate (step C1) is an upper bound on what the executor measures, without
//! being absurdly loose, and it responds to the levers the way the executor does. This is what lets
//! the server offer a task only to a worker that can hold it.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, ExecOptions, Prepared};
use gg_compute::graph::memory::{profile_level, required_profile, task_memory};
use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
use gg_compute::graph::{self, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

fn example(name: &str) -> ModelGraphV1 {
    let path = format!(
        "{}/../../examples/models/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn lm(positions: Positions, layers: u64) -> ModelGraphV1 {
    transformer(TransformerDims {
        layers,
        dim: 32,
        heads: 4,
        seq: 8,
        vocab: 40,
        ffn: 64,
        causal: true,
        positions,
    })
    .unwrap()
}

/// A random micro-batch valid for the description: bounded indices for the i64 inputs and the
/// cross-entropy target.
fn inputs(graph: &ModelGraphV1, batch: u64, classes: u32) -> BTreeMap<String, TensorF32> {
    let mut d = Drbg::new("test/memory-estimate", batch);
    let mut specs: Vec<_> = graph.inputs.iter().collect();
    specs.push(graph.loss.target());
    specs
        .into_iter()
        .map(|s| {
            let shape: Vec<u64> = s
                .shape
                .iter()
                .map(|x| if *x == 0 { batch } else { *x })
                .collect();
            let n: u64 = shape.iter().product();
            let data = (0..n)
                .map(|_| {
                    if s.dtype == "i64" {
                        d.next_index(classes) as f32
                    } else {
                        d.next_sym_f32()
                    }
                })
                .collect();
            (s.name.clone(), TensorF32::new(shape, data).unwrap())
        })
        .collect()
}

#[test]
fn the_estimate_bounds_the_executor_measurement() {
    let cases: Vec<(&str, ModelGraphV1, u32)> = vec![
        ("transformer without positions", lm(Positions::None, 3), 40),
        ("RoPE transformer", lm(Positions::rotary(10000.0), 3), 40),
        (
            "learned-position transformer",
            lm(Positions::Learned, 2),
            40,
        ),
        ("perceptron", example("mlp-tabular"), 10),
        ("time series", example("series-horizon-1"), 1),
    ];
    let mut worst_ratio = 0f64;
    for (label, graph, classes) in &cases {
        let prepared = Prepared::new(graph).unwrap();
        let report = graph::validate(graph).unwrap();
        let params = exec::init_parameters(graph, 3);
        for batch in [1u64, 4] {
            let data = inputs(graph, batch, *classes);
            for segments in [1usize, 2, 4, 8] {
                for offload in [false, true] {
                    let options = ExecOptions {
                        segments,
                        offload_parameters: offload,
                    };
                    let step = exec::gradient_cpu_with(&prepared, &params, &data, 1, options)
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    let measured = step.peak_activation_bytes + step.peak_parameter_bytes;
                    let estimate = task_memory(graph, &report, batch, options);
                    assert!(
                        estimate.total() >= measured,
                        "{label}, batch {batch}, {segments} segments, offload {offload}: estimate \
{estimate:?} = {} < measured {measured}",
                        estimate.total()
                    );
                    worst_ratio = worst_ratio.max(estimate.total() as f64 / measured as f64);
                }
            }
        }
    }
    eprintln!("estimate / measurement ratio, worst case: {worst_ratio:.2}");
    assert!(worst_ratio <= 4.0, "estimate too loose: {worst_ratio:.2}×");
}

#[test]
fn the_levers_lower_the_estimate() {
    let graph = lm(Positions::rotary(10000.0), 6);
    let report = graph::validate(&graph).unwrap();
    let est = |segments, offload_parameters| {
        task_memory(
            &graph,
            &report,
            16,
            ExecOptions {
                segments,
                offload_parameters,
            },
        )
    };
    let plain = est(1, false);
    let segmented = est(6, false);
    let offloaded = est(6, true);
    assert!(
        segmented.activations < plain.activations,
        "{segmented:?} {plain:?}"
    );
    assert!(
        offloaded.parameters < segmented.parameters,
        "{offloaded:?} {segmented:?}"
    );
    assert_eq!(plain.gradients, report.trainable_parameter_bytes);
    // The batch multiplies activations and data, not parameters.
    let big = task_memory(&graph, &report, 64, ExecOptions::default());
    assert_eq!(big.parameters, plain.parameters);
    assert!(big.activations > 3 * plain.activations);
}

#[test]
fn the_required_profile_version_follows_what_the_description_uses() {
    assert_eq!(profile_level("gg-ops/4"), Some(4));
    for bad in ["gg-ops/", "gg-ops/x", "gg-ops/0", "onnx/17", ""] {
        assert_eq!(profile_level(bad), None, "{bad}");
    }
    assert_eq!(required_profile(&example("mlp-tabular")), 1);
    assert_eq!(required_profile(&example("cnn-cifar")), 1);
    // Causal mask: gg-ops/2.
    assert_eq!(required_profile(&lm(Positions::None, 1)), 2);
    // Learned positions: no Slice, but the mask is there.
    assert_eq!(required_profile(&lm(Positions::Learned, 1)), 2);
    // RoPE from a base: generated tables and Slice, gg-ops/3.
    assert_eq!(required_profile(&lm(Positions::rotary(10000.0), 1)), 3);
    // Scaled RoPE: explicit frequencies, gg-ops/4.
    let scaled = lm(
        Positions::Rotary {
            base: 10000.0,
            scaling: gg_compute::graph::rope::RopeScaling::Linear { factor: 2.0 },
        },
        1,
    );
    assert_eq!(required_profile(&scaled), 4);
    assert!(required_profile(&scaled) <= gg_compute::graph::memory::own_profile_level());
    // A real captured Llama (RMSNorm written by hand): gg-ops/5.
    let hf: ModelGraphV1 =
        serde_json::from_str(include_str!("fixtures/llama-hf-mini.json")).unwrap();
    assert_eq!(required_profile(&hf), 5);
}

#[test]
fn the_recommendation_picks_the_cheapest_lever_that_fits() {
    use gg_compute::graph::memory::recommend;
    let graph = lm(Positions::rotary(10000.0), 6);
    let report = graph::validate(&graph).unwrap();
    let plain = task_memory(&graph, &report, 16, ExecOptions::default());
    // Enough memory: no lever.
    let (o, m) = recommend(&graph, &report, 16, plain.total()).unwrap();
    assert_eq!(o, ExecOptions::default());
    assert_eq!(m, plain);
    // A little less: segments, without offloading.
    let (o, m) = recommend(&graph, &report, 16, plain.total() - 1).unwrap();
    assert!(o.segments > 1 && !o.offload_parameters, "{o:?}");
    assert!(m.total() < plain.total());
    // All options: below the minimum without offloading, only offloading can fit.
    let n = report.order.len();
    let all = |offload_parameters: bool| {
        (1..=n)
            .map(|segments| {
                task_memory(
                    &graph,
                    &report,
                    16,
                    ExecOptions {
                        segments,
                        offload_parameters,
                    },
                )
                .total()
            })
            .min()
            .unwrap()
    };
    let (best_plain, best_offload) = (all(false), all(true));
    assert!(best_offload < best_plain, "{best_offload} {best_plain}");
    let (o, m) = recommend(&graph, &report, 16, best_plain - 1).unwrap();
    assert!(o.offload_parameters, "{o:?}");
    assert!(m.total() < best_plain);
    // Below the absolute minimum, nothing fits: None.
    assert!(recommend(&graph, &report, 16, best_offload.min(best_plain) - 1).is_none());
}
