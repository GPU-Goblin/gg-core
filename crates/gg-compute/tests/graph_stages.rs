//! Stages (gg-server/docs/ROADMAP.md, P11 D2). The property that matters: **chaining the stages, each one alone
//! with only its own parameters, gives back exactly the gradient** of a single-machine run split at
//! the same bounds. This is what lets a model be spread across machines without changing what it
//! learns.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, ExecOptions, Prepared};
use gg_compute::graph::stages::{self, StageSpec};
use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
use gg_compute::graph::{self, ModelGraphV1};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

type B = burn::backend::Autodiff<burn::backend::Flex>;
const DEV: burn::backend::flex::FlexDevice = burn::backend::flex::FlexDevice;

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

fn hf_llama() -> ModelGraphV1 {
    serde_json::from_str(include_str!("fixtures/llama-hf-mini.json")).unwrap()
}

/// Non-zero weights for every trained parameter (the driver declares those of a captured module as
/// zero: they travel in the checkpoint).
fn params(graph: &ModelGraphV1) -> BTreeMap<String, TensorF32> {
    let mut p = exec::init_parameters(graph, 3);
    let mut d = Drbg::new("test/stages/params", 1);
    for spec in graph.parameters.iter().filter(|s| s.trainable) {
        let t = p.get_mut(&spec.name).unwrap();
        if t.data.iter().all(|v| *v == 0.0) {
            for v in &mut t.data {
                *v = 0.1 * d.next_sym_f32()
                    + if spec.name.ends_with("norm.weight") {
                        1.0
                    } else {
                        0.0
                    };
            }
        }
    }
    p
}

fn batch(graph: &ModelGraphV1, n: u64, vocab: u32) -> BTreeMap<String, TensorF32> {
    let mut d = Drbg::new("test/stages/batch", n);
    graph
        .inputs
        .iter()
        .chain(std::iter::once(graph.loss.target()))
        .map(|s| {
            let shape: Vec<u64> = s
                .shape
                .iter()
                .map(|x| if *x == 0 { n } else { *x })
                .collect();
            let count: u64 = shape.iter().product();
            let data = (0..count)
                .map(|_| {
                    if s.dtype == "i64" {
                        d.next_index(vocab) as f32
                    } else {
                        d.next_sym_f32()
                    }
                })
                .collect();
            (s.name.clone(), TensorF32::new(shape, data).unwrap())
        })
        .collect()
}

/// The grid's path, in a single process: forward passes in order, backward passes in reverse, each
/// stage receiving only its own parameters and inputs.
fn run_chain(
    prepared: &Prepared,
    stages: &[StageSpec],
    params: &BTreeMap<String, TensorF32>,
    data: &BTreeMap<String, TensorF32>,
) -> (f32, BTreeMap<String, TensorF32>) {
    let mut produced: BTreeMap<String, TensorF32> = data.clone();
    let mut kept_inputs = Vec::new();
    for s in stages {
        let only_params: BTreeMap<String, TensorF32> = s
            .params
            .iter()
            .map(|n| (n.clone(), params[n].clone()))
            .collect();
        let inputs: BTreeMap<String, TensorF32> = s
            .inputs
            .iter()
            .map(|n| (n.clone(), produced[n].clone()))
            .collect();
        let (outputs, _) =
            exec::stage_forward::<burn::backend::Flex>(&DEV, prepared, s, &only_params, &inputs, 0)
                .unwrap();
        assert_eq!(outputs.keys().collect::<Vec<_>>(), {
            let mut o: Vec<&String> = s.outputs.iter().collect();
            o.sort();
            o
        });
        produced.extend(outputs);
        kept_inputs.push((only_params, inputs));
    }
    // Gradients received per tensor, in decreasing stage order.
    let mut received: BTreeMap<String, Vec<TensorF32>> = BTreeMap::new();
    let mut total: BTreeMap<String, TensorF32> = BTreeMap::new();
    let mut loss = f32::NAN;
    for (s, (only_params, inputs)) in stages.iter().zip(&kept_inputs).rev() {
        let r = exec::stage_backward::<B>(&DEV, prepared, s, only_params, inputs, &received, 0)
            .unwrap();
        if let Some(l) = r.loss {
            loss = l;
        }
        for (name, g) in r.input_gradients {
            received.entry(name).or_default().push(g);
        }
        for (name, g) in r.gradients {
            match total.get_mut(&name) {
                Some(acc) => {
                    for (a, b) in acc.data.iter_mut().zip(&g.data) {
                        *a += *b;
                    }
                }
                None => {
                    total.insert(name, g);
                }
            }
        }
    }
    (loss, total)
}

#[test]
fn chaining_the_stages_gives_back_exactly_the_gradient() {
    let cases: Vec<(&str, ModelGraphV1, u32)> = vec![
        ("RoPE transformer", lm(Positions::rotary(10000.0), 3), 40),
        (
            "learned-position transformer",
            lm(Positions::Learned, 2),
            40,
        ),
        ("transformers Llama", hf_llama(), 64),
    ];
    for (label, graph, vocab) in cases {
        let prepared = Prepared::new(&graph).unwrap();
        let report = graph::validate(&graph).unwrap();
        let p = params(&graph);
        let data = batch(&graph, 4, vocab);
        let n = report.order.len();
        for k in [2usize, 3, 5] {
            let bounds = stages::even_bounds(n, k);
            let st = stages::stages(&graph, &report, &bounds).unwrap();
            // Every parameter is held by at least one stage, and a stage holds only its own.
            let held: usize = st.iter().map(|s| s.params.len()).sum();
            assert!(held >= graph.parameters.len(), "{label}");
            let (loss, grads) = run_chain(&prepared, &st, &p, &data);
            let reference = exec::gradient_cpu_with(
                &prepared,
                &p,
                &data,
                0,
                ExecOptions {
                    segments: k,
                    offload_parameters: false,
                },
            )
            .unwrap();
            assert_eq!(loss, reference.loss, "{label}, {k} stages: loss");
            let mut worst = 0f32;
            for (name, r) in &reference.gradients {
                let g = &grads[name];
                let w = g
                    .data
                    .iter()
                    .zip(&r.data)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                if w > 0.0 {
                    eprintln!("  {name}: {w:e}");
                }
                worst = worst.max(w);
            }
            eprintln!("{label}, {k} stages: gradient gap {worst:e}");
            assert_eq!(worst, 0.0, "{label}, {k} stages: the gradients differ");
        }
    }
}

#[test]
fn stages_describe_themselves_and_refusals_say_why() {
    let graph = lm(Positions::rotary(10000.0), 2);
    let report = graph::validate(&graph).unwrap();
    let n = report.order.len();
    let st = stages::stages(&graph, &report, &stages::even_bounds(n, 3)).unwrap();
    assert_eq!(st.len(), 3);
    assert!(st[0].inputs.contains(&"ids".to_string()));
    assert!(st[2].last && st[2].inputs.last().unwrap() == "y");
    // An output of a stage is an input of a later stage.
    for (i, s) in st.iter().enumerate() {
        for o in &s.outputs {
            assert!(
                st[i + 1..].iter().any(|t| t.inputs.contains(o)),
                "output {o} of stage {i} is read by nobody"
            );
        }
    }
    for bad in [
        vec![0, n as u32],
        vec![1, n as u32],
        vec![0, 5, 5, n as u32],
        vec![0, n as u32 + 1],
    ] {
        let r = stages::stages(&graph, &report, &bad);
        if bad == vec![0, n as u32] {
            assert!(r.is_ok());
        } else {
            assert!(r.is_err(), "{bad:?}");
        }
    }
}

#[test]
fn greedy_partition_fits_the_given_memory_and_measurement_confirms_it() {
    let graph = hf_llama();
    let report = graph::validate(&graph).unwrap();
    let prepared = Prepared::new(&graph).unwrap();
    let p = params(&graph);
    let data = batch(&graph, 4, 64);
    let whole = stages::stages(&graph, &report, &[0, report.order.len() as u32]).unwrap();
    let alone = stages::stage_memory(&graph, &report, &whole[0], 4);
    for divisor in [2u64, 3, 5] {
        let cap = alone / divisor;
        let bounds = stages::partition(&graph, &report, 4, cap).unwrap();
        let st = stages::stages(&graph, &report, &bounds).unwrap();
        assert!(
            st.len() >= divisor as usize,
            "{divisor}: {} stages",
            st.len()
        );
        // Every stage fits, and the measurement stays below the estimate.
        let mut produced = data.clone();
        for s in &st {
            let est = stages::stage_memory(&graph, &report, s, 4);
            assert!(est <= cap);
            let only: BTreeMap<String, TensorF32> =
                s.params.iter().map(|n| (n.clone(), p[n].clone())).collect();
            let inputs: BTreeMap<String, TensorF32> = s
                .inputs
                .iter()
                .map(|n| (n.clone(), produced[n].clone()))
                .collect();
            let (outputs, fwd_peak) =
                exec::stage_forward::<burn::backend::Flex>(&DEV, &prepared, s, &only, &inputs, 0)
                    .unwrap();
            let fake: BTreeMap<String, Vec<TensorF32>> = outputs
                .iter()
                .map(|(k, v)| (k.clone(), vec![v.clone()]))
                .collect();
            let back =
                exec::stage_backward::<B>(&DEV, &prepared, s, &only, &inputs, &fake, 0).unwrap();
            assert!(
                fwd_peak <= est && back.peak_bytes <= est,
                "stage {}: estimate {est}, measurements {fwd_peak} and {}",
                s.index,
                back.peak_bytes
            );
            produced.extend(outputs);
        }
        eprintln!("memory divided by {divisor}: {} stages", st.len());
    }
    // A node that does not fit on its own: a named refusal.
    let err = stages::partition(&graph, &report, 4, 1000).unwrap_err();
    assert!(err.contains("does not fit on its own"), "{err}");
}

/// Tied weights (a Llama whose language-model head is the embedding matrix), captured by the
/// PyTorch driver (`gg-python/tools/make_torch_fixtures.py`, captured at 2 examples, checked at 3):
/// **one** parameter, whose gradient sums both uses — PyTorch's, within 1e-5 — on one machine and
/// across stages. A stage split that puts the embedding and the head in different stages gives each
/// stage a partial gradient of the shared matrix; the grid sums them (every stage that uses a
/// parameter contributes, `gg-control` staged aggregation), in decreasing stage order, like the
/// segmented run: the same bytes.
#[test]
fn a_tied_parameter_is_one_parameter_on_one_machine_and_across_stages() {
    use gg_protocol::tensor_file::{BundleF32, TensorLimits};

    let dir = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
    let bundle = |file: &str| {
        BundleF32::decode(
            &std::fs::read(format!("{dir}/{file}")).unwrap(),
            &TensorLimits::default(),
        )
        .unwrap()
        .entries
    };
    let g: ModelGraphV1 = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/llama-tied-mini.json")).unwrap(),
    )
    .unwrap();
    let torch = bundle("llama-tied-mini.torch.ggb");
    let torch_grad = bundle("llama-tied-mini.grad.ggb");
    let shared = "inner.model.embed_tokens.weight";
    assert!(g.parameters.iter().any(|p| p.name == shared));
    assert!(
        !g.parameters.iter().any(|p| p.name.contains("lm_head")),
        "the head is not a second parameter"
    );
    let users = g
        .nodes
        .iter()
        .filter(|n| n.inputs.iter().any(|i| i == shared))
        .count();
    assert!(users >= 2, "the shared matrix has {users} consumer(s)");

    let prepared = Prepared::new(&g).unwrap();
    let report = graph::validate(&g).unwrap();
    let p: BTreeMap<String, TensorF32> = g
        .parameters
        .iter()
        .map(|q| (q.name.clone(), torch[&q.name].clone()))
        .collect();
    let data = BTreeMap::from([
        (g.inputs[0].name.clone(), torch["__x"].clone()),
        ("y".to_string(), torch["__y"].clone()),
    ]);
    // One machine: PyTorch's loss and gradients, the shared one included.
    let one = exec::gradient_cpu(&prepared, &p, &data, 0).unwrap();
    let expected = torch["__loss"].data[0];
    assert!(
        (one.loss - expected).abs() <= 1e-5 * expected.max(1.0),
        "loss {} vs {expected}",
        one.loss
    );
    assert_eq!(one.gradients.len(), torch_grad.len());
    for (name, r) in &torch_grad {
        let got = &one.gradients[name];
        let scale = r.data.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        let gap = got
            .data
            .iter()
            .zip(&r.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / scale;
        assert!(gap <= 1e-5, "{name}: relative gap {gap:e} to PyTorch");
    }
    // Across stages: bounds such that the embedding and the head are in different stages.
    let n = report.order.len();
    for k in [2usize, 3] {
        let bounds = stages::even_bounds(n, k);
        let st = stages::stages(&g, &report, &bounds).unwrap();
        let holders = st
            .iter()
            .filter(|s| s.params.iter().any(|q| q == shared))
            .count();
        assert!(
            holders >= 2,
            "{k} stages: the shared matrix is held by {holders} stage(s)"
        );
        let (loss, grads) = run_chain(&prepared, &st, &p, &data);
        let segmented = exec::gradient_cpu_with(
            &prepared,
            &p,
            &data,
            0,
            ExecOptions {
                segments: k,
                offload_parameters: false,
            },
        )
        .unwrap();
        assert_eq!(loss, segmented.loss, "{k} stages: loss");
        for (name, r) in &segmented.gradients {
            assert_eq!(&grads[name], r, "{k} stages: gradient of {name}");
        }
    }
}

/// The greedy partition as first written: each candidate stage derived from the whole graph with
/// the public functions. A reference for the indexed partition, which must give the same bounds.
fn naive_partition(
    graph: &ModelGraphV1,
    report: &graph::GraphReport,
    batch: u64,
    cap: u64,
) -> Option<Vec<u32>> {
    let n = report.order.len();
    let mut bounds = vec![0u32];
    let mut start = 0;
    let fits = |bounds: &[u32], end: usize| {
        let mut b = bounds.to_vec();
        b.push(end as u32);
        if end < n {
            b.push(n as u32);
        }
        let st = stages::stages(graph, report, &b).ok()?;
        Some(stages::stage_memory(graph, report, &st[bounds.len() - 1], batch) <= cap)
    };
    while start < n {
        let mut end = start + 1;
        if !fits(&bounds, end)? {
            return None;
        }
        while end < n && fits(&bounds, end + 1)? {
            end += 1;
        }
        bounds.push(end as u32);
        start = end;
    }
    Some(bounds)
}

/// Partitioning derives each candidate stage from an index built once: the same bounds as the
/// naive partition, and a 30-layer model partitions in well under a second, where the naive one
/// grew with the cube of the number of nodes.
#[test]
fn indexed_partition_matches_the_naive_one() {
    for graph in [hf_llama(), lm(Positions::Learned, 3)] {
        let report = graph::validate(&graph).unwrap();
        let whole = stages::stages(&graph, &report, &[0, report.order.len() as u32]).unwrap();
        let alone = stages::stage_memory(&graph, &report, &whole[0], 4);
        for divisor in [1u64, 2, 3, 5, 8] {
            let cap = alone / divisor;
            let fast = stages::partition(&graph, &report, 4, cap).ok();
            assert_eq!(fast, naive_partition(&graph, &report, 4, cap), "÷{divisor}");
            // Every stage the index derives is the one `stages` derives.
            if let Some(b) = fast {
                for s in stages::stages(&graph, &report, &b).unwrap() {
                    assert!(stages::stage_memory(&graph, &report, &s, 4) <= cap);
                }
            }
        }
    }
    let big = lm(Positions::Learned, 30);
    let report = graph::validate(&big).unwrap();
    let whole = stages::stages(&big, &report, &[0, report.order.len() as u32]).unwrap();
    let alone = stages::stage_memory(&big, &report, &whole[0], 4);
    let t = std::time::Instant::now();
    let bounds = stages::partition(&big, &report, 4, alone / 6).unwrap();
    eprintln!(
        "{} nodes in {} stages: {:?}",
        report.order.len(),
        bounds.len() - 1,
        t.elapsed()
    );
}
