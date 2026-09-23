//! `graph.grad.v1` end to end in one process: catalog definition, execution from files,
//! verification by recomputation, refusal of tampered outputs and false declarations.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_compute::graph::exec::init_parameters;
use gg_compute::graph::task::{self, CHECKPOINT_INPUT, GRADIENT_OUTPUT, GRAPH_INPUT, META_NAME};
use gg_compute::graph::{self, ModelGraphV1};
use gg_compute::{EngineSelection, execute_task, validate_definition, verify_task};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::{ArtifactRef, DatasetSpecV1, GraphGradParamsV1};
use gg_protocol::tensor_file::{BundleF32, Dtype, TensorF32, TensorLimits, bf16_bits, bf16_value};

const MLP: &str = r#"{
  "format": "gg-graph/v1", "name": "mlp-task",
  "inputs": [{"name": "x", "shape": [0, 6]}],
  "parameters": [
    {"name": "w1", "shape": [6, 10], "init": "xavier_uniform"},
    {"name": "b1", "shape": [10], "init": "zeros"},
    {"name": "w2", "shape": [10, 3], "init": "xavier_uniform"},
    {"name": "b2", "shape": [3], "init": "zeros"}
  ],
  "nodes": [
    {"op": "MatMul", "inputs": ["x", "w1"], "output": "h"},
    {"op": "Add", "inputs": ["h", "b1"], "output": "hb"},
    {"op": "Relu", "inputs": ["hb"], "output": "a"},
    {"op": "MatMul", "inputs": ["a", "w2"], "output": "o"},
    {"op": "Add", "inputs": ["o", "b2"], "output": "logits"}
  ],
  "loss": {"cross_entropy": {"prediction": "logits", "target": {"name": "y", "shape": [0], "dtype": "i64"}}}
}"#;

struct Fixture {
    dir: tempfile::TempDir,
    inputs: BTreeMap<String, PathBuf>,
    def: gg_protocol::messages::TaskDefinitionV1,
}

fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> (PathBuf, ArtifactRef) {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    (
        path,
        ArtifactRef {
            hash: Sha256Digest::of_bytes(bytes),
            size: bytes.len() as u64,
        },
    )
}

fn fixture(total: u64, start: u64, count: u32) -> Fixture {
    fixture_with(total, start, count, false)
}

fn fixture_with(total: u64, start: u64, count: u32, gradient_bf16: bool) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let graph: ModelGraphV1 = serde_json::from_str(MLP).unwrap();
    let report = graph::validate(&graph).unwrap();
    let graph_bytes = MLP.as_bytes().to_vec();
    let checkpoint = BundleF32::new(init_parameters(&graph, 7)).unwrap();
    let ck_bytes = checkpoint.encode();
    let mut d = Drbg::new("test/graph-task", 1);
    let x = TensorF32::new(
        vec![total, 6],
        (0..total * 6).map(|_| d.next_sym_f32()).collect(),
    )
    .unwrap();
    let y = TensorF32::new(
        vec![total],
        (0..total).map(|_| d.next_index(3) as f32).collect(),
    )
    .unwrap();

    let mut inputs = BTreeMap::new();
    let mut refs = BTreeMap::new();
    for (name, bytes) in [
        (GRAPH_INPUT, graph_bytes.clone()),
        (CHECKPOINT_INPUT, ck_bytes.clone()),
        ("x", x.encode()),
        ("y", y.encode()),
    ] {
        let (path, r) = write(&dir, name, &bytes);
        inputs.insert(name.to_string(), path);
        refs.insert(name.to_string(), r);
    }
    let p = GraphGradParamsV1 {
        graph_hash: Sha256Digest::of_bytes(&graph_bytes).to_hex(),
        graph_bytes: graph_bytes.len() as u64,
        parameters: report.parameters,
        checkpoint_bytes: ck_bytes.len() as u64,
        gradient_bytes: if gradient_bf16 {
            BundleF32::encoded_size_as(
                graph
                    .parameters
                    .iter()
                    .map(|q| (q.name.as_str(), q.shape.as_slice())),
                Dtype::Bf16,
            )
        } else {
            task::gradient_bundle_size(&graph)
        },
        segments: 1,
        offload_parameters: false,
        training_flops_per_example: report.training_flops_per_example,
        activation_bytes_per_example: report.activation_bytes_per_example,
        datasets: vec![
            DatasetSpecV1 {
                name: "x".into(),
                shape: vec![total, 6],
                dtype: "f32".into(),
            },
            DatasetSpecV1 {
                name: "y".into(),
                shape: vec![total],
                dtype: "i64".into(),
            },
        ],
        round_seed: 42,
        example_start: start,
        example_count: count,
        total_examples: total,
        shard_bytes: 0,
        batch_inputs: false,
        gradient_bf16,
    };
    let def = gg_compute::catalog::graph_grad_definition(p, refs).expect("valid definition");
    Fixture { dir, inputs, def }
}

fn claimed(
    report: &gg_compute::ExecutionReport,
) -> (BTreeMap<String, PathBuf>, BTreeMap<String, ArtifactRef>) {
    let mut paths = BTreeMap::new();
    let mut refs = BTreeMap::new();
    for (name, f) in &report.outputs {
        paths.insert(name.clone(), f.path.clone());
        refs.insert(name.clone(), f.artifact_ref());
    }
    (paths, refs)
}

#[test]
fn execute_then_verify_accepts() {
    let f = fixture(64, 16, 8);
    let task = validate_definition(&f.def).unwrap();
    assert_eq!(task.params.compute_kind(), "graph.grad.v1");
    assert_eq!(task.input_shapes["x"], vec![64, 6]);
    let out = f.dir.path().join("out");
    let report = execute_task(EngineSelection::FlexCpu, &task, &f.inputs, &out).unwrap();
    assert!(report.outputs.contains_key(GRADIENT_OUTPUT));
    assert!(report.outputs.contains_key(META_NAME));
    assert!(report.metrics["loss"].is_finite());
    // The gradient has exactly the layout of the checkpoint.
    let g = BundleF32::read_from_path(&report.outputs[GRADIENT_OUTPUT].path, &Default::default())
        .unwrap();
    assert_eq!(g.entries.len(), 4);
    assert_eq!(g.entries["w1"].shape, vec![6, 10]);
    assert_eq!(
        report.outputs[GRADIENT_OUTPUT].size,
        8 + (2 + 2 + 8) * 4
            + g.entries
                .values()
                .map(|t| TensorF32::encoded_size(&t.shape))
                .sum::<u64>()
    );

    let (paths, refs) = claimed(&report);
    let v = verify_task(
        &task,
        &f.inputs,
        &paths,
        &refs,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Accepted, "{:?}", v.checks);
    assert!(
        v.checks
            .iter()
            .any(|c| c.name == "reference.recomputed" && c.passed)
    );
    assert!(
        v.checks
            .iter()
            .any(|c| c.name == "grad_meta.loss" && c.passed)
    );
    assert_eq!(v.artifacts_checked.len(), 2);
}

#[test]
fn two_slices_of_a_round_are_disjoint_and_deterministic() {
    let a = fixture(64, 0, 8);
    let b = fixture(64, 8, 8);
    let ia = task::round_indices(42, 64, 0, 8);
    let ib = task::round_indices(42, 64, 8, 8);
    assert!(ia.iter().all(|i| !ib.contains(i)), "disjoint slices");
    let ta = validate_definition(&a.def).unwrap();
    let tb = validate_definition(&b.def).unwrap();
    let ra = execute_task(
        EngineSelection::FlexCpu,
        &ta,
        &a.inputs,
        &a.dir.path().join("o"),
    )
    .unwrap();
    let rb = execute_task(
        EngineSelection::FlexCpu,
        &tb,
        &b.inputs,
        &b.dir.path().join("o"),
    )
    .unwrap();
    assert_ne!(
        ra.outputs[GRADIENT_OUTPUT].hash,
        rb.outputs[GRADIENT_OUTPUT].hash
    );
    let ra2 = execute_task(
        EngineSelection::FlexCpu,
        &ta,
        &a.inputs,
        &a.dir.path().join("o2"),
    )
    .unwrap();
    assert_eq!(
        ra.outputs[GRADIENT_OUTPUT].hash, ra2.outputs[GRADIENT_OUTPUT].hash,
        "same slice, same bytes"
    );
}

#[test]
fn a_tampered_gradient_is_rejected() {
    let f = fixture(64, 16, 8);
    let task = validate_definition(&f.def).unwrap();
    let report = execute_task(
        EngineSelection::FlexCpu,
        &task,
        &f.inputs,
        &f.dir.path().join("out"),
    )
    .unwrap();
    let (paths, mut refs) = claimed(&report);
    // Tamper with one gradient value and recompute the hash: only the comparison can catch it.
    let path = &paths[GRADIENT_OUTPUT];
    let mut g = BundleF32::read_from_path(path, &Default::default()).unwrap();
    g.entries.get_mut("w2").unwrap().data[0] += 0.5;
    let bytes = g.encode();
    std::fs::write(path, &bytes).unwrap();
    refs.insert(
        GRADIENT_OUTPUT.into(),
        ArtifactRef {
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        },
    );
    let v = verify_task(
        &task,
        &f.inputs,
        &paths,
        &refs,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);
    assert!(
        v.checks
            .iter()
            .any(|c| c.name.starts_with("outputs.gradient.w2") && !c.passed)
    );
    // Hash that no longer matches the file: rejected before any recomputation.
    let (paths, mut refs) = claimed(&report);
    refs.get_mut(GRADIENT_OUTPUT).unwrap().hash = Sha256Digest::of_bytes(b"other");
    let v = verify_task(
        &task,
        &f.inputs,
        &paths,
        &refs,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);
    assert!(!v.checks.iter().any(|c| c.name == "reference.recomputed"));
}

#[test]
fn a_false_declaration_is_refused() {
    let f = fixture(64, 16, 8);
    // Parameter count misstated in the task: refused when the graph is opened, before any compute.
    let mut def = f.def.clone();
    if let gg_protocol::messages::TaskParamsV1::GraphGrad(p) = &mut def.params {
        p.parameters += 1;
    }
    let task = validate_definition(&def).unwrap();
    let err = execute_task(
        EngineSelection::FlexCpu,
        &task,
        &f.inputs,
        &f.dir.path().join("out"),
    )
    .unwrap_err();
    assert!(err.to_string().contains("parameters"), "{err}");
    // Checkpoint size inconsistent with the artifact: refused as early as catalog validation.
    let mut def = f.def.clone();
    if let gg_protocol::messages::TaskParamsV1::GraphGrad(p) = &mut def.params {
        p.checkpoint_bytes += 4;
    }
    assert!(validate_definition(&def).is_err());
    // Profile not admitted: inconclusive verdict, never accepted.
    let task = validate_definition(&f.def).unwrap();
    let report = execute_task(
        EngineSelection::FlexCpu,
        &task,
        &f.inputs,
        &f.dir.path().join("out2"),
    )
    .unwrap();
    let (paths, refs) = claimed(&report);
    let v = verify_task(&task, &f.inputs, &paths, &refs, "unknown-profile").unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Inconclusive);
}

#[test]
fn named_optimizer_matches_the_mlp_kernel() {
    use gg_compute::grad::{CoordOptim, OptimizerState, apply_step};
    use gg_compute::graph::task::{NamedOptimizerState, aggregate_named, apply_step_named};
    use gg_compute::mlp::Weights;
    let mut d = Drbg::new("test/optim", 3);
    let mk = |shape: &[u64], d: &mut Drbg| {
        TensorF32::new(
            shape.to_vec(),
            (0..shape.iter().product::<u64>())
                .map(|_| d.next_sym_f32())
                .collect(),
        )
        .unwrap()
    };
    let w = Weights {
        w1: mk(&[4, 5], &mut d),
        b1: mk(&[5], &mut d),
        w2: mk(&[5, 2], &mut d),
        b2: mk(&[2], &mut d),
    };
    let g1 = Weights {
        w1: mk(&[4, 5], &mut d),
        b1: mk(&[5], &mut d),
        w2: mk(&[5, 2], &mut d),
        b2: mk(&[2], &mut d),
    };
    let g2 = Weights {
        w1: mk(&[4, 5], &mut d),
        b1: mk(&[5], &mut d),
        w2: mk(&[5, 2], &mut d),
        b2: mk(&[2], &mut d),
    };
    let named = |w: &Weights| -> BTreeMap<String, TensorF32> {
        w.named()
            .iter()
            .map(|(k, t)| ((*k).to_string(), (*t).clone()))
            .collect()
    };
    // Reference: aggregation and two Adam steps on the typed weights.
    let mut ref_w = w.clone();
    let mut ref_state = OptimizerState::new();
    let ref_g = gg_compute::grad::aggregate(&[(3, g1.clone()), (5, g2.clone())]).unwrap();
    apply_step(CoordOptim::Adam, 0.01, &mut ref_w, &ref_g, &mut ref_state);
    apply_step(CoordOptim::Adam, 0.01, &mut ref_w, &ref_g, &mut ref_state);
    // Same thing on the named parameters.
    let mut nw = named(&w);
    let mut ns = NamedOptimizerState::default();
    let ng = aggregate_named(&[(3, named(&g1)), (5, named(&g2))]).unwrap();
    apply_step_named(CoordOptim::Adam, 0.01, &mut nw, &ng, &mut ns).unwrap();
    apply_step_named(CoordOptim::Adam, 0.01, &mut nw, &ng, &mut ns).unwrap();
    for (k, t) in named(&ref_w) {
        assert_eq!(nw[&k].encode(), t.encode(), "{k}");
    }
    assert_eq!(ns.t, 2);
}

/// ADR-0035: the same slice with gradients in bf16 is half the bytes, holds the f32 gradient rounded
/// to nearest, and is accepted; the validator reads it as bf16 and nothing else.
#[test]
fn a_bf16_gradient_is_half_the_bytes_and_verified() {
    let f32_fixture = fixture(64, 16, 8);
    let f = fixture_with(64, 16, 8, true);
    let t32 = validate_definition(&f32_fixture.def).unwrap();
    let task = validate_definition(&f.def).unwrap();
    let r32 = execute_task(
        EngineSelection::FlexCpu,
        &t32,
        &f32_fixture.inputs,
        &f32_fixture.dir.path().join("out"),
    )
    .unwrap();
    let report = execute_task(
        EngineSelection::FlexCpu,
        &task,
        &f.inputs,
        &f.dir.path().join("out"),
    )
    .unwrap();
    let (s32, s16) = (
        r32.outputs[GRADIENT_OUTPUT].size,
        report.outputs[GRADIENT_OUTPUT].size,
    );
    // 4 index entries of 12 bytes and 4 headers stay; the 97 values take half.
    assert_eq!(s32 - s16, 2 * (6 * 10 + 10 + 10 * 3 + 3));
    let g32 =
        BundleF32::read_from_path(&r32.outputs[GRADIENT_OUTPUT].path, &Default::default()).unwrap();
    let g16 = BundleF32::read_from_path_as(
        &report.outputs[GRADIENT_OUTPUT].path,
        &TensorLimits::default(),
        Dtype::Bf16,
    )
    .unwrap();
    for (name, t) in &g32.entries {
        let rounded: Vec<f32> = t.data.iter().map(|x| bf16_value(bf16_bits(*x))).collect();
        assert_eq!(g16.entries[name].data, rounded, "{name}");
    }

    let (paths, refs) = claimed(&report);
    let v = verify_task(
        &task,
        &f.inputs,
        &paths,
        &refs,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Accepted, "{:?}", v.checks);

    // The f32 gradient handed in for a bf16 task: refused on its size, before any recomputation.
    let (mut paths32, mut refs32) = claimed(&r32);
    paths32.insert(META_NAME.into(), paths[META_NAME].clone());
    refs32.insert(META_NAME.into(), refs[META_NAME]);
    let v = verify_task(
        &task,
        &f.inputs,
        &paths32,
        &refs32,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);
    assert!(
        v.checks
            .iter()
            .any(|c| c.name == "outputs.gradient.size" && !c.passed)
    );

    // A wrong bf16 value, the hash recomputed: caught by the comparison.
    let mut bad = g16.clone();
    bad.entries.get_mut("w2").unwrap().data[0] += 0.5;
    let bytes = bad.encode_as(Dtype::Bf16).unwrap();
    std::fs::write(&paths[GRADIENT_OUTPUT], &bytes).unwrap();
    let mut refs_bad = refs.clone();
    refs_bad.insert(
        GRADIENT_OUTPUT.into(),
        ArtifactRef {
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        },
    );
    let v = verify_task(
        &task,
        &f.inputs,
        &paths,
        &refs_bad,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);
    assert!(
        v.checks
            .iter()
            .any(|c| c.name.starts_with("outputs.gradient.w2") && !c.passed)
    );

    // A task declaring bf16 with the f32 size (or the reverse) is refused by the catalog.
    let gg_protocol::messages::TaskParamsV1::GraphGrad(mut p) = f.def.params.clone() else {
        unreachable!()
    };
    p.gradient_bytes = task::gradient_bundle_size(&serde_json::from_str(MLP).unwrap());
    let def = gg_compute::catalog::graph_grad_definition(p, f.def.inputs.clone()).unwrap();
    let refused = validate_definition(&def)
        .map_err(|e| e.to_string())
        .and_then(|t| {
            execute_task(
                EngineSelection::FlexCpu,
                &t,
                &f.inputs,
                &f.dir.path().join("x"),
            )
            .map_err(|e| e.to_string())
        });
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains("gradient of this graph")),
        "{refused:?}"
    );
}

/// The server's step spreads its work across threads: the result must be bit for bit that of the
/// plain serial formulas, over several rounds, tensors of several work units, and a frozen parameter.
#[test]
fn the_parallel_server_step_equals_the_serial_formulas() {
    use gg_compute::grad::CoordOptim;
    use gg_compute::graph::task::{NamedOptimizerState, aggregate_named, apply_step_named};
    let mut d = Drbg::new("test/parallel-step", 3);
    let mut tensor = |n: usize| {
        TensorF32::new(vec![n as u64], (0..n).map(|_| d.next_sym_f32()).collect()).unwrap()
    };
    // 300 000 elements: several units of work; 7: fewer than one.
    let mut weights: BTreeMap<String, TensorF32> = [("a", 300_000), ("b", 7), ("frozen", 1000)]
        .map(|(k, n)| (k.to_string(), tensor(n)))
        .into();
    let frozen = weights["frozen"].clone();
    let mut serial = weights.clone();
    let (mut sm, mut sv, mut st) = (
        BTreeMap::<String, Vec<f32>>::new(),
        BTreeMap::<String, Vec<f32>>::new(),
        0i32,
    );
    let mut state = NamedOptimizerState::default();
    for _round in 0..3 {
        let parts: Vec<(u32, BTreeMap<String, TensorF32>)> = [3u32, 5, 8]
            .iter()
            .map(|n| {
                (
                    *n,
                    [("a", 300_000), ("b", 7)]
                        .map(|(k, n)| (k.to_string(), tensor(n)))
                        .into(),
                )
            })
            .collect();
        let g = aggregate_named(&parts).unwrap();
        apply_step_named(CoordOptim::Adam, 1e-3, &mut weights, &g, &mut state).unwrap();

        // The serial formulas, as they were written before.
        let total: f64 = parts.iter().map(|(n, _)| *n as f64).sum();
        let inv = (1.0 / total) as f32;
        st += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let (c1, c2) = (1.0 - b1.powi(st), 1.0 - b2.powi(st));
        for k in ["a", "b"] {
            let n = serial[k].data.len();
            let mut acc = vec![0.0f32; n];
            for (c, p) in &parts {
                let w = *c as f32;
                for (a, gi) in acc.iter_mut().zip(&p[k].data) {
                    *a += w * gi;
                }
            }
            acc.iter_mut().for_each(|a| *a *= inv);
            assert_eq!(acc, g[k].data, "mean of {k}");
            let m = sm.entry(k.into()).or_insert(vec![0.0; n]);
            let v = sv.entry(k.into()).or_insert(vec![0.0; n]);
            for (mi, gi) in m.iter_mut().zip(&acc) {
                *mi = b1 * *mi + (1.0 - b1) * gi;
            }
            for (vi, gi) in v.iter_mut().zip(&acc) {
                *vi = b2 * *vi + (1.0 - b2) * gi * gi;
            }
            let w = serial.get_mut(k).unwrap();
            for ((wi, mi), vi) in w.data.iter_mut().zip(m.iter()).zip(v.iter()) {
                *wi -= 1e-3f32 * (mi / c1) / ((vi / c2).sqrt() + eps);
            }
        }
    }
    assert_eq!(weights, serial);
    assert_eq!(weights["frozen"], frozen);
    assert_eq!(state.m.as_ref().unwrap()["a"].data, sm["a"]);
    assert_eq!(state.v.as_ref().unwrap()["b"].data, sv["b"]);
    // Replaying only the moments from the same gradient gives the same state (ADR-0036).
    let parts: Vec<(u32, BTreeMap<String, TensorF32>)> = vec![(
        4,
        [("a", 300_000), ("b", 7)]
            .map(|(k, n)| (k.to_string(), tensor(n)))
            .into(),
    )];
    let g = aggregate_named(&parts).unwrap();
    let mut replayed = state.clone();
    gg_compute::graph::task::advance_moments_named(CoordOptim::Adam, &g, &mut replayed).unwrap();
    apply_step_named(CoordOptim::Adam, 1e-3, &mut weights, &g, &mut state).unwrap();
    assert_eq!(replayed, state);
}
