//! Sharded checkpoint (gg-server/docs/ROADMAP.md, P11 C2). What must hold:
//!
//! 1. sharding is a deterministic function that covers each parameter once, respects the target
//!    size, follows first-use order, and refuses by naming the remedy;
//! 2. a sharded task computes **exactly** the same gradient as a single-piece task;
//! 3. the validator accepts it, and rejects a tampered, missing or misplaced shard.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_compute::graph::exec::init_parameters;
use gg_compute::graph::shards::{self, MAX_SHARD_BYTES, MAX_SHARDS};
use gg_compute::graph::task::{self, GRAPH_INPUT};
use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
use gg_compute::graph::{self, ModelGraphV1};
use gg_compute::{EngineSelection, execute_task, validate_definition, verify_task};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::{ArtifactRef, DatasetSpecV1, GraphGradParamsV1};
use gg_protocol::tensor_file::{BundleF32, TensorF32, TensorLimits};

const SEQ: u64 = 8;
const VOCAB: u64 = 30;

/// A transformer with frozen parameters (RoPE tables, the −1 constant): shards without a gradient
/// exist.
fn model() -> ModelGraphV1 {
    transformer(TransformerDims {
        layers: 2,
        dim: 32,
        heads: 4,
        seq: SEQ,
        vocab: VOCAB,
        ffn: 64,
        causal: true,
        positions: Positions::rotary(10000.0),
    })
    .unwrap()
}

#[test]
fn sharding_covers_each_parameter_once_and_respects_the_target() {
    let graph = model();
    let report = graph::validate(&graph).unwrap();
    let whole = shards::plan(&graph, &report, 0).unwrap();
    assert!(!whole.sharded);
    assert_eq!(whole.shards.len(), 1);
    assert_eq!(whole.checkpoint_name(0), "checkpoint");
    assert_eq!(
        whole.checkpoint_bytes(),
        task::checkpoint_bundle_size(&graph)
    );
    assert_eq!(whole.gradient_bytes(), task::gradient_bundle_size(&graph));

    for target in [2_000u64, 8_000, 20_000, 1 << 20] {
        let plan = shards::plan(&graph, &report, target).unwrap();
        assert!(plan.sharded);
        // Deterministic.
        assert_eq!(plan, shards::plan(&graph, &report, target).unwrap());
        // Each parameter exactly once.
        let mut seen: Vec<&str> = plan
            .shards
            .iter()
            .flat_map(|s| s.params.iter().map(String::as_str))
            .collect();
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n, "duplicated parameter");
        assert_eq!(n, graph.parameters.len());
        // Below the target, except for a single parameter larger than it.
        for s in &plan.shards {
            assert!(
                s.checkpoint_bytes <= target || s.params.len() == 1,
                "shard {s:?} above {target}"
            );
        }
        // Shards without a trained parameter: no gradient.
        for s in &plan.shards {
            let trained = s
                .params
                .iter()
                .any(|n| graph.parameters.iter().any(|p| &p.name == n && p.trainable));
            assert_eq!(s.gradient_bytes > 0, trained, "{s:?}");
        }
        // Sizes add up beyond a single container (one header per shard).
        assert!(plan.checkpoint_bytes() >= whole.checkpoint_bytes());
        eprintln!("target {target}: {} shards", plan.shards.len());
    }
    // First-use order: the embedding comes first.
    let plan = shards::plan(&graph, &report, 2_000).unwrap();
    assert_eq!(plan.shards[0].params[0], "emb");
}

#[test]
fn sharding_refuses_by_naming_the_remedy() {
    // Too many shards: eight layers make more than forty parameters, one per shard.
    let deep = transformer(TransformerDims {
        layers: 8,
        dim: 32,
        heads: 4,
        seq: SEQ,
        vocab: VOCAB,
        ffn: 64,
        causal: true,
        positions: Positions::rotary(10000.0),
    })
    .unwrap();
    let deep_report = graph::validate(&deep).unwrap();
    let err = shards::plan(&deep, &deep_report, 64).unwrap_err();
    assert!(err.contains(&MAX_SHARDS.to_string()), "{err}");
    // A parameter larger than a shard: tile the matrix.
    let mut big = model();
    big.parameters[0].shape = vec![(MAX_SHARD_BYTES / 4) / 32 + 1, 32];
    let big_report = graph::validate(&big).expect("enlarged embedding is valid");
    let err = shards::plan(&big, &big_report, 256 << 20).unwrap_err();
    assert!(err.contains("tile the matrix"), "{err}");
    assert_eq!(shards::default_shard_bytes(1 << 20), 0);
    assert_eq!(shards::default_shard_bytes(300 << 20), 256 << 20);
    // Large enough to exceed 40 shards of 256 MiB: the target goes up.
    assert!(shards::default_shard_bytes(20 << 30) > 256 << 20);
}

struct Task {
    dir: tempfile::TempDir,
    inputs: BTreeMap<String, PathBuf>,
    def: gg_protocol::messages::TaskDefinitionV1,
}

fn task_for(shard_bytes: u64) -> Task {
    let dir = tempfile::tempdir().unwrap();
    let graph = model();
    let report = graph::validate(&graph).unwrap();
    let plan = shards::plan(&graph, &report, shard_bytes).unwrap();
    let graph_bytes = serde_json::to_vec(&graph).unwrap();
    let weights = init_parameters(&graph, 7);
    let total = 48u64;
    let mut d = Drbg::new("test/shards", 1);
    let ids = |d: &mut Drbg| {
        TensorF32::new(
            vec![total, SEQ],
            (0..total * SEQ)
                .map(|_| d.next_index(VOCAB as u32) as f32)
                .collect(),
        )
        .unwrap()
    };
    let (x, y) = (ids(&mut d), ids(&mut d));
    let mut files: Vec<(String, Vec<u8>)> = vec![
        (GRAPH_INPUT.into(), graph_bytes.clone()),
        ("ids".into(), x.encode()),
        ("y".into(), y.encode()),
    ];
    for (i, s) in plan.shards.iter().enumerate() {
        let part: BTreeMap<String, TensorF32> = s
            .params
            .iter()
            .map(|n| (n.clone(), weights[n].clone()))
            .collect();
        files.push((
            plan.checkpoint_name(i),
            BundleF32::new(part).unwrap().encode(),
        ));
    }
    let mut inputs = BTreeMap::new();
    let mut refs = BTreeMap::new();
    for (name, bytes) in files {
        let path = dir.path().join(&name);
        std::fs::write(&path, &bytes).unwrap();
        inputs.insert(name.clone(), path);
        refs.insert(
            name,
            ArtifactRef {
                hash: Sha256Digest::of_bytes(&bytes),
                size: bytes.len() as u64,
            },
        );
    }
    let p = GraphGradParamsV1 {
        graph_hash: Sha256Digest::of_bytes(&graph_bytes).to_hex(),
        graph_bytes: graph_bytes.len() as u64,
        parameters: report.parameters,
        checkpoint_bytes: plan.checkpoint_bytes(),
        gradient_bytes: plan.gradient_bytes(),
        segments: 1,
        offload_parameters: false,
        training_flops_per_example: report.training_flops_per_example,
        activation_bytes_per_example: report.activation_bytes_per_example,
        datasets: vec![
            DatasetSpecV1 {
                name: "ids".into(),
                shape: vec![total, SEQ],
                dtype: "i64".into(),
            },
            DatasetSpecV1 {
                name: "y".into(),
                shape: vec![total, SEQ],
                dtype: "i64".into(),
            },
        ],
        round_seed: 42,
        example_start: 8,
        example_count: 8,
        total_examples: total,
        shard_bytes,
        batch_inputs: false,
        gradient_bf16: false,
    };
    let def = gg_compute::catalog::graph_grad_definition(p, refs).expect("valid definition");
    Task { dir, inputs, def }
}

fn gradients_of(report: &gg_compute::ExecutionReport) -> BTreeMap<String, TensorF32> {
    let mut out = BTreeMap::new();
    for (name, f) in &report.outputs {
        if name.starts_with("gradient") {
            let b = BundleF32::read_from_path(
                &f.path,
                &TensorLimits {
                    max_elements: 1 << 24,
                },
            )
            .unwrap();
            out.extend(b.entries);
        }
    }
    out
}

#[test]
fn a_sharded_task_computes_exactly_the_same_gradient_and_is_accepted() {
    let whole = task_for(0);
    let whole_task = validate_definition(&whole.def).unwrap();
    let whole_report = execute_task(
        EngineSelection::FlexCpu,
        &whole_task,
        &whole.inputs,
        &whole.dir.path().join("out"),
    )
    .unwrap();
    let reference = gradients_of(&whole_report);

    let cut = task_for(6_000);
    let names: Vec<&String> = cut.def.inputs.keys().collect();
    assert!(
        names
            .iter()
            .filter(|n| n.starts_with("checkpoint."))
            .count()
            >= 4,
        "{names:?}"
    );
    let cut_task = validate_definition(&cut.def).unwrap();
    let report = execute_task(
        EngineSelection::FlexCpu,
        &cut_task,
        &cut.inputs,
        &cut.dir.path().join("out"),
    )
    .unwrap();
    let got = gradients_of(&report);
    assert_eq!(got.len(), reference.len());
    for (name, t) in &reference {
        assert_eq!(&got[name], t, "{name}: sharding changes the gradient");
    }
    // One output per trained shard, none for frozen shards.
    assert!(!report.outputs.contains_key("gradient"));
    assert!(
        report
            .outputs
            .keys()
            .filter(|n| n.starts_with("gradient."))
            .count()
            >= 3
    );

    let (paths, refs): (BTreeMap<_, _>, BTreeMap<_, _>) = report
        .outputs
        .iter()
        .map(|(n, f)| ((n.clone(), f.path.clone()), (n.clone(), f.artifact_ref())))
        .unzip();
    let v = verify_task(
        &cut_task,
        &cut.inputs,
        &paths,
        &refs,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Accepted, "{:?}", v.checks);

    // A tampered shard with a recomputed hash: only the comparison can catch it.
    let victim = refs
        .keys()
        .find(|n| n.starts_with("gradient."))
        .unwrap()
        .clone();
    let mut bad_paths = paths.clone();
    let mut bad_refs = refs.clone();
    let mut b = BundleF32::read_from_path(
        &paths[&victim],
        &TensorLimits {
            max_elements: 1 << 24,
        },
    )
    .unwrap();
    let first = b.entries.keys().next().unwrap().clone();
    b.entries.get_mut(&first).unwrap().data[0] += 0.5;
    let bytes = b.encode();
    let p = cut.dir.path().join("altered.ggb");
    std::fs::write(&p, &bytes).unwrap();
    bad_paths.insert(victim.clone(), p);
    bad_refs.insert(
        victim.clone(),
        ArtifactRef {
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        },
    );
    let v = verify_task(
        &cut_task,
        &cut.inputs,
        &bad_paths,
        &bad_refs,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);

    // A missing shard: rejected on the names, before any recomputation.
    let mut missing = refs.clone();
    missing.remove(&victim);
    let v = verify_task(
        &cut_task,
        &cut.inputs,
        &paths,
        &missing,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);
    assert!(
        v.checks
            .iter()
            .any(|c| c.name == "outputs.names" && !c.passed)
    );

    // Two swapped shards: sizes or contents that no longer match their shard.
    let others: Vec<String> = refs
        .keys()
        .filter(|n| n.starts_with("gradient."))
        .cloned()
        .collect();
    let (a, c) = (&others[0], &others[1]);
    let mut swapped_paths = paths.clone();
    let mut swapped_refs = refs.clone();
    swapped_paths.insert(a.clone(), paths[c].clone());
    swapped_paths.insert(c.clone(), paths[a].clone());
    swapped_refs.insert(a.clone(), refs[c]);
    swapped_refs.insert(c.clone(), refs[a]);
    let v = verify_task(
        &cut_task,
        &cut.inputs,
        &swapped_paths,
        &swapped_refs,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);
}

#[test]
fn the_catalog_checks_the_declared_shards() {
    let cut = task_for(6_000);
    // A shard removed from the definition: the sum no longer adds up.
    let mut def = cut.def.clone();
    def.inputs.remove("checkpoint.1");
    assert!(validate_definition(&def).is_err());
    // An extra shard, out of sequence: the input count no longer adds up.
    let mut def = cut.def.clone();
    let r = def.inputs["checkpoint.0"];
    def.inputs.insert("checkpoint.99".into(), r);
    assert!(validate_definition(&def).is_err());
    // A shard file that does not match the sharding: refused at load time.
    let task = validate_definition(&cut.def).unwrap();
    let mut inputs = cut.inputs.clone();
    let (a, b) = (
        inputs["checkpoint.0"].clone(),
        inputs["checkpoint.1"].clone(),
    );
    inputs.insert("checkpoint.0".into(), b);
    inputs.insert("checkpoint.1".into(), a);
    let err = execute_task(
        EngineSelection::FlexCpu,
        &task,
        &inputs,
        &cut.dir.path().join("out2"),
    )
    .unwrap_err();
    assert!(err.to_string().contains("checkpoint."), "{err}");
}
