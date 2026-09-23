//! `graph.stage.v1` end to end in one process (gg-server/docs/ROADMAP.md P11 D2): a full step executed as stage
//! tasks — each with only its own parameters, input files and output files — gives back the
//! reference gradient; every task is accepted by the validator; a tampered activation is rejected
//! at the stage that produced it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_compute::graph::exec::{self, ExecOptions, Prepared};
use gg_compute::graph::stage_task::{self, Expected, Resolved};
use gg_compute::graph::stages;
use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
use gg_compute::graph::{self, ModelGraphV1};
use gg_compute::{EngineSelection, execute_task_limited, validate_definition, verify_task};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::{ArtifactRef, DatasetSpecV1, GraphStageParamsV1};
use gg_protocol::tensor_file::{BundleF32, TensorF32, TensorLimits};

const SEQ: u64 = 8;
const VOCAB: u64 = 30;
const N: u32 = 6;

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

struct Store {
    dir: tempfile::TempDir,
    files: BTreeMap<Sha256Digest, PathBuf>,
}

impl Store {
    fn put(&mut self, bytes: &[u8]) -> ArtifactRef {
        let hash = Sha256Digest::of_bytes(bytes);
        let path = self.dir.path().join(hash.to_hex());
        std::fs::write(&path, bytes).unwrap();
        self.files.insert(hash, path);
        ArtifactRef {
            hash,
            size: bytes.len() as u64,
        }
    }
    fn path(&self, r: &ArtifactRef) -> PathBuf {
        self.files[&r.hash].clone()
    }
}

struct Run {
    graph: ModelGraphV1,
    resolved: Resolved,
    bounds: Vec<u32>,
    graph_ref: ArtifactRef,
    shard_refs: Vec<ArtifactRef>,
    data_refs: BTreeMap<String, ArtifactRef>,
    params: BTreeMap<String, TensorF32>,
    data: BTreeMap<String, TensorF32>,
}

fn setup(store: &mut Store, k: usize) -> Run {
    let graph = model();
    let report = graph::validate(&graph).unwrap();
    let bounds = stages::even_bounds(report.order.len(), k);
    let graph_bytes = serde_json::to_vec(&graph).unwrap();
    let graph_ref = store.put(&graph_bytes);
    let resolved = stage_task::resolve(graph.clone(), &bounds).unwrap();
    let params = exec::init_parameters(&graph, 5);
    let shard_refs = resolved
        .plan
        .shards
        .iter()
        .map(|s| {
            let part: BTreeMap<String, TensorF32> = s
                .params
                .iter()
                .map(|n| (n.clone(), params[n].clone()))
                .collect();
            store.put(&BundleF32::new(part).unwrap().encode())
        })
        .collect();
    let mut d = Drbg::new("test/stage-task", 1);
    let ids = |d: &mut Drbg| {
        TensorF32::new(
            vec![N as u64, SEQ],
            (0..N as u64 * SEQ)
                .map(|_| d.next_index(VOCAB as u32) as f32)
                .collect(),
        )
        .unwrap()
    };
    let data = BTreeMap::from([
        ("ids".to_string(), ids(&mut d)),
        ("y".to_string(), ids(&mut d)),
    ]);
    let data_refs = data
        .iter()
        .map(|(n, t)| (n.clone(), store.put(&t.encode())))
        .collect();
    Run {
        graph,
        resolved,
        bounds,
        graph_ref,
        shard_refs,
        data_refs,
        params,
        data,
    }
}

/// The host-memory limit the control plane gives this stage task.
fn stage_ram(run: &Run, p: &GraphStageParamsV1) -> u64 {
    gg_compute::catalog::graph_stage_ram_bytes(stages::stage_memory(
        &run.graph,
        run.resolved.prepared.report(),
        &run.resolved.stages[p.stage as usize],
        p.example_count as u64,
    ))
}

fn params_for(run: &Run, stage: usize, backward: bool) -> GraphStageParamsV1 {
    let (flops, act) = stage_task::stage_costs(&run.resolved, stage);
    GraphStageParamsV1 {
        graph_hash: run.graph_ref.hash.to_hex(),
        graph_bytes: run.graph_ref.size,
        parameters: run.resolved.prepared.report().parameters,
        stage_bounds: run.bounds.clone(),
        stage: stage as u32,
        backward,
        datasets: vec![
            DatasetSpecV1 {
                name: "ids".into(),
                shape: vec![N as u64, SEQ],
                dtype: "i64".into(),
            },
            DatasetSpecV1 {
                name: "y".into(),
                shape: vec![N as u64, SEQ],
                dtype: "i64".into(),
            },
        ],
        round_seed: 7,
        example_start: 0,
        example_count: N,
        total_examples: N as u64,
        training_flops_per_example: flops,
        activation_bytes_per_example: act,
        gradient_bf16: false,
    }
}

/// The grid coordinator in miniature: it assembles a task's inputs from the outputs already
/// produced.
fn task_inputs(
    run: &Run,
    p: &GraphStageParamsV1,
    produced: &BTreeMap<String, ArtifactRef>,
    received: &BTreeMap<(String, usize), ArtifactRef>,
) -> BTreeMap<String, ArtifactRef> {
    let mut inputs = BTreeMap::from([("graph".to_string(), run.graph_ref)]);
    for (name, what) in stage_task::expected_inputs(&run.resolved, p) {
        let r = match what {
            Expected::Shard(j) => run.shard_refs[j],
            Expected::Dataset(t) => run.data_refs[&t],
            Expected::Activation(t) => produced[&t],
            Expected::OutputGradient(t, c) => received[&(t, c)],
        };
        inputs.insert(name, r);
    }
    inputs
}

fn files(store: &Store, inputs: &BTreeMap<String, ArtifactRef>) -> BTreeMap<String, PathBuf> {
    inputs
        .iter()
        .map(|(n, r)| (n.clone(), store.path(r)))
        .collect()
}

#[test]
fn a_step_as_stage_tasks_gives_back_the_gradient_and_each_task_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store {
        dir,
        files: BTreeMap::new(),
    };
    let k = 3;
    let run = setup(&mut store, k);
    let out_root = tempfile::tempdir().unwrap();
    let mut produced: BTreeMap<String, ArtifactRef> = BTreeMap::new();
    let mut received: BTreeMap<(String, usize), ArtifactRef> = BTreeMap::new();
    let mut gradients: BTreeMap<String, TensorF32> = BTreeMap::new();
    let mut loss = None;
    let limits = TensorLimits {
        max_elements: 1 << 24,
    };
    let mut accepted = 0;
    let order: Vec<(usize, bool)> = (0..k)
        .map(|s| (s, false))
        .chain((0..k).rev().map(|s| (s, true)))
        .collect();
    for (s, backward) in order {
        let p = params_for(&run, s, backward);
        let inputs = task_inputs(&run, &p, &produced, &received);
        let def = gg_compute::catalog::graph_stage_definition(
            p.clone(),
            inputs.clone(),
            stage_ram(&run, &p),
            stage_task::output_bytes(&run.resolved, &p),
        )
        .unwrap();
        let task = validate_definition(&def).unwrap();
        let out = out_root.path().join(format!("{s}-{backward}"));
        let report = execute_task_limited(
            EngineSelection::FlexCpu,
            &task,
            &files(&store, &inputs),
            &out,
            None,
        )
        .unwrap();
        let mut names: Vec<&String> = report.outputs.keys().collect();
        names.sort();
        assert_eq!(
            names.iter().map(|n| n.as_str()).collect::<Vec<_>>(),
            stage_task::expected_outputs(&run.resolved, &p)
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        // The validator recomputes from the same inputs.
        let candidates = report
            .outputs
            .iter()
            .map(|(n, f)| (n.clone(), f.path.clone()))
            .collect();
        let claimed = report
            .outputs
            .iter()
            .map(|(n, f)| (n.clone(), f.artifact_ref()))
            .collect();
        let v = verify_task(
            &task,
            &files(&store, &inputs),
            &candidates,
            &claimed,
            report.engine.numeric_profile,
        )
        .unwrap();
        assert_eq!(
            v.outcome,
            VerificationOutcome::Accepted,
            "stage {s} {backward}: {:?}",
            v.checks
        );
        accepted += 1;
        // What flows to the following tasks.
        let st = &run.resolved.stages[s];
        for (name, f) in &report.outputs {
            let r = store.put(&std::fs::read(&f.path).unwrap());
            if let Some(k) = name.strip_prefix("out.") {
                produced.insert(st.outputs[k.parse::<usize>().unwrap()].clone(), r);
            } else if let Some(k) = name.strip_prefix("din.") {
                let tensor = st.inputs[k.parse::<usize>().unwrap()].clone();
                // This gradient goes to the producer of `tensor`; its rank among the consumers.
                let producer = run
                    .resolved
                    .stages
                    .iter()
                    .position(|t| t.outputs.contains(&tensor))
                    .unwrap();
                let c = run
                    .resolved
                    .consumers(producer, &tensor)
                    .iter()
                    .position(|x| *x == s)
                    .unwrap();
                received.insert((tensor, c), r);
            } else if name.starts_with("gradient.") {
                for (pn, g) in BundleF32::decode(&std::fs::read(&f.path).unwrap(), &limits)
                    .unwrap()
                    .entries
                {
                    match gradients.get_mut(&pn) {
                        Some(acc) => acc.data.iter_mut().zip(&g.data).for_each(|(a, b)| *a += *b),
                        None => {
                            gradients.insert(pn, g);
                        }
                    }
                }
            } else if name == stage_task::META_NAME {
                let m: stage_task::StageMeta =
                    serde_json::from_slice(&std::fs::read(&f.path).unwrap()).unwrap();
                if m.loss.is_some() {
                    loss = m.loss;
                }
            }
        }
    }
    assert_eq!(accepted, 2 * k);
    // Reference: the segmented execution at the same bounds, on one machine.
    let prepared = Prepared::new(&run.graph).unwrap();
    let reference = exec::gradient_with_bounds::<burn::backend::Autodiff<burn::backend::Flex>>(
        &burn::backend::flex::FlexDevice,
        &prepared,
        &run.params,
        &run.data,
        7,
        ExecOptions {
            segments: k,
            offload_parameters: false,
        },
        Some(&run.bounds),
    )
    .unwrap();
    // 1. The tasks give back exactly the stage chain computed in memory: files, tasks and
    //    verification add nothing.
    let chain = in_memory_chain(&run, &prepared);
    assert_eq!(gradients.len(), chain.len());
    for (name, t) in &chain {
        assert_eq!(
            &gradients[name], t,
            "{name}: tasks and in-memory chain differ"
        );
    }
    // 2. Against the segmented execution on one machine: same loss, gradients equal up to
    //    rounding. When a residual connection spans a stage, the embedding may differ by one or
    //    two ulp (accumulation order of the `Gather` backward pass); measured ≤ 3.6e-7.
    assert_eq!(loss, Some(reference.loss));
    let mut worst = 0f32;
    for (name, t) in &reference.gradients {
        worst = gradients[name]
            .data
            .iter()
            .zip(&t.data)
            .map(|(a, b)| (a - b).abs())
            .fold(worst, f32::max);
    }
    eprintln!("stage tasks vs segmented execution: gap {worst:e}");
    assert!(worst <= 1e-6, "gap {worst:e}");
}

/// The same chain, in memory, without tasks or files.
fn in_memory_chain(run: &Run, prepared: &Prepared) -> BTreeMap<String, TensorF32> {
    type A = burn::backend::Autodiff<burn::backend::Flex>;
    let dev = burn::backend::flex::FlexDevice;
    let stages = &run.resolved.stages;
    let mut produced = run.data.clone();
    for s in stages {
        let only: BTreeMap<String, TensorF32> = s
            .params
            .iter()
            .map(|n| (n.clone(), run.params[n].clone()))
            .collect();
        let ins: BTreeMap<String, TensorF32> = s
            .inputs
            .iter()
            .map(|n| (n.clone(), produced[n].clone()))
            .collect();
        let (o, _) =
            exec::stage_forward::<burn::backend::Flex>(&dev, prepared, s, &only, &ins, 7).unwrap();
        produced.extend(o);
    }
    let mut received: BTreeMap<String, Vec<TensorF32>> = BTreeMap::new();
    let mut total: BTreeMap<String, TensorF32> = BTreeMap::new();
    for s in stages.iter().rev() {
        let only: BTreeMap<String, TensorF32> = s
            .params
            .iter()
            .map(|n| (n.clone(), run.params[n].clone()))
            .collect();
        let ins: BTreeMap<String, TensorF32> = s
            .inputs
            .iter()
            .map(|n| (n.clone(), produced[n].clone()))
            .collect();
        let b = exec::stage_backward::<A>(&dev, prepared, s, &only, &ins, &received, 7).unwrap();
        for (n, g) in b.input_gradients {
            received.entry(n).or_default().push(g);
        }
        for (n, g) in b.gradients {
            match total.get_mut(&n) {
                Some(acc) => acc.data.iter_mut().zip(&g.data).for_each(|(a, b)| *a += *b),
                None => {
                    total.insert(n, g);
                }
            }
        }
    }
    total
}

#[test]
fn a_tampered_activation_is_rejected_and_the_declared_memory_is_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store {
        dir,
        files: BTreeMap::new(),
    };
    let run = setup(&mut store, 3);
    let p = params_for(&run, 0, false);
    let inputs = task_inputs(&run, &p, &BTreeMap::new(), &BTreeMap::new());
    let def = gg_compute::catalog::graph_stage_definition(
        p.clone(),
        inputs.clone(),
        stage_ram(&run, &p),
        stage_task::output_bytes(&run.resolved, &p),
    )
    .unwrap();
    let task = validate_definition(&def).unwrap();
    let out = tempfile::tempdir().unwrap();
    // Declared memory too small: refused before any compute.
    let need = stages::stage_memory(
        &run.graph,
        run.resolved.prepared.report(),
        &run.resolved.stages[0],
        N as u64,
    );
    let err = execute_task_limited(
        EngineSelection::FlexCpu,
        &task,
        &files(&store, &inputs),
        out.path(),
        Some(need - 1),
    )
    .unwrap_err();
    assert!(err.to_string().contains("memory limit"), "{err}");
    let report = execute_task_limited(
        EngineSelection::FlexCpu,
        &task,
        &files(&store, &inputs),
        out.path(),
        Some(need),
    )
    .unwrap();
    // A tampered output with a recomputed hash: only the comparison can catch it.
    let (name, f) = report
        .outputs
        .iter()
        .find(|(n, _)| n.starts_with("out."))
        .unwrap();
    let mut t = TensorF32::read_from_path(&f.path, &Default::default()).unwrap();
    t.data[3] += 0.25;
    let bytes = t.encode();
    let bad = out.path().join("bad.ggt");
    std::fs::write(&bad, &bytes).unwrap();
    let mut candidates: BTreeMap<String, PathBuf> = report
        .outputs
        .iter()
        .map(|(n, f)| (n.clone(), f.path.clone()))
        .collect();
    let mut claimed: BTreeMap<String, ArtifactRef> = report
        .outputs
        .iter()
        .map(|(n, f)| (n.clone(), f.artifact_ref()))
        .collect();
    candidates.insert(name.clone(), bad);
    claimed.insert(
        name.clone(),
        ArtifactRef {
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        },
    );
    let v = verify_task(
        &task,
        &files(&store, &inputs),
        &candidates,
        &claimed,
        report.engine.numeric_profile,
    )
    .unwrap();
    assert_eq!(v.outcome, VerificationOutcome::Rejected);
    // A shard from another stage in place of the right one: refused at load time.
    let mut wrong = inputs.clone();
    let key = wrong
        .keys()
        .find(|k| k.starts_with("checkpoint."))
        .unwrap()
        .clone();
    wrong.insert(key, *run.shard_refs.last().unwrap());
    let def2 = gg_compute::catalog::graph_stage_definition(
        p.clone(),
        wrong.clone(),
        stage_ram(&run, &p),
        stage_task::output_bytes(&run.resolved, &p),
    )
    .unwrap();
    let task2 = validate_definition(&def2).unwrap();
    assert!(
        execute_task_limited(
            EngineSelection::FlexCpu,
            &task2,
            &files(&store, &wrong),
            out.path(),
            None
        )
        .is_err()
    );
}
