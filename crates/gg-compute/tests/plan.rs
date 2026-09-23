//! Distribution plan on the real graph (ADR-0017).

use gg_compute::graph::plan::{self, Bound, Machine, PlanError, Workload};
use gg_compute::graph::templates::{TransformerDims, transformer};
use gg_compute::graph::{self, ModelGraphV1};

const GIB: u64 = 1 << 30;

fn home(gflops: f64, memory_gib: u64) -> Machine {
    Machine {
        flops_per_s: gflops * 1e9,
        memory_bytes: memory_gib * GIB,
        uplink_bytes_per_s: 10e6 / 8.0,
        downlink_bytes_per_s: 100e6 / 8.0,
        host_link_bytes_per_s: 16e9,
    }
}

fn work(examples: u64, batch: u64, contributors: u64, local_steps: u64) -> Workload {
    Workload {
        examples,
        batch,
        contributors,
        local_steps,
        audit_rate: 0.25,
        topology: plan::Topology::Peer,
        coordinator_link: None,
        memory: plan::MemoryPolicy::default(),
    }
}

fn small_transformer() -> ModelGraphV1 {
    transformer(TransformerDims {
        layers: 4,
        dim: 256,
        heads: 4,
        seq: 128,
        vocab: 1000,
        ffn: 1024,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    })
    .expect("valid skeleton")
}

#[test]
fn the_generated_skeleton_is_valid_and_counts_correctly() {
    let dims = TransformerDims {
        layers: 24,
        dim: 2048,
        heads: 16,
        seq: 1024,
        vocab: 32000,
        ffn: 8192,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let model = transformer(dims.clone()).expect("skeleton");
    let report = graph::validate(&model).expect("validated by the profile");
    assert_eq!(report.parameters, dims.parameters());
    assert_eq!(report.parameters, 1_339_031_552);
    // 23 nodes per layer: the original 22 plus the causal mask between scores and Softmax.
    assert_eq!(report.order.len(), 1 + 24 * 23 + 3);
    assert_eq!(report.shapes["l0.scores"], "[N, 16, 1024, 1024]");
    assert_eq!(report.shapes["l0.masked"], "[N, 16, 1024, 1024]");
    // Without the mask we get back exactly the original shape: same parameters, one node fewer
    // per layer, and the mask costs no parameters.
    let encoder = transformer(TransformerDims {
        causal: false,
        ..dims
    })
    .expect("bidirectional skeleton");
    let encoder_report = graph::validate(&encoder).expect("validated");
    assert_eq!(encoder_report.order.len(), 1 + 24 * 22 + 3);
    assert_eq!(encoder_report.parameters, report.parameters);
    assert!(!encoder_report.shapes.contains_key("l0.masked"));
    assert_eq!(report.shapes["logits"], "[N, 32000]");
    // Every parameter is assigned to exactly one node.
    let assigned: usize = report.nodes.iter().map(|n| n.parameter_names.len()).sum();
    assert_eq!(assigned, model.parameters.len());
    let bytes: u64 = report.nodes.iter().map(|n| n.parameter_bytes).sum();
    assert_eq!(bytes, report.parameter_bytes);
}

#[test]
fn inconsistent_dimensions_are_rejected() {
    let err = transformer(TransformerDims {
        layers: 1,
        dim: 100,
        heads: 3,
        seq: 8,
        vocab: 10,
        ffn: 16,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    })
    .expect_err("100 is not a multiple of 3");
    assert!(err.contains("multiple"));
}

#[test]
fn a_model_that_fits_uses_data_parallelism() {
    let report = graph::validate(&small_transformer()).unwrap();
    let p = plan::plan(&report, &home(300.0, 12), &work(10_000, 8, 10, 1)).expect("plan");
    assert_eq!(p.stages.len(), 1);
    assert!(p.cuts.is_empty());
    assert_eq!(p.pipelines, 10);
    assert_eq!(p.contributors_used, 10);
    assert_eq!(p.bound, Bound::Compute);
    // 1250 micro-batches × 1.25 for audits = 1563, spread over 10 machines.
    assert_eq!(p.micro_batches, 1563);
    assert!(p.notes.iter().any(|n| n.contains("fits on one machine")));
}

#[test]
fn a_model_too_large_is_cut_at_layer_boundaries() {
    let dims = TransformerDims {
        layers: 8,
        dim: 1024,
        heads: 8,
        seq: 256,
        vocab: 8000,
        ffn: 4096,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let report = graph::validate(&transformer(dims).unwrap()).unwrap();
    // 117 M parameters × 4 (Adam) ≈ 1.9 GiB: does not fit in 1 GiB.
    let p = plan::plan(&report, &home(300.0, 1), &work(10_000, 4, 12, 1)).expect("plan");
    assert!(p.stages.len() >= 2, "{} stage(s)", p.stages.len());
    for s in &p.stages {
        assert!(
            s.memory_bytes <= GIB,
            "stage {}..{} overflows",
            s.first,
            s.last
        );
    }
    // Stages are contiguous and cover everything.
    assert_eq!(p.stages[0].first, 0);
    assert_eq!(p.stages.last().unwrap().last, report.order.len() - 1);
    for w in p.stages.windows(2) {
        assert_eq!(w[0].last + 1, w[1].first);
    }
    assert_eq!(p.cuts.len(), p.stages.len() - 1);
    // The fallback picks a boundary where only the residual stream crosses: one tensor, not four.
    for c in &p.cuts {
        assert!(
            c.tensors.len() <= 2,
            "cut after '{}' carries {:?}",
            c.after,
            c.tensors
        );
        assert!(
            c.after.ends_with(".h2")
                || c.after.ends_with(".h1")
                || c.after == "h0"
                || c.after == "nf",
            "cut in the middle of a block: {}",
            c.after
        );
    }
    assert!(p.pipelines >= 1);
    assert_eq!(p.bound, Bound::Network, "10 Mbit/s cannot keep up");
}

#[test]
fn an_oversized_node_is_named() {
    let report = graph::validate(&small_transformer()).unwrap();
    let err = plan::plan(&report, &home(300.0, 0), &work(100, 8, 4, 1)).expect_err("0 GiB");
    match err {
        PlanError::NodeTooLarge { node, .. } => assert!(!node.is_empty()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn not_enough_contributors_for_the_stages() {
    let dims = TransformerDims {
        layers: 8,
        dim: 1024,
        heads: 8,
        seq: 256,
        vocab: 8000,
        ffn: 4096,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let report = graph::validate(&transformer(dims).unwrap()).unwrap();
    let err = plan::plan(&report, &home(300.0, 1), &work(10_000, 4, 1, 1)).expect_err("1 machine");
    match err {
        PlanError::NotEnoughContributors {
            stages,
            contributors,
        } => {
            assert!(stages >= 2);
            assert_eq!(contributors, 1);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn local_steps_reduce_synchronization() {
    let report = graph::validate(&small_transformer()).unwrap();
    let one = plan::plan(&report, &home(300.0, 12), &work(100_000, 8, 10, 1)).unwrap();
    let fifty = plan::plan(&report, &home(300.0, 12), &work(100_000, 8, 10, 50)).unwrap();
    assert!(fifty.sync_bytes_per_machine * 40 < one.sync_bytes_per_machine);
    assert!(fifty.wall_s < one.wall_s);
}

#[test]
fn a_fast_link_makes_compute_the_bottleneck() {
    let dims = TransformerDims {
        layers: 8,
        dim: 1024,
        heads: 8,
        seq: 256,
        vocab: 8000,
        ffn: 4096,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let report = graph::validate(&transformer(dims).unwrap()).unwrap();
    let fibre = Machine {
        uplink_bytes_per_s: 1e9 / 8.0,
        downlink_bytes_per_s: 1e9 / 8.0,
        ..home(300.0, 1)
    };
    let p = plan::plan(&report, &fibre, &work(10_000, 4, 12, 1)).unwrap();
    assert_eq!(p.bound, Bound::Compute);
    assert!(p.network_over_compute < 1.0);
}

#[test]
fn absurd_parameters_are_rejected() {
    let report = graph::validate(&small_transformer()).unwrap();
    assert!(matches!(
        plan::plan(&report, &home(300.0, 12), &work(0, 8, 4, 1)),
        Err(PlanError::Invalid(_))
    ));
    let mut w = work(100, 8, 4, 1);
    w.audit_rate = 1.5;
    assert!(matches!(
        plan::plan(&report, &home(300.0, 12), &w),
        Err(PlanError::Invalid(_))
    ));
    let mut m = home(300.0, 12);
    m.flops_per_s = f64::NAN;
    assert!(matches!(
        plan::plan(&report, &m, &work(100, 8, 4, 1)),
        Err(PlanError::Invalid(_))
    ));
}

#[test]
fn durations_and_bytes_are_human_readable() {
    assert_eq!(plan::human_duration(30.0), "30 s");
    assert_eq!(plan::human_duration(3600.0 * 5.0), "5.0 h");
    assert_eq!(plan::human_duration(86_400.0 * 30.0), "30.0 days");
    assert_eq!(plan::human_duration(86_400.0 * 365.25 * 2.0), "2.0 years");
    assert_eq!(plan::human_bytes(5 * GIB), "5.00 GiB");
}

/// Both topologies on the same graph, the same cut, the same machines: only the route of the
/// activations changes (ADR-0018). This is the number that decides between the two designs.
#[test]
fn the_star_routes_every_activation_through_the_coordinator() {
    use gg_compute::graph::plan::Topology;
    let dims = TransformerDims {
        layers: 8,
        dim: 1024,
        heads: 8,
        seq: 256,
        vocab: 8000,
        ffn: 4096,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let report = graph::validate(&transformer(dims).unwrap()).unwrap();
    let fibre = Machine {
        uplink_bytes_per_s: 300e6 / 8.0,
        downlink_bytes_per_s: 1e9 / 8.0,
        ..home(10_000.0, 1)
    };
    let chain = plan::plan(&report, &fibre, &work(100_000, 8, 12, 50)).expect("chain");
    let mut star_work = work(100_000, 8, 12, 50);
    star_work.topology = Topology::Coordinator;
    star_work.coordinator_link = Some((300e6 / 8.0, 1e9 / 8.0));
    let star = plan::plan(&report, &fibre, &star_work).expect("star");

    // Same partition, same stages: only the route differs.
    assert_eq!(chain.stages, star.stages);
    assert_eq!(chain.cuts.len(), star.cuts.len());
    assert!(chain.stages.len() >= 3, "{} stages", chain.stages.len());

    // In a chain, the coordinator carries no tensor.
    assert_eq!(chain.coordinator_bytes_per_micro_batch, 0);
    assert_eq!(chain.coordinator_s_per_micro_batch, 0.0);

    // In a star, each cut crosses the coordinator's link four times.
    let cut_bytes: u64 = star.cuts.iter().map(|c| c.bytes_per_example * 8).sum();
    assert_eq!(star.coordinator_bytes_per_micro_batch, cut_bytes * 4);
    assert!(star.coordinator_s_per_micro_batch > 0.0);

    // And total traffic doubles, because a cut byte crosses two links instead of one.
    assert!(star.total_network_bytes > chain.total_network_bytes);

    // The star is slower, and the plan says why and by how much.
    assert!(
        star.wall_s > chain.wall_s,
        "star {} s, chain {} s",
        star.wall_s,
        chain.wall_s
    );
    assert!(
        star.notes.iter().any(|n| n.contains("global ceiling")),
        "{:?}",
        star.notes
    );
    assert!(
        star.notes.iter().any(|n| n.contains("peer to peer")),
        "the plan must name the alternative: {:?}",
        star.notes
    );
}

#[test]
fn the_star_requires_the_coordinator_link() {
    use gg_compute::graph::plan::Topology;
    let report = graph::validate(&small_transformer()).unwrap();
    let mut w = work(1000, 8, 4, 1);
    w.topology = Topology::Coordinator;
    let err = plan::plan(&report, &home(300.0, 12), &w).expect_err("missing link");
    assert!(err.to_string().contains("coordinator_link"), "{err}");
    w.coordinator_link = Some((0.0, 1e9));
    assert!(matches!(
        plan::plan(&report, &home(300.0, 12), &w),
        Err(PlanError::Invalid(_))
    ));
}

/// A model that fits on one machine has no cut: the topology changes nothing, and that is what
/// makes data parallelism so valuable.
#[test]
fn without_a_cut_the_topology_costs_nothing() {
    use gg_compute::graph::plan::Topology;
    let report = graph::validate(&small_transformer()).unwrap();
    let mut w = work(10_000, 8, 10, 1);
    w.topology = Topology::Coordinator;
    w.coordinator_link = Some((10e6 / 8.0, 100e6 / 8.0));
    let p = plan::plan(&report, &home(300.0, 12), &w).expect("plan");
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.coordinator_bytes_per_micro_batch, 0);
    assert_eq!(p.coordinator_s_per_micro_batch, 0.0);
}

/// When a single node overflows, the plan does not stop: it says what would make it fit, and at
/// what cost (ADR-0019). The four levers are measured on the same node.
#[test]
fn an_oversized_node_comes_with_its_remedies() {
    let dims = TransformerDims {
        layers: 2,
        dim: 2048,
        heads: 16,
        seq: 512,
        vocab: 64_000,
        ffn: 8192,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let report = graph::validate(&transformer(dims).unwrap()).unwrap();
    // 64,000 × 2048 embedding parameters = 512 MiB, × 4 for Adam = 2 GiB: does not fit in 1.
    let err = plan::plan(&report, &home(10_000.0, 1), &work(10_000, 8, 64, 1))
        .expect_err("a node overflows");
    let PlanError::NodeTooLarge {
        node,
        needs,
        has,
        breakdown,
        remedies,
    } = err.clone()
    else {
        panic!("{err:?}");
    };
    assert!(!node.is_empty());
    assert!(needs > has);
    assert!(
        breakdown.contains("parameters") && breakdown.contains("activations"),
        "{breakdown}"
    );
    // The levers that act on this node are offered; those that change nothing are not.
    // Activation recomputation is one of the latter: on an isolated node, there is nothing it
    // could avoid keeping.
    let actions: Vec<&str> = remedies.iter().map(|r| r.action.as_str()).collect();
    assert!(
        actions.iter().any(|a| a.contains("micro-batch")),
        "{actions:?}"
    );
    assert!(
        actions.iter().any(|a| a.contains("adapters")),
        "{actions:?}"
    );
    assert!(
        actions.iter().any(|a| a.contains("host memory")),
        "{actions:?}"
    );
    assert!(
        !actions.iter().any(|a| a.contains("recompute")),
        "recomputation saves nothing on an isolated node, it must not be offered: {actions:?}"
    );
    // Each one really reduces memory, and the message makes them readable.
    for r in &remedies {
        assert!(r.memory_bytes < needs, "{} reduces nothing", r.action);
        assert!(!r.cost.is_empty());
    }
    // Offloading is enough here: the embedding has no activation, all its weight is parameters.
    let offload = remedies
        .iter()
        .find(|r| r.action.contains("host memory"))
        .unwrap();
    assert!(offload.sufficient, "{offload:?}");
    let text = err.to_string();
    assert!(
        text.contains("[enough]") || text.contains("[helps]"),
        "{text}"
    );
}

/// Offloading puts the parameters in host memory and only costs PCIe traffic: on a contributor's
/// machine that link is three orders of magnitude faster than the network, so almost nothing.
#[test]
fn offloading_makes_the_stage_fit_at_a_measured_pcie_cost() {
    let dims = TransformerDims {
        layers: 4,
        dim: 1024,
        heads: 8,
        seq: 256,
        vocab: 16_000,
        ffn: 4096,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let report = graph::validate(&transformer(dims).unwrap()).unwrap();
    let machine = Machine {
        uplink_bytes_per_s: 300e6 / 8.0,
        downlink_bytes_per_s: 1e9 / 8.0,
        ..home(10_000.0, 1)
    };
    let without =
        plan::plan(&report, &machine, &work(50_000, 8, 64, 10)).expect("without offloading");
    let mut w = work(50_000, 8, 64, 10);
    w.memory.host_offload_bytes = 32 * GIB;
    let with = plan::plan(&report, &machine, &w).expect("with offloading");

    // Fewer stages: each machine carries more parameters.
    assert!(
        with.stages.len() < without.stages.len(),
        "{} stages with offloading versus {} without",
        with.stages.len(),
        without.stages.len()
    );
    assert_eq!(without.host_bytes_per_machine, 0);
    assert!(with.host_bytes_per_machine > 0);
    // The PCIe transfer stays small compared with the step time.
    let stage = with.stages.iter().max_by_key(|s| s.host_bytes).unwrap();
    assert!(stage.host_transfer_s_per_batch > 0.0);
    assert!(
        stage.host_transfer_s_per_batch < stage.compute_s_per_batch,
        "PCIe {} s versus compute {} s",
        stage.host_transfer_s_per_batch,
        stage.compute_s_per_batch
    );
    assert!(
        with.notes.iter().any(|n| n.contains("offloading")),
        "{:?}",
        with.notes
    );
}

/// Recomputing activations and freezing the base both reduce memory, each at its own cost.
#[test]
fn recomputation_and_freezing_reduce_memory_at_a_cost() {
    let dims = TransformerDims {
        layers: 4,
        dim: 512,
        heads: 8,
        seq: 256,
        vocab: 8000,
        ffn: 2048,
        causal: true,
        positions: gg_compute::graph::templates::Positions::None,
    };
    let report = graph::validate(&transformer(dims).unwrap()).unwrap();
    let machine = home(10_000.0, 2);
    let base = plan::plan(&report, &machine, &work(20_000, 8, 32, 10)).expect("base");

    let mut w = work(20_000, 8, 32, 10);
    w.memory.recompute_activations = true;
    let recompute = plan::plan(&report, &machine, &w).expect("recomputation");
    assert!(recompute.stages.len() <= base.stages.len());
    // The extra compute is actually accounted for.
    let a: f64 = base.stages.iter().map(|s| s.compute_s_per_batch).sum();
    let b: f64 = recompute.stages.iter().map(|s| s.compute_s_per_batch).sum();
    assert!(b > a * 1.2, "recomputation {b} versus {a}");

    let mut w = work(20_000, 8, 32, 10);
    w.memory.trainable_fraction = 0.01;
    let lora = plan::plan(&report, &machine, &w).expect("adapters");
    assert!(lora.stages.len() <= base.stages.len());
    // Synchronization only covers what is trained.
    assert!(
        lora.sync_bytes_per_machine * 20 < base.sync_bytes_per_machine,
        "{} versus {}",
        lora.sync_bytes_per_machine,
        base.sync_bytes_per_machine
    );
    assert!(
        lora.notes.iter().any(|n| n.contains("memory policy")),
        "{:?}",
        lora.notes
    );
}

#[test]
fn an_absurd_trainable_fraction_is_rejected() {
    let report = graph::validate(&small_transformer()).unwrap();
    let mut w = work(1000, 8, 4, 1);
    w.memory.trainable_fraction = 1.5;
    assert!(matches!(
        plan::plan(&report, &home(300.0, 12), &w),
        Err(PlanError::Invalid(_))
    ));
}
