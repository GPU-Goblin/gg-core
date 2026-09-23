//! Plan for distributing a training run across contributor machines (ADR-0017).
//!
//! This module answers the question a researcher asks before submitting: "with N contributors and
//! their home connections, does my model train, and how long does it take?". The answer is
//! computed on the real graph: where to cut it so each piece fits in memory, how many bytes cross
//! each cut, and whether compute or the network sets the pace.
//!
//! The cost model is deliberately simple and its assumptions are written down in docs/MODELS.md.
//! It does not claim to predict a duration within 10 %; it distinguishes "feasible in weeks",
//! "feasible in years" and "impossible with these machines", which is the decision to be made.

use serde::{Deserialize, Serialize};

use super::GraphReport;

/// Memory factor of a parameter trained with Adam: value, gradient and two moments.
pub const OPTIMIZER_FACTOR: u64 = 4;

/// How a stage's memory is used. These are the four levers a researcher has when a stage does not
/// fit, from cheapest to most expensive (ADR-0019).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryPolicy {
    /// Fraction of the parameters that receive a gradient and optimizer state. `1.0` trains
    /// everything; `0.01` is the order of magnitude of low-rank adapters on a frozen base. The
    /// memory factor goes from 4 to `1 + 3·t`.
    pub trainable_fraction: f64,
    /// Recompute activations during the backward pass instead of keeping all of them. Costs
    /// about a third more compute and reduces a stage's activations to its input boundary, its
    /// output boundary and its largest node.
    pub recompute_activations: bool,
    /// Bytes of host memory to offload parameters and optimizer state to. The GPU then keeps only
    /// the working window of the largest node, plus the activations.
    pub host_offload_bytes: u64,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            trainable_fraction: 1.0,
            recompute_activations: false,
            host_offload_bytes: 0,
        }
    }
}

impl MemoryPolicy {
    /// Memory factor of a parameter: its value, plus gradient and moments if it is trained.
    pub fn factor(&self) -> f64 {
        1.0 + 3.0 * self.trainable_fraction.clamp(0.0, 1.0)
    }
    /// Compute overhead of activation recomputation.
    pub fn compute_overhead(&self) -> f64 {
        if self.recompute_activations {
            4.0 / 3.0
        } else {
            1.0
        }
    }
    /// The policy, readable in one line.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.trainable_fraction < 1.0 {
            parts.push(format!(
                "{:.1} % of parameters trained",
                self.trainable_fraction * 100.0
            ));
        }
        if self.recompute_activations {
            parts.push("activations recomputed".to_owned());
        }
        if self.host_offload_bytes > 0 {
            parts.push(format!(
                "offload to {} of host memory",
                human_bytes(self.host_offload_bytes)
            ));
        }
        if parts.is_empty() {
            "everything in GPU memory, all parameters trained".to_owned()
        } else {
            parts.join(", ")
        }
    }
}

/// The route activations take between two stages.
///
/// This choice decides feasibility far more than machine speed does: the same cut of the same
/// graph costs anywhere from a few seconds to several minutes per micro-batch depending on the
/// answer (ADR-0018).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    /// **Chain**: stage *i* sends its activations directly to stage *i+1*. Each cut has its own
    /// link, and the cuts work in parallel. The coordinator only schedules the work; tensors
    /// never pass through it.
    Peer,
    /// **Star**: every packet returns to the source, which emits the next one. Simple to
    /// reschedule and to resume after a failure, but **all** activations cross the coordinator's
    /// link, twice per cut and per direction.
    Coordinator,
}

/// A contributor machine and its link.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    /// Sustained executor throughput, in FLOP/s.
    pub flops_per_s: f64,
    /// Memory available for the model, its gradients, the optimizer and the activations.
    pub memory_bytes: u64,
    /// Upstream bandwidth, in bytes per second.
    pub uplink_bytes_per_s: f64,
    /// Downstream bandwidth, in bytes per second.
    pub downlink_bytes_per_s: f64,
    /// Host-GPU link bandwidth (PCIe), in bytes per second. Used to cost offloading: about
    /// 16 GB/s on PCIe 4.0 ×16, three orders of magnitude above an Internet connection.
    #[serde(default = "default_host_link")]
    pub host_link_bytes_per_s: f64,
}

fn default_host_link() -> f64 {
    16e9
}

/// The requested work.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    /// Total examples to process (dataset size × epochs).
    pub examples: u64,
    /// Examples per micro-batch.
    pub batch: u64,
    /// Contributors available at the same time.
    pub contributors: u64,
    /// Micro-batches computed locally between two parameter synchronizations.
    pub local_steps: u64,
    /// Fraction of micro-batches recomputed by a validator.
    pub audit_rate: f64,
    /// The route activations take.
    #[serde(default = "default_topology")]
    pub topology: Topology,
    /// Coordinator link, in bytes per second upstream and downstream. Only used for
    /// `Topology::Coordinator`; in a chain, the coordinator only exchanges orders.
    #[serde(default)]
    pub coordinator_link: Option<(f64, f64)>,
    /// How a stage's memory is used.
    #[serde(default)]
    pub memory: MemoryPolicy,
}

fn default_topology() -> Topology {
    Topology::Peer
}

/// A stage: a run of consecutive nodes assigned to the same machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    /// First node (index in execution order).
    pub first: usize,
    /// Last node, inclusive.
    pub last: usize,
    /// Number of nodes.
    pub nodes: usize,
    /// Parameter bytes of the stage.
    pub parameter_bytes: u64,
    /// Estimated memory for one micro-batch, optimizer included.
    pub memory_bytes: u64,
    /// Training FLOPs for one example.
    pub training_flops_per_example: u64,
    /// Compute seconds for one micro-batch, recomputation overhead included.
    pub compute_s_per_batch: f64,
    /// Bytes of parameters and optimizer state held in host memory (0 without offloading).
    pub host_bytes: u64,
    /// Host-GPU transfer seconds per micro-batch (0 without offloading).
    pub host_transfer_s_per_batch: f64,
}

/// A cut between two stages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cut {
    /// Last node before the cut.
    pub after: String,
    /// Tensors that cross it.
    pub tensors: Vec<String>,
    /// Bytes per example, in one direction.
    pub bytes_per_example: u64,
    /// Transfer seconds for one micro-batch, forward (activations) and back (gradients).
    pub transfer_s_per_batch: f64,
    /// FLOP per transferred byte, on the less loaded side of the cut.
    pub intensity_flop_per_byte: f64,
}

/// What sets the pace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// The executor.
    Compute,
    /// A link between stages.
    Network,
}

/// The computed plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    /// Stages.
    pub stages: Vec<Stage>,
    /// Cuts, one fewer than stages.
    pub cuts: Vec<Cut>,
    /// Parallel pipelines (contributors / stages).
    pub pipelines: u64,
    /// Contributors actually used.
    pub contributors_used: u64,
    /// Micro-batches to process, audit included.
    pub micro_batches: u64,
    /// Seconds per micro-batch in steady state, for one pipeline.
    pub s_per_micro_batch: f64,
    /// What sets this pace.
    pub bound: Bound,
    /// Ratio of network time to compute time on the slowest element.
    pub network_over_compute: f64,
    /// Bytes a machine exchanges to synchronize its parameters, in total.
    pub sync_bytes_per_machine: u64,
    /// Seconds a machine spends synchronizing, in total.
    pub sync_s_per_machine: f64,
    /// Estimated total duration, in seconds.
    pub wall_s: f64,
    /// Bytes that cross the Internet in total, cuts and synchronizations.
    pub total_network_bytes: u64,
    /// Memory policy applied.
    pub memory: MemoryPolicy,
    /// Bytes held in host memory by the most loaded machine (0 without offloading).
    pub host_bytes_per_machine: u64,
    /// Chosen topology.
    pub topology: Topology,
    /// Bytes that cross the coordinator's link for one micro-batch (star only).
    pub coordinator_bytes_per_micro_batch: u64,
    /// Seconds this takes it (star only); this is a global ceiling, not a per-pipeline one.
    pub coordinator_s_per_micro_batch: f64,
    /// What is worth telling the researcher.
    pub notes: Vec<String>,
}

/// Why no plan exists.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanError {
    /// A single node exceeds a machine's memory: no cut at node granularity can help, but
    /// `remedies` says what would make it fit, and at what price.
    NodeTooLarge {
        /// The node.
        node: String,
        /// What it needs.
        needs: u64,
        /// What the machine offers.
        has: u64,
        /// What is missing and why: parameters, activations, or both.
        breakdown: String,
        /// Computed remedies, from cheapest to most expensive.
        remedies: Vec<Remedy>,
    },
    /// The graph needs more stages than there are contributors.
    NotEnoughContributors {
        /// Stages needed.
        stages: usize,
        /// Contributors available.
        contributors: u64,
    },
    /// Nonsensical parameter.
    Invalid(String),
}

/// A way to make a stage fit, with what it costs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Remedy {
    /// What to do, in one sentence.
    pub action: String,
    /// Node memory once applied.
    pub memory_bytes: u64,
    /// Is it enough on its own?
    pub sufficient: bool,
    /// What it costs, in one sentence.
    pub cost: String,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::NodeTooLarge {
                node,
                needs,
                has,
                breakdown,
                remedies,
            } => {
                write!(
                    f,
                    "node '{node}' alone needs {}, the machine offers {} ({breakdown})",
                    human_bytes(*needs),
                    human_bytes(*has)
                )?;
                for r in remedies {
                    write!(
                        f,
                        "\n  {} {} → {} ({})",
                        if r.sufficient { "[enough]" } else { "[helps] " },
                        r.action,
                        human_bytes(r.memory_bytes),
                        r.cost
                    )?;
                }
                if remedies.iter().all(|r| !r.sufficient) {
                    write!(
                        f,
                        "\n  none of these levers is enough on its own: combine them, reduce the width \
of this layer, or split this single operator across several peers — the latter is not \
implemented and costs duplicating the input activation to every part (ADR-0019)"
                    )?;
                }
                Ok(())
            }
            PlanError::NotEnoughContributors {
                stages,
                contributors,
            } => write!(
                f,
                "the model must be cut into {stages} stages, so at least {stages} simultaneous \
contributors are needed, {contributors} are announced"
            ),
            PlanError::Invalid(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for PlanError {}

/// Computes the plan.
pub fn plan(report: &GraphReport, machine: &Machine, work: &Workload) -> Result<Plan, PlanError> {
    if work.batch == 0 || work.examples == 0 {
        return Err(PlanError::Invalid(
            "examples and batch must be strictly positive".into(),
        ));
    }
    if work.contributors == 0 {
        return Err(PlanError::Invalid("no contributors".into()));
    }
    if work.local_steps == 0 {
        return Err(PlanError::Invalid("local_steps must be at least 1".into()));
    }
    let positive = |x: f64| x.is_finite() && x > 0.0;
    if !positive(machine.flops_per_s)
        || !positive(machine.uplink_bytes_per_s)
        || !positive(machine.downlink_bytes_per_s)
    {
        return Err(PlanError::Invalid(
            "compute and link throughputs must be strictly positive".into(),
        ));
    }
    if !(0.0..=1.0).contains(&work.audit_rate) || work.audit_rate.is_nan() {
        return Err(PlanError::Invalid("audit_rate outside [0, 1]".into()));
    }
    let coordinator_link = match (work.topology, work.coordinator_link) {
        (Topology::Peer, _) => None,
        (Topology::Coordinator, Some((up, down))) => {
            let positive = |x: f64| x.is_finite() && x > 0.0;
            if !positive(up) || !positive(down) {
                return Err(PlanError::Invalid(
                    "coordinator_link expects strictly positive throughputs".into(),
                ));
            }
            Some((up, down))
        }
        (Topology::Coordinator, None) => {
            return Err(PlanError::Invalid(
                "the star topology requires coordinator_link (upstream, downstream)".into(),
            ));
        }
    };
    if report.nodes.is_empty() {
        return Err(PlanError::Invalid("graph has no nodes".into()));
    }

    let batch = work.batch;
    let link = machine.uplink_bytes_per_s.min(machine.downlink_bytes_per_s);

    // Bytes per example of a tensor that can cross a cut: a node output or a graph input.
    let bytes_of = |name: &str| -> u64 {
        report
            .nodes
            .iter()
            .find(|n| n.output == name)
            .map(|n| n.activation_bytes_per_example)
            .or_else(|| report.input_bytes_per_example.get(name).copied())
            .unwrap_or(0)
    };
    // Position of a tensor's producer; graph inputs and parameters have none.
    let producer =
        |name: &str| -> Option<usize> { report.nodes.iter().position(|n| n.output == name) };

    // Bytes that would cross a cut placed right after node i: every tensor produced before or at
    // i (or a graph input) and consumed after i. Independent of the stages.
    let n = report.nodes.len();
    let mut cross_after: Vec<u64> = vec![0; n];
    // A tensor crosses every cut between its producer and its last consumer, and it is counted
    // only once even if it has several consumers (the residual stream has two).
    let mut span: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for (j, node) in report.nodes.iter().enumerate() {
        for input in &node.inputs {
            let from = match producer(input) {
                Some(p) => p,
                None if report.input_bytes_per_example.contains_key(input) => 0,
                None => continue, // parameter: it lives where it is used
            };
            span.entry(input.as_str())
                .and_modify(|(_, last)| *last = (*last).max(j))
                .or_insert((from, j));
        }
    }
    for (name, (from, last)) in span {
        let b = bytes_of(name);
        // The cut after i (from ≤ i < last) is crossed by this tensor.
        for slot in cross_after.iter_mut().take(last).skip(from) {
            *slot = slot.saturating_add(b);
        }
    }

    let policy = work.memory;
    if !(0.0..=1.0).contains(&policy.trainable_fraction) || policy.trainable_fraction.is_nan() {
        return Err(PlanError::Invalid(
            "trainable_fraction outside [0, 1]".into(),
        ));
    }
    if !(machine.host_link_bytes_per_s.is_finite() && machine.host_link_bytes_per_s > 0.0) {
        return Err(PlanError::Invalid(
            "host_link_bytes_per_s must be strictly positive".into(),
        ));
    }
    let factor = policy.factor();
    let scaled = |bytes: u64| -> u64 { (bytes as f64 * factor) as u64 };

    // Memory of a range of nodes, under the chosen policy.
    //
    // Approximations, written down to be challenged: with recomputation, a stage keeps only its
    // input boundary, its output boundary and the largest node of the range (the working window
    // of the backward pass); with offloading, the GPU keeps only the parameters of the largest
    // node, the rest living in host memory.
    let interval = |first: usize, last: usize| -> (u64, u64, u64, u64, u64) {
        let slice = &report.nodes[first..=last];
        let params: u64 = slice.iter().map(|x| x.parameter_bytes).sum();
        let max_param = slice.iter().map(|x| x.parameter_bytes).max().unwrap_or(0);
        let all_act: u64 = slice.iter().map(|x| x.activation_bytes_per_example).sum();
        let max_act = slice
            .iter()
            .map(|x| x.activation_bytes_per_example)
            .max()
            .unwrap_or(0);
        let incoming = if first == 0 {
            0
        } else {
            cross_after[first - 1]
        };
        let outgoing = cross_after[last];
        let kept = if policy.recompute_activations {
            incoming.saturating_add(outgoing).saturating_add(max_act)
        } else {
            all_act.saturating_add(incoming)
        };
        (params, max_param, kept, max_act, incoming)
    };
    let make_stage = |first: usize, last: usize| -> Stage {
        let (params, max_param, kept_act, _, _) = interval(first, last);
        let slice = &report.nodes[first..=last];
        let flops: u64 = slice.iter().map(|x| x.training_flops_per_example).sum();
        let (gpu_params, host_bytes) = if policy.host_offload_bytes > 0 {
            (scaled(max_param), scaled(params))
        } else {
            (scaled(params), 0)
        };
        // Offloaded parameters cross the host link on the way in and out of every micro-batch.
        let host_transfer_s_per_batch = if host_bytes > 0 {
            2.0 * host_bytes as f64 / machine.host_link_bytes_per_s
        } else {
            0.0
        };
        Stage {
            first,
            last,
            nodes: last - first + 1,
            parameter_bytes: params,
            memory_bytes: gpu_params.saturating_add(kept_act.saturating_mul(batch)),
            training_flops_per_example: flops,
            compute_s_per_batch: (flops as f64) * (batch as f64) * policy.compute_overhead()
                / machine.flops_per_s
                + host_transfer_s_per_batch,
            host_bytes,
            host_transfer_s_per_batch,
        }
    };

    // When a single node overflows, say what would make it fit rather than stopping there.
    let remedies_for = |i: usize| -> (String, Vec<Remedy>) {
        let node = &report.nodes[i];
        let (params, _, _, max_act, incoming) = interval(i, i);
        let act_now = if policy.recompute_activations {
            incoming
                .saturating_add(cross_after[i])
                .saturating_add(max_act)
        } else {
            node.activation_bytes_per_example.saturating_add(incoming)
        }
        .saturating_mul(batch);
        let par_now = if policy.host_offload_bytes > 0 {
            0
        } else {
            scaled(params)
        };
        let breakdown = format!(
            "{} of parameters and optimizer state, {} of activations for a batch of {batch}",
            human_bytes(par_now),
            human_bytes(act_now)
        );
        let has = machine.memory_bytes;
        let needs = par_now.saturating_add(act_now);
        let mut out = Vec::new();
        // A lever is only offered if it actually reduces this node's memory. Activation
        // recomputation, for example, does not help on an isolated node: it has nothing to drop.
        let mut add = |action: String, mem: u64, cost: String| {
            if mem < needs {
                out.push(Remedy {
                    action,
                    memory_bytes: mem,
                    sufficient: mem <= has,
                    cost,
                });
            }
        };
        // 1. Smaller micro-batch: only affects activations.
        if batch > 1 && act_now > 0 {
            let per_example = act_now / batch.max(1);
            add(
                "divide the micro-batch by 4".to_owned(),
                par_now.saturating_add(per_example.saturating_mul(batch.div_ceil(4))),
                "more micro-batches, hence more synchronizations".to_owned(),
            );
        }
        // 2. Activation recomputation.
        if !policy.recompute_activations {
            let kept = incoming
                .saturating_add(cross_after[i])
                .saturating_add(max_act)
                .saturating_mul(batch);
            add(
                "recompute activations (--recompute)".to_owned(),
                par_now.saturating_add(kept),
                "about a third more compute".to_owned(),
            );
        }
        // 3. Freeze the base, train only adapters.
        if policy.trainable_fraction > 0.02 {
            let lora = (params as f64 * (1.0 + 3.0 * 0.01)) as u64;
            add(
                "train only adapters (--trainable-fraction 0.01)".to_owned(),
                lora.saturating_add(act_now),
                "the base model is no longer updated".to_owned(),
            );
        }
        // 4. Offloading to host memory.
        if policy.host_offload_bytes == 0 {
            let host = scaled(params);
            let s = 2.0 * host as f64 / machine.host_link_bytes_per_s;
            add(
                format!(
                    "offload to host memory (--host-offload-gib, {} needed)",
                    human_bytes(host)
                ),
                act_now,
                format!("{s:.2} s of PCIe transfer per micro-batch"),
            );
        }
        (breakdown, out)
    };

    // Greedy cutting with backoff: a stage grows as long as it fits in memory; when it overflows,
    // the cut is not placed where it overflows but at the stage's least loaded boundary
    // (typically between two layers, where only the residual stream passes). The number of stages
    // stays close to the minimum and the cuts avoid the core of attention blocks.
    let mut stages: Vec<Stage> = Vec::new();
    let mut first = 0usize;
    while first < n {
        let mut last = first;
        loop {
            let candidate = make_stage(first, last);
            if candidate.memory_bytes > machine.memory_bytes {
                if last == first {
                    let (breakdown, remedies) = remedies_for(first);
                    return Err(PlanError::NodeTooLarge {
                        node: report.nodes[first].output.clone(),
                        needs: candidate.memory_bytes,
                        has: machine.memory_bytes,
                        breakdown,
                        remedies,
                    });
                }
                // Backoff: best boundary in [first, last-1], the latest one on a tie.
                let best = (first..last)
                    .rev()
                    .min_by_key(|&j| cross_after[j])
                    .expect("non-empty range");
                last = best;
                break;
            }
            if last + 1 == n {
                break;
            }
            last += 1;
        }
        stages.push(make_stage(first, last));
        first = last + 1;
    }

    let mut cuts: Vec<Cut> = Vec::with_capacity(stages.len().saturating_sub(1));
    for window in stages.windows(2) {
        let (before, after) = (&window[0], &window[1]);
        let mut tensors: Vec<String> = Vec::new();
        for node in &report.nodes[after.first..] {
            for input in &node.inputs {
                let crosses = match producer(input) {
                    Some(p) => p <= before.last,
                    None => report.input_bytes_per_example.contains_key(input),
                };
                if crosses && !tensors.contains(input) {
                    tensors.push(input.clone());
                }
            }
        }
        let bytes_per_example: u64 = tensors.iter().map(|t| bytes_of(t)).sum();
        debug_assert_eq!(bytes_per_example, cross_after[before.last]);
        let transfer_s_per_batch = 2.0 * (bytes_per_example as f64) * (batch as f64) / link;
        let lighter = before
            .training_flops_per_example
            .min(after.training_flops_per_example) as f64;
        let intensity_flop_per_byte = if bytes_per_example == 0 {
            f64::INFINITY
        } else {
            lighter / bytes_per_example as f64
        };
        cuts.push(Cut {
            after: report.nodes[before.last].output.clone(),
            tensors,
            bytes_per_example,
            transfer_s_per_batch,
            intensity_flop_per_byte,
        });
    }

    let stage_count = stages.len();
    let pipelines = work.contributors / stage_count as u64;
    if pipelines == 0 {
        return Err(PlanError::NotEnoughContributors {
            stages: stage_count,
            contributors: work.contributors,
        });
    }

    let slowest_compute = stages
        .iter()
        .map(|s| s.compute_s_per_batch)
        .fold(0.0_f64, f64::max);
    let slowest_link = cuts
        .iter()
        .map(|c| c.transfer_s_per_batch)
        .fold(0.0_f64, f64::max);

    // In a star, every tensor that crosses a cut passes through the coordinator twice (it
    // receives it from one stage and sends it to the next), forward as well as backward: four
    // crossings per cut and per micro-batch. And all cuts of all pipelines share this link.
    let cut_bytes_per_batch: u64 = cuts
        .iter()
        .map(|c| c.bytes_per_example.saturating_mul(batch))
        .sum();
    let (coordinator_bytes_per_micro_batch, coordinator_s_per_micro_batch) = match coordinator_link
    {
        None => (0, 0.0),
        Some((up, down)) => {
            let bytes = cut_bytes_per_batch.saturating_mul(4);
            // Half upstream, half downstream; the slower one dominates.
            let half = cut_bytes_per_batch as f64 * 2.0;
            (bytes, (half / up).max(half / down))
        }
    };

    let (s_per_micro_batch, bound) = if slowest_link > slowest_compute {
        (slowest_link, Bound::Network)
    } else {
        (slowest_compute, Bound::Compute)
    };
    let network_over_compute = if slowest_compute > 0.0 {
        slowest_link / slowest_compute
    } else {
        f64::INFINITY
    };

    let micro_batches = work.examples.div_ceil(batch);
    let audited = ((micro_batches as f64) * (1.0 + work.audit_rate)).ceil() as u64;
    let per_pipeline = audited.div_ceil(pipelines);

    // Synchronization: every machine sends and receives the parameters of its stage every
    // `local_steps` micro-batches. The heaviest stage sets the cost.
    // Only trained parameters are exchanged: freezing the base reduces synchronization accordingly.
    let heaviest_stage = (stages.iter().map(|s| s.parameter_bytes).max().unwrap_or(0) as f64
        * policy.trainable_fraction) as u64;
    let syncs = per_pipeline.div_ceil(work.local_steps);
    let sync_bytes_per_machine = heaviest_stage.saturating_mul(2).saturating_mul(syncs);
    let sync_s_per_machine = sync_bytes_per_machine as f64 / link;

    let fill = (stage_count as f64 - 1.0) * s_per_micro_batch;
    let wall_s = match work.topology {
        // In a chain, pipelines advance in parallel: the duration is that of one pipeline.
        Topology::Peer => per_pipeline as f64 * s_per_micro_batch + sync_s_per_machine + fill,
        // In a star, the coordinator's link is a global ceiling: every micro-batch crosses it,
        // whatever the number of pipelines and the speed of the machines.
        Topology::Coordinator => {
            let by_coordinator = audited as f64 * coordinator_s_per_micro_batch;
            let by_pipelines = per_pipeline as f64 * slowest_compute + fill;
            by_coordinator.max(by_pipelines) + sync_s_per_machine
        }
    };

    let cut_bytes_total: u64 = cuts
        .iter()
        .map(|c| c.bytes_per_example.saturating_mul(2).saturating_mul(batch))
        .sum::<u64>()
        .saturating_mul(audited);
    let sync_total: u64 = stages
        .iter()
        .map(|s| {
            ((s.parameter_bytes as f64 * policy.trainable_fraction) as u64)
                .saturating_mul(2)
                .saturating_mul(syncs)
        })
        .sum::<u64>()
        .saturating_mul(pipelines);
    let total_network_bytes = match work.topology {
        Topology::Peer => cut_bytes_total.saturating_add(sync_total),
        // Every cut byte crosses two links instead of one.
        Topology::Coordinator => cut_bytes_total.saturating_mul(2).saturating_add(sync_total),
    };

    let host_bytes_per_machine = stages.iter().map(|s| s.host_bytes).max().unwrap_or(0);
    let mut notes = Vec::new();
    if policy != MemoryPolicy::default() {
        notes.push(format!("memory policy: {}", policy.describe()));
    }
    if let Some(worst) = stages.iter().max_by_key(|s| s.host_bytes)
        && worst.host_bytes > 0
    {
        if worst.host_bytes > policy.host_offload_bytes {
            notes.push(format!(
                "the heaviest stage offloads {} to host memory, the machine only announces {}: \
offloading does not fit either",
                human_bytes(worst.host_bytes),
                human_bytes(policy.host_offload_bytes)
            ));
        } else {
            notes.push(format!(
                "offloading: {} in host memory, {:.2} s of PCIe per micro-batch, i.e. {:.0} % of the pace",
                human_bytes(worst.host_bytes),
                worst.host_transfer_s_per_batch,
                100.0 * worst.host_transfer_s_per_batch
                    / worst.compute_s_per_batch.max(f64::MIN_POSITIVE)
            ));
        }
    }
    if stage_count == 1 {
        notes.push(
            "the model fits on one machine: pure data parallelism, no cut, \
each contributor computes whole micro-batches"
                .to_owned(),
        );
    } else {
        notes.push(format!(
            "the model does not fit on one machine: cut into {stage_count} stages, \
{pipelines} pipeline(s) in parallel; a failure in a stage stops its pipeline"
        ));
    }
    if work.topology == Topology::Coordinator && stage_count > 1 {
        notes.push(format!(
            "star topology: {} per micro-batch cross the coordinator's link ({:.1} s), \
a global ceiling that neither the number of contributors nor their speed raises",
            human_bytes(coordinator_bytes_per_micro_batch),
            coordinator_s_per_micro_batch
        ));
        if coordinator_s_per_micro_batch > slowest_link && slowest_link > 0.0 {
            notes.push(format!(
                "the same cut peer to peer would cost {:.1} s per micro-batch instead of {:.1}: \
pass activations between stages, and keep only orders at the coordinator",
                slowest_link, coordinator_s_per_micro_batch
            ));
        }
    }
    match bound {
        Bound::Network => notes.push(format!(
            "the link sets the pace: {:.0} times slower than compute on the most loaded cut; \
enlarge the stages (more memory) or reduce the width of the activations",
            network_over_compute
        )),
        Bound::Compute => notes.push(format!(
            "the executor sets the pace; the network takes only {:.0} % of it",
            network_over_compute * 100.0
        )),
    }
    let required_intensity = machine.flops_per_s / link;
    if let Some(worst) = cuts.iter().min_by(|a, b| {
        a.intensity_flop_per_byte
            .total_cmp(&b.intensity_flop_per_byte)
    }) {
        notes.push(format!(
            "weakest cut after '{}': {:.0} FLOP per transferred byte, {:.0} are needed \
for this link to keep up with this executor",
            worst.after, worst.intensity_flop_per_byte, required_intensity
        ));
    }
    if sync_s_per_machine > 0.1 * wall_s {
        notes.push(format!(
            "parameter synchronization takes {:.0} % of the time: increase local_steps or \
train only adapters",
            100.0 * sync_s_per_machine / wall_s
        ));
    }
    if work.contributors > pipelines * stage_count as u64 {
        notes.push(format!(
            "{} contributor(s) unused: the number of machines is not a multiple of \
{stage_count}",
            work.contributors - pipelines * stage_count as u64
        ));
    }

    Ok(Plan {
        stages,
        cuts,
        pipelines,
        contributors_used: pipelines * stage_count as u64,
        micro_batches: audited,
        s_per_micro_batch,
        bound,
        network_over_compute,
        sync_bytes_per_machine,
        sync_s_per_machine,
        wall_s,
        total_network_bytes,
        memory: policy,
        host_bytes_per_machine,
        topology: work.topology,
        coordinator_bytes_per_micro_batch,
        coordinator_s_per_micro_batch,
        notes,
    })
}

/// Time a single accelerator at the given throughput would take for the same work, without a
/// network. An honest point of comparison: a network of contributors is never free in time.
pub fn single_accelerator_s(report: &GraphReport, work: &Workload, flops_per_s: f64) -> f64 {
    let flops = report.training_flops_per_example as f64 * work.examples as f64;
    flops / flops_per_s
}

/// Human-readable duration.
pub fn human_duration(s: f64) -> String {
    if !s.is_finite() {
        return "∞".to_owned();
    }
    const MIN: f64 = 60.0;
    const HOUR: f64 = 3600.0;
    const DAY: f64 = 86_400.0;
    const YEAR: f64 = 365.25 * DAY;
    if s < MIN {
        format!("{s:.0} s")
    } else if s < HOUR {
        format!("{:.0} min", s / MIN)
    } else if s < 2.0 * DAY {
        format!("{:.1} h", s / HOUR)
    } else if s < YEAR {
        format!("{:.1} days", s / DAY)
    } else {
        format!("{:.1} years", s / YEAR)
    }
}

/// Human-readable bytes.
pub fn human_bytes(b: u64) -> String {
    const KIB: f64 = 1024.0;
    let b = b as f64;
    if b < KIB {
        format!("{b:.0} B")
    } else if b < KIB * KIB {
        format!("{:.1} KiB", b / KIB)
    } else if b < KIB * KIB * KIB {
        format!("{:.1} MiB", b / (KIB * KIB))
    } else if b < KIB * KIB * KIB * KIB {
        format!("{:.2} GiB", b / (KIB * KIB * KIB))
    } else if b < KIB * KIB * KIB * KIB * KIB {
        format!("{:.2} TiB", b / (KIB * KIB * KIB * KIB))
    } else {
        format!("{:.2} PiB", b / (KIB * KIB * KIB * KIB * KIB))
    }
}
