//! Working memory of a `graph.grad.v1` task on the accelerator, and the required profile version.
//!
//! This is what lets the control plane offer a task only to a worker able to hold it
//! (gg-server/docs/ROADMAP.md, step C1), and lets a worker cleanly refuse a task too large for the memory it
//! declares. The estimate follows **the same split** as the executor
//! ([`super::exec::gradient_with`]): same segment boundaries, same release of tensors after their
//! last use, same parameters loaded segment by segment with offloading. It is an upper bound on
//! what the executor measures (`peak_activation_bytes + peak_parameter_bytes`), checked by
//! `tests/graph_memory_estimate.rs`.
//!
//! This is not the resident memory of the process: the engine's internal buffers (autodiff tape,
//! kernel workspaces) come on top of it. A contributor therefore declares memory with a margin,
//! and docs/MODELS.md says how much.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::exec::ExecOptions;
use super::{GraphReport, Init, ModelGraphV1};

/// Level of a published profile: `gg-ops/N` → `N`.
pub fn profile_level(version: &str) -> Option<u32> {
    version
        .strip_prefix("gg-ops/")?
        .parse()
        .ok()
        .filter(|n| *n >= 1)
}

/// Profile level of this worker.
pub fn own_profile_level() -> u32 {
    profile_level(super::PROFILE_VERSION).expect("PROFILE_VERSION is well-formed")
}

/// Lowest profile version able to execute this description. A worker on an earlier version would
/// refuse it, naming what it lacks; the server therefore does not offer it to that worker.
pub fn required_profile(graph: &ModelGraphV1) -> u32 {
    let mut level = 1;
    for p in &graph.parameters {
        let l = match &p.init {
            // ADR-0022: explicit frequencies.
            Init::RotaryCos(r) | Init::RotarySin(r) if r.frequencies.is_some() => 4,
            // ADR-0021: generated position tables.
            Init::RotaryCos(_) | Init::RotarySin(_) | Init::Sinusoidal { .. } => 3,
            // ADR-0020: frozen constants.
            Init::Constant { .. } => 2,
            _ => 1,
        };
        level = level.max(l);
        if !p.trainable {
            level = level.max(2);
        }
    }
    let ints: std::collections::BTreeSet<&str> = graph
        .inputs
        .iter()
        .filter(|i| i.dtype == "i64")
        .map(|i| i.name.as_str())
        .collect();
    let rank_of = |t: &str| -> usize {
        match graph.parameters.iter().find(|p| p.name == t) {
            Some(p) => p.shape.len(),
            // A computed table: its rank from validation (rare; it also validates the rest).
            None => super::validate(graph)
                .ok()
                .and_then(|r| r.shapes.get(t).map(|s| s.matches(',').count() + 1))
                .unwrap_or(2),
        }
    };
    for n in &graph.nodes {
        // ADR-0029: integer tensors computed in the graph, tables of rank 3 and 4, i64 inputs
        // outside `Gather`.
        let integer = match n.op.as_str() {
            "ArgMax" | "ArgMin" | "ArgTopK" | "ArgSort" | "Cast" | "GatherElements"
            | "ScatterElements" => true,
            "Gather" => rank_of(&n.inputs[0]) > 2,
            // The tanh form of Gelu was accepted and computed as the exact one before gg-ops/8:
            // an earlier worker would compute another function.
            "Gelu" => {
                matches!(n.attrs.get("approximate"), Some(super::Attr::Text(a)) if a == "tanh")
            }
            _ => n.inputs.iter().any(|i| ints.contains(i.as_str())),
        };
        if integer {
            level = level.max(8);
            continue;
        }
        let l = match n.op.as_str() {
            "ConvTranspose2d" | "Pad" | "Resize" | "Flip" | "ReduceSum" | "ReduceMax"
            | "ReduceMin" | "ReduceProd" | "CumSum" | "IndexSelect" | "IndexAdd" => 7,
            // ADR-0027: functions added to the grammar.
            "Formula"
                if matches!(n.attrs.get("expr"), Some(super::Attr::Text(e))
                    if super::formula::parse(e).is_ok_and(|x| x.needs_profile_7())) =>
            {
                7
            }
            "Formula" => 6,
            "ReduceMean" | "Rsqrt" => 5,
            "Slice" => 3,
            "CausalMask" => 2,
            _ => 1,
        };
        level = level.max(l);
    }
    level
}

/// Estimated working memory of a task, in bytes, item by item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskMemory {
    /// Parameters loaded at the peak: all of them, or the largest segment with offloading.
    pub parameters: u64,
    /// Gradients of the trained parameters, accumulated until the end.
    pub gradients: u64,
    /// Activations at the peak: forward pass (tensors still needed) or backward pass of one
    /// segment (its outputs, its parameters, and the saved boundaries).
    pub activations: u64,
    /// Inputs and target of the micro-batch.
    pub datasets: u64,
}

impl TaskMemory {
    /// Total.
    pub fn total(&self) -> u64 {
        self.parameters
            .saturating_add(self.gradients)
            .saturating_add(self.activations)
            .saturating_add(self.datasets)
    }
}

/// Estimates the working memory of a gradient step over `batch` examples with these levers.
pub fn task_memory(
    graph: &ModelGraphV1,
    report: &GraphReport,
    batch: u64,
    options: ExecOptions,
) -> TaskMemory {
    let order = &report.order;
    let n = order.len();
    let node_by_output: BTreeMap<&str, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.output.as_str(), i))
        .collect();
    let param_bytes: BTreeMap<&str, u64> = graph
        .parameters
        .iter()
        .map(|p| (p.name.as_str(), p.shape.iter().product::<u64>() * 4))
        .collect();
    let act: BTreeMap<&str, u64> = report
        .nodes
        .iter()
        .map(|c| {
            (
                c.output.as_str(),
                c.activation_bytes_per_example.saturating_mul(batch),
            )
        })
        .collect();
    let datasets = report
        .input_bytes_per_example
        .values()
        .sum::<u64>()
        .saturating_add({
            let t = graph.loss.target();
            let elems: u64 = t.shape.iter().map(|d| (*d).max(1)).product();
            elems * if t.dtype == "i64" { 8 } else { 4 }
        })
        .saturating_mul(batch);
    let inputs_of = |name: &str| -> &[String] { &graph.nodes[node_by_output[name]].inputs };

    let segments = options.segments.clamp(1, n.max(1));
    let bounds: Vec<(usize, usize)> = (0..segments)
        .map(|s| (n * s / segments, n * (s + 1) / segments))
        .filter(|(a, b)| a < b)
        .collect();

    // Parameters of each segment.
    let segment_params = |(a, b): (usize, usize)| -> u64 {
        let mut seen = BTreeSet::new();
        for name in &order[a..b] {
            for i in inputs_of(name) {
                if param_bytes.contains_key(i.as_str()) {
                    seen.insert(i.as_str());
                }
            }
        }
        seen.iter().map(|p| param_bytes[p]).sum()
    };
    let all_params: u64 = param_bytes.values().sum();
    let parameters = if options.offload_parameters {
        bounds.iter().map(|r| segment_params(*r)).max().unwrap_or(0)
    } else {
        all_params
    };
    let gradients = graph
        .parameters
        .iter()
        .filter(|p| p.trainable)
        .map(|p| param_bytes[p.name.as_str()])
        .sum();

    // Last use of each tensor, as in the executor.
    let mut last_use: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, name) in order.iter().enumerate() {
        for input in inputs_of(name) {
            last_use.insert(input.as_str(), i);
        }
    }
    last_use.insert(graph.loss.prediction(), n);

    // Saved boundaries: what each segment consumes that comes from before it.
    let mut saved: BTreeSet<&str> = BTreeSet::new();
    for (a, b) in &bounds {
        for name in &order[*a..*b] {
            for input in inputs_of(name) {
                if act.contains_key(input.as_str()) && !order[*a..*b].contains(input) {
                    saved.insert(input.as_str());
                }
            }
        }
    }
    let saved_bytes: u64 = if segments > 1 {
        saved.iter().map(|t| act[t]).sum()
    } else {
        0
    };

    // Forward pass: tensors live at the peak.
    let mut live: BTreeMap<&str, u64> = BTreeMap::new();
    let mut forward_peak = 0u64;
    for (i, name) in order.iter().enumerate() {
        live.insert(name.as_str(), act[name.as_str()]);
        live.retain(|k, _| last_use.get(k).is_some_and(|l| *l > i));
        forward_peak = forward_peak.max(live.values().sum::<u64>());
    }
    // Backward pass, one segment at a time: its outputs, its parameters (loaded into its
    // environment), its inputs from earlier segments (copied from the boundaries) and the
    // micro-batch. Without segments, it is the whole graph at once.
    let segment_inputs = |(a, b): (usize, usize)| -> u64 {
        let mut seen = BTreeSet::new();
        for name in &order[a..b] {
            for i in inputs_of(name) {
                if act.contains_key(i.as_str()) && !order[a..b].contains(i) {
                    seen.insert(i.as_str());
                }
            }
        }
        seen.iter().map(|t| act[t]).sum()
    };
    let backward_peak = bounds
        .iter()
        .map(|(a, b)| {
            order[*a..*b].iter().map(|o| act[o.as_str()]).sum::<u64>()
                + segment_params((*a, *b))
                + segment_inputs((*a, *b))
                + datasets
        })
        .max()
        .unwrap_or(0);
    let activations = forward_peak
        .saturating_add(saved_bytes)
        .max(backward_peak.saturating_add(saved_bytes));

    TaskMemory {
        parameters,
        gradients,
        activations,
        datasets,
    }
}

/// Cheapest levers that fit a task of `batch` examples into `capacity` bytes (C4): first without
/// offloading, with more and more segments (each segment costs a recomputation), then with
/// offloading. `None` if nothing fits: a smaller micro-batch is then needed, or staged execution
/// (gg-server/docs/ROADMAP.md, D2).
pub fn recommend(
    graph: &ModelGraphV1,
    report: &GraphReport,
    batch: u64,
    capacity: u64,
) -> Option<(ExecOptions, TaskMemory)> {
    let nodes = report.order.len().max(1);
    let mut segments = Vec::new();
    let mut s = 1usize;
    while s < nodes {
        segments.push(s);
        s *= 2;
    }
    segments.push(nodes);
    for offload_parameters in [false, true] {
        for &segments in &segments {
            let options = ExecOptions {
                segments,
                offload_parameters,
            };
            let m = task_memory(graph, report, batch, options);
            if m.total() <= capacity {
                return Some((options, m));
            }
        }
    }
    None
}
