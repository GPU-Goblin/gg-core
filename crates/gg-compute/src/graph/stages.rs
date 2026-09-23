//! Stages of a model split across several machines (ADR-0018, gg-server/docs/ROADMAP.md P11 step D2).
//!
//! A stage is a **contiguous** run of nodes in execution order. It holds only its own parameters
//! and activations: this is what makes it possible to train a model that no single machine can
//! hold. Its **inputs** are the tensors it consumes without producing them (datasets, or outputs of
//! an earlier stage, not necessarily the previous one: a residual connection can skip over a
//! stage); its **outputs** are the tensors it produces that a later stage consumes, or that the
//! loss reads. The last stage computes the loss, and therefore receives the target.
//!
//! The bounds are indices in execution order, `0 = b₀ < b₁ < … < b_K = n`. They travel in the
//! task; everything else — parameters, inputs, outputs, checkpoint shards — is derived from them
//! deterministically, by the server, the worker and the validator each.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{GraphReport, ModelGraphV1};

/// Maximum number of stages.
pub const MAX_STAGES: usize = 32;

/// A stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageSpec {
    /// Stage index.
    pub index: usize,
    /// First node, in execution order.
    pub first: usize,
    /// Exclusive end.
    pub end: usize,
    /// Parameters consumed, in order of first use.
    pub params: Vec<String>,
    /// Tensors consumed without being produced here, in order of first use; for the last stage,
    /// the loss target comes last.
    pub inputs: Vec<String>,
    /// Tensors produced here and read afterwards (by another stage, or by the loss).
    pub outputs: Vec<String>,
    /// True for the last stage: it computes the loss.
    pub last: bool,
}

/// The stages of a description, for the given bounds. A bound that an integer tensor would cross
/// is refused (ADR-0029).
pub fn stages(
    graph: &ModelGraphV1,
    report: &GraphReport,
    bounds: &[u32],
) -> Result<Vec<StageSpec>, String> {
    let out = stages_unchecked(graph, report, bounds)?;
    for st in &out {
        // An integer tensor stays inside its task (ADR-0029): tasks exchange floats, compared
        // within a tolerance, and an index is either right or wrong.
        if let Some(t) = st
            .outputs
            .iter()
            .find(|o| report.integer_tensors.contains_key(*o))
        {
            return Err(format!(
                "stage bound {} cuts through the integer tensor '{t}': an integer tensor does not \
cross a stage boundary (ADR-0029); move the bound after its last use",
                st.end
            ));
        }
    }
    Ok(out)
}

fn stages_unchecked(
    graph: &ModelGraphV1,
    report: &GraphReport,
    bounds: &[u32],
) -> Result<Vec<StageSpec>, String> {
    let n = report.order.len();
    if bounds.len() < 2
        || bounds[0] != 0
        || *bounds.last().expect("non-empty") as usize != n
        || bounds.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(format!(
            "invalid stage bounds {bounds:?}: expected 0 = b₀ < … < b_K = {n}"
        ));
    }
    if bounds.len() - 1 > MAX_STAGES {
        return Err(format!(
            "{} stages, the maximum is {MAX_STAGES}",
            bounds.len() - 1
        ));
    }
    let index = Index::new(graph, report);
    let k = bounds.len() - 1;
    Ok((0..k)
        .map(|s| {
            index.stage(
                graph,
                report,
                s,
                bounds[s] as usize,
                bounds[s + 1] as usize,
                s + 1 == k,
            )
        })
        .collect())
}

/// What deriving a stage needs about the whole graph, computed once: a stage is then derived in
/// time proportional to its own nodes. Deriving each candidate stage from the whole graph made
/// partitioning cubic in the number of nodes, minutes of server time for a 30-layer model.
struct Index<'a> {
    node_of: BTreeMap<&'a str, usize>,
    position: BTreeMap<&'a str, usize>,
    is_param: BTreeSet<&'a str>,
    prediction: &'a str,
    target: &'a str,
    /// For each position, the last position whose node reads its output (`None`: none does).
    last_read: Vec<Option<usize>>,
}

impl<'a> Index<'a> {
    fn new(graph: &'a ModelGraphV1, report: &'a GraphReport) -> Self {
        let order = &report.order;
        let node_of: BTreeMap<&str, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, nd)| (nd.output.as_str(), i))
            .collect();
        let position: BTreeMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, o)| (o.as_str(), i))
            .collect();
        let mut last_read = vec![None; order.len()];
        for (q, name) in order.iter().enumerate() {
            for i in &graph.nodes[node_of[name.as_str()]].inputs {
                if let Some(&p) = position.get(i.as_str()) {
                    last_read[p] = Some(q);
                }
            }
        }
        Index {
            node_of,
            position,
            is_param: graph.parameters.iter().map(|p| p.name.as_str()).collect(),
            prediction: graph.loss.prediction(),
            target: graph.loss.target().name.as_str(),
            last_read,
        }
    }

    /// Is the output at position `p` read by a node at position `b` or later?
    fn read_from(&self, p: usize, b: usize) -> bool {
        self.last_read[p].is_some_and(|q| q >= b)
    }

    /// Stage `s`, nodes `a..b` in execution order.
    fn stage(
        &self,
        graph: &ModelGraphV1,
        report: &GraphReport,
        s: usize,
        a: usize,
        b: usize,
        last: bool,
    ) -> StageSpec {
        let order = &report.order;
        let mut params: Vec<String> = Vec::new();
        let mut inputs: Vec<String> = Vec::new();
        for name in &order[a..b] {
            for i in &graph.nodes[self.node_of[name.as_str()]].inputs {
                let produced_here = self
                    .position
                    .get(i.as_str())
                    .is_some_and(|p| *p >= a && *p < b);
                if self.is_param.contains(i.as_str()) {
                    if !params.contains(i) {
                        params.push(i.clone());
                    }
                } else if !produced_here && !inputs.contains(i) {
                    inputs.push(i.clone());
                }
            }
        }
        if last {
            let pred_here = self
                .position
                .get(self.prediction)
                .is_some_and(|p| *p >= a && *p < b);
            if !pred_here && !inputs.iter().any(|x| x == self.prediction) {
                inputs.push(self.prediction.to_owned());
            }
            inputs.push(self.target.to_owned());
        }
        // Outputs: produced here, read by a node of a later stage or by the loss.
        let outputs = (a..b)
            .filter(|&p| self.read_from(p, b) || (!last && order[p] == self.prediction))
            .map(|p| order[p].clone())
            .collect();
        StageSpec {
            index: s,
            first: a,
            end: b,
            params,
            inputs,
            outputs,
            last,
        }
    }
}

/// Bounds of `k` stages with the same number of nodes, like the executor's segments.
pub fn even_bounds(nodes: usize, k: usize) -> Vec<u32> {
    let k = k.clamp(1, nodes.max(1));
    (0..=k).map(|s| (nodes * s / k) as u32).collect()
}

/// Estimated memory of a stage task, backward pass included (the most demanding): parameters and
/// their gradients, the stage's activations, inputs and their gradients, gradients received for
/// the outputs, and the micro-batch. An upper bound on the executor's measurement
/// (`tests/graph_stages.rs`).
pub fn stage_memory(
    graph: &ModelGraphV1,
    report: &GraphReport,
    stage: &StageSpec,
    batch: u64,
) -> u64 {
    Sizes::new(graph, report, batch).stage_memory(stage)
}

/// Bytes of every tensor of the graph for a micro-batch, computed once per partition.
struct Sizes<'a> {
    act: BTreeMap<&'a str, u64>,
    data: BTreeMap<&'a str, u64>,
    /// A trainable parameter counts twice (value and gradient).
    params: BTreeMap<&'a str, u64>,
    /// Bytes of the output at each position of the execution order.
    at: Vec<u64>,
}

impl<'a> Sizes<'a> {
    fn new(graph: &'a ModelGraphV1, report: &'a GraphReport, batch: u64) -> Self {
        let act = report
            .nodes
            .iter()
            .map(|c| {
                (
                    c.output.as_str(),
                    c.activation_bytes_per_example.saturating_mul(batch),
                )
            })
            .collect();
        let data = report
            .input_bytes_per_example
            .iter()
            .map(|(k, v)| (k.as_str(), v.saturating_mul(batch)))
            .chain(std::iter::once({
                let t = graph.loss.target();
                let elems: u64 = t.shape.iter().map(|d| (*d).max(1)).product();
                (
                    t.name.as_str(),
                    (elems * if t.dtype == "i64" { 8 } else { 4 }).saturating_mul(batch),
                )
            }))
            .collect();
        let params = graph
            .parameters
            .iter()
            .map(|p| {
                let b = p.shape.iter().product::<u64>() * 4;
                (p.name.as_str(), if p.trainable { 2 * b } else { b })
            })
            .collect();
        let mut sizes = Sizes {
            act,
            data,
            params,
            at: Vec::new(),
        };
        sizes.at = report.order.iter().map(|o| sizes.bytes_of(o)).collect();
        sizes
    }

    fn bytes_of(&self, t: &str) -> u64 {
        self.act
            .get(t)
            .or_else(|| self.data.get(t))
            .copied()
            .unwrap_or(0)
    }

    fn stage_memory(&self, stage: &StageSpec) -> u64 {
        let params: u64 = stage
            .params
            .iter()
            .filter_map(|n| self.params.get(n.as_str()))
            .sum();
        let activations: u64 = self.at[stage.first..stage.end].iter().sum();
        let inputs: u64 = stage.inputs.iter().map(|t| 2 * self.bytes_of(t)).sum();
        let outputs: u64 = stage.outputs.iter().map(|t| self.bytes_of(t)).sum();
        params
            .saturating_add(activations)
            .saturating_add(inputs)
            .saturating_add(outputs)
    }
}

/// Greedy bounds: each stage takes as many nodes as `capacity` allows. Rejects, by name, a node
/// that does not fit on its own: its matrix must then be tiled (D3).
pub fn partition(
    graph: &ModelGraphV1,
    report: &GraphReport,
    batch: u64,
    capacity: u64,
) -> Result<Vec<u32>, String> {
    let n = report.order.len();
    let mut bounds = vec![0u32];
    let mut start = 0usize;
    // Cuts that an integer tensor would cross (ADR-0029) are not candidates.
    let index = Index::new(graph, report);
    let sizes = Sizes::new(graph, report, batch);
    let crossing = |cut: usize| -> bool {
        (0..cut).any(|p| {
            report.integer_tensors.contains_key(&report.order[p]) && index.read_from(p, cut)
        })
    };
    while start < n {
        let mut end = start + 1;
        let fits = |end: usize| -> Result<bool, String> {
            let st = index.stage(graph, report, bounds.len() - 1, start, end, end == n);
            Ok(sizes.stage_memory(&st) <= capacity)
        };
        if !fits(end)? {
            return Err(format!(
                "node '{}' does not fit on its own in {capacity} bytes with a micro-batch of \
{batch}: tile its matrix (ADR-0019) or reduce the micro-batch",
                report.order[start]
            ));
        }
        while end < n && fits(end + 1)? {
            end += 1;
        }
        // Back off to the last cut no integer tensor crosses.
        let mut cut = end;
        while cut < n && cut > start && crossing(cut) {
            cut -= 1;
        }
        if cut == start {
            return Err(format!(
                "no stage bound after node '{}' fits in {capacity} bytes without cutting through an \
integer tensor (ADR-0029)",
                report.order[start]
            ));
        }
        let end = cut;
        bounds.push(end as u32);
        start = end;
        if bounds.len() - 1 > MAX_STAGES {
            return Err(format!(
                "more than {MAX_STAGES} stages of {capacity} bytes: larger machines or a smaller \
micro-batch are needed"
            ));
        }
    }
    Ok(bounds)
}
