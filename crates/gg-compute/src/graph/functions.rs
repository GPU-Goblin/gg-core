//! Description functions: named subgraphs, called like operators (ADR-0025).
//!
//! A function groups nodes under a name — a transformer block, a hand-written normalization, a
//! recurrent cell — and is called with `"op": "fn:<name>"`, as many times as needed, each call with
//! its own inputs and its own parameters. It is a custom operator without code: its body uses only
//! the published profile, `Formula` included, and other functions declared before it (hence no
//! recursion).
//!
//! A function is **closed**: its body reads only its formal inputs and what it produces itself.
//! The parameters of a call are passed to it as inputs, which keeps every call readable.
//!
//! The server **inlines** calls at submission (`inline`): what then travels to the workers, the
//! validator and the plan is a flat description, to which everything else — validation, stages,
//! shards, matrix tiling — applies unchanged. An inner tensor of a call whose output is `h1` is
//! named `h1/<inner name>`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{GraphError, ModelGraphV1, Node, bounds, check_name};

/// Prefix of a function call in a node's `op` field.
pub const CALL_PREFIX: &str = "fn:";
/// Maximum number of functions in a description.
pub const MAX_FUNCTIONS: usize = 64;
/// Maximum number of formal inputs of a function.
pub const MAX_FUNCTION_INPUTS: usize = 32;

/// A function: formal inputs, body, output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionSpec {
    /// Name, called with `fn:<name>`.
    pub name: String,
    /// Formal inputs, in the order of the call's inputs.
    pub inputs: Vec<String>,
    /// Body: profile nodes or calls to functions declared earlier.
    pub nodes: Vec<Node>,
    /// Tensor of the body that the function returns.
    pub output: String,
}

fn fail(msg: String) -> GraphError {
    GraphError::Format(msg)
}

/// Checks the functions themselves, before any inlining: names, inputs, closure, order.
fn check_functions(functions: &[FunctionSpec]) -> Result<(), GraphError> {
    if functions.len() > MAX_FUNCTIONS {
        return Err(GraphError::Bound(format!(
            "{} functions, the maximum is {MAX_FUNCTIONS}",
            functions.len()
        )));
    }
    let mut declared: BTreeSet<&str> = BTreeSet::new();
    for f in functions {
        check_name("function", &f.name)?;
        if f.inputs.is_empty() || f.inputs.len() > MAX_FUNCTION_INPUTS {
            return Err(GraphError::Bound(format!(
                "function '{}': 1 to {MAX_FUNCTION_INPUTS} inputs, {} declared",
                f.name,
                f.inputs.len()
            )));
        }
        if f.nodes.is_empty() || f.nodes.len() > bounds::MAX_NODES {
            return Err(GraphError::Bound(format!(
                "function '{}': 1 to {} nodes, {} declared",
                f.name,
                bounds::MAX_NODES,
                f.nodes.len()
            )));
        }
        let mut known: BTreeSet<&str> = BTreeSet::new();
        for i in &f.inputs {
            check_name("function input", i)?;
            if !known.insert(i) {
                return Err(GraphError::Name(format!(
                    "function '{}': input '{i}' declared twice",
                    f.name
                )));
            }
        }
        let produced: BTreeSet<&str> = f.nodes.iter().map(|n| n.output.as_str()).collect();
        for n in &f.nodes {
            check_name("function node", &n.output)?;
            if known.contains(n.output.as_str()) {
                return Err(GraphError::Name(format!(
                    "function '{}': '{}' is both an input and a node output",
                    f.name, n.output
                )));
            }
            for i in &n.inputs {
                if !known.contains(i.as_str()) && !produced.contains(i.as_str()) {
                    return Err(fail(format!(
                        "function '{}': tensor '{i}' is neither one of its inputs nor produced by \
it; a function reads only its inputs and what it produces, pass it in as an input",
                        f.name
                    )));
                }
            }
            if let Some(callee) = n.op.strip_prefix(CALL_PREFIX)
                && !declared.contains(callee)
            {
                return Err(fail(format!(
                    "function '{}': call to '{callee}', which is not declared before it (a \
function only calls the functions that precede it, and therefore never itself)",
                    f.name
                )));
            }
        }
        if produced.len() != f.nodes.len() {
            return Err(GraphError::Name(format!(
                "function '{}': a tensor is produced twice",
                f.name
            )));
        }
        if !produced.contains(f.output.as_str()) {
            return Err(fail(format!(
                "function '{}': its output '{}' is not produced by any of its nodes",
                f.name, f.output
            )));
        }
        if !declared.insert(&f.name) {
            return Err(GraphError::Name(format!(
                "function '{}' declared twice",
                f.name
            )));
        }
    }
    Ok(())
}

struct Inliner<'a> {
    functions: BTreeMap<&'a str, &'a FunctionSpec>,
    out: Vec<Node>,
}

impl<'a> Inliner<'a> {
    /// Emits `node`, whose inputs are already names in the flat graph.
    fn emit(&mut self, node: Node) -> Result<(), GraphError> {
        let Some(name) = node.op.strip_prefix(CALL_PREFIX) else {
            if self.out.len() >= bounds::MAX_NODES {
                return Err(GraphError::Bound(format!(
                    "more than {} nodes once functions are inlined",
                    bounds::MAX_NODES
                )));
            }
            self.out.push(node);
            return Ok(());
        };
        let f = *self.functions.get(name).ok_or_else(|| {
            fail(format!(
                "node '{}': function '{name}' is not declared",
                node.output
            ))
        })?;
        if !node.attrs.is_empty() {
            return Err(GraphError::Attribute(format!(
                "node '{}': a function call takes no attributes",
                node.output
            )));
        }
        if node.inputs.len() != f.inputs.len() {
            return Err(GraphError::Shape(format!(
                "node '{}': function '{name}' expects {} inputs ({}), {} given",
                node.output,
                f.inputs.len(),
                f.inputs.join(", "),
                node.inputs.len()
            )));
        }
        let formal: BTreeMap<&str, &str> = f
            .inputs
            .iter()
            .map(String::as_str)
            .zip(node.inputs.iter().map(String::as_str))
            .collect();
        let rename = |inner: &str| -> String {
            if let Some(actual) = formal.get(inner) {
                (*actual).to_owned()
            } else if inner == f.output {
                node.output.clone()
            } else {
                format!("{}/{inner}", node.output)
            }
        };
        for inner in &f.nodes {
            self.emit(Node {
                op: inner.op.clone(),
                inputs: inner.inputs.iter().map(|i| rename(i)).collect(),
                output: rename(&inner.output),
                attrs: inner.attrs.clone(),
            })?;
        }
        Ok(())
    }
}

/// The flat description: each call replaced by its function's body, renamed. A description
/// without functions is returned as is. Deterministic: same description, same result, byte for
/// byte.
pub fn inline(graph: &ModelGraphV1) -> Result<ModelGraphV1, GraphError> {
    if graph.functions.is_empty() {
        return Ok(graph.clone());
    }
    check_functions(&graph.functions)?;
    let mut inliner = Inliner {
        functions: graph
            .functions
            .iter()
            .map(|f| (f.name.as_str(), f))
            .collect(),
        out: Vec::with_capacity(graph.nodes.len()),
    };
    for node in &graph.nodes {
        inliner.emit(node.clone())?;
    }
    Ok(ModelGraphV1 {
        nodes: inliner.out,
        functions: Vec::new(),
        ..graph.clone()
    })
}
