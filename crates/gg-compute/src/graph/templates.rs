//! Generated model skeletons, for sizing without writing hundreds of nodes by hand.
//!
//! These skeletons have the right structure, the right parameters and the right activation flows.
//! The transformer has been **causal** by default since `gg-ops/2` (ADR-0020): a position sees only
//! the positions before it, which makes it an autoregressive language model. It encodes positions
//! with RoPE by default (ADR-0021), scaling included (ADR-0022).

use std::collections::BTreeMap;

use super::ops::Attr;
use super::rope::{self, RopeScaling};
use super::{FORMAT, Init, Loss, ModelGraphV1, Node, ParamSpec, TensorSpec};

/// How the skeleton encodes a token's position (ADR-0021).
#[derive(Clone, Debug, PartialEq)]
pub enum Positions {
    /// No encoding: only the causal mask distinguishes positions.
    None,
    /// One learned vector per position, added to the embedding (GPT-2, nanoGPT).
    Learned,
    /// Rotation of queries and keys according to position (RoPE, Llama models).
    Rotary {
        /// Frequency base.
        base: f64,
        /// Frequency scaling (Llama 3.1, YaRN, LongRoPE…), ADR-0022.
        scaling: RopeScaling,
    },
}

impl Positions {
    /// RoPE with base `base`, without scaling.
    pub fn rotary(base: f64) -> Self {
        Positions::Rotary {
            base,
            scaling: RopeScaling::None,
        }
    }
}

/// Dimensions of a decoder transformer.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformerDims {
    /// Layers.
    pub layers: u64,
    /// Model width.
    pub dim: u64,
    /// Attention heads; `dim` must be a multiple of it.
    pub heads: u64,
    /// Sequence length.
    pub seq: u64,
    /// Vocabulary size.
    pub vocab: u64,
    /// Width of the feed-forward network (often 4 × dim).
    pub ffn: u64,
    /// Causal attention: a position sees only the positions before it. This is what a language
    /// model predicting the next token requires; without it, the model reads the answer.
    pub causal: bool,
    /// Position encoding.
    pub positions: Positions,
}

impl TransformerDims {
    /// All parameters of the skeleton, trainable and frozen.
    pub fn parameters(&self) -> u64 {
        self.trainable_parameters() + self.frozen_parameters()
    }

    /// Parameters that receive a gradient.
    pub fn trainable_parameters(&self) -> u64 {
        let per_layer = 4 * self.dim * self.dim + 2 * self.dim * self.ffn;
        let learned = match self.positions {
            Positions::Learned => self.seq * self.dim,
            _ => 0,
        };
        self.layers * per_layer + 2 * self.vocab * self.dim + learned
    }

    /// Frozen constants: the rotation tables and the −1 of the half rotation.
    pub fn frozen_parameters(&self) -> u64 {
        match self.positions {
            Positions::Rotary { .. } => 2 * self.seq * (self.dim / self.heads) + 1,
            _ => 0,
        }
    }
}

fn ints(v: &[i64]) -> Attr {
    Attr::Ints(v.to_vec())
}

fn node(op: &str, inputs: &[&str], output: &str, attrs: &[(&str, Attr)]) -> Node {
    Node {
        op: op.to_owned(),
        inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
        output: output.to_owned(),
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect::<BTreeMap<_, _>>(),
    }
}

fn param(name: &str, shape: &[u64], init: Init) -> ParamSpec {
    ParamSpec {
        name: name.to_owned(),
        shape: shape.to_vec(),
        init,
        trainable: true,
    }
}

fn frozen(name: &str, shape: &[u64], init: Init) -> ParamSpec {
    ParamSpec {
        trainable: false,
        ..param(name, shape, init)
    }
}

/// RoPE-style rotation, "rotate_half" layout: `x·cos + [−x₂, x₁]·sin` on the last axis of a
/// `[N, heads, T, d_head]` tensor, using only the profile's operators. Uses the frozen parameters
/// `rope.cos`, `rope.sin` (`[1, 1, T, d_head]` tables) and `rope.neg` (−1), which the caller
/// declares. Returns the name of the output.
pub fn rotary(nodes: &mut Vec<Node>, x: &str, out: &str, dh: u64) -> String {
    rotate(nodes, x, out, dh)
}

fn rotate(nodes: &mut Vec<Node>, x: &str, out: &str, dh: u64) -> String {
    let half = (dh / 2) as i64;
    let full = dh as i64;
    let s = |suffix: &str| format!("{out}.{suffix}");
    nodes.extend([
        node(
            "Slice",
            &[x],
            &s("x1"),
            &[
                ("axis", Attr::Int(3)),
                ("start", Attr::Int(0)),
                ("end", Attr::Int(half)),
            ],
        ),
        node(
            "Slice",
            &[x],
            &s("x2"),
            &[
                ("axis", Attr::Int(3)),
                ("start", Attr::Int(half)),
                ("end", Attr::Int(full)),
            ],
        ),
        node("Mul", &[&s("x2"), "rope.neg"], &s("nx2"), &[]),
        node(
            "Concat",
            &[&s("nx2"), &s("x1")],
            &s("half"),
            &[("axis", Attr::Int(3))],
        ),
        node("Mul", &[x, "rope.cos"], &s("c"), &[]),
        node("Mul", &[&s("half"), "rope.sin"], &s("r"), &[]),
        node("Add", &[&s("c"), &s("r")], out, &[]),
    ]);
    out.to_owned()
}

/// Decoder transformer with multi-head attention, RMS pre-normalization and GELU activation.
///
/// The attention's 1/√d_head scale is folded into the initialization of `wq`, which is equivalent
/// for sizing and for training.
pub fn transformer(d: TransformerDims) -> Result<ModelGraphV1, String> {
    if d.layers == 0 || d.dim == 0 || d.heads == 0 || d.seq == 0 || d.vocab == 0 || d.ffn == 0 {
        return Err("all dimensions must be strictly positive".into());
    }
    if !d.dim.is_multiple_of(d.heads) {
        return Err(format!(
            "dim {} is not a multiple of heads {}",
            d.dim, d.heads
        ));
    }
    let dh = d.dim / d.heads;
    let (t, dim, nh, ffn, v) = (d.seq as i64, d.dim as i64, d.heads as i64, d.ffn, d.vocab);
    let q_std = 0.02 / (dh as f64).sqrt();

    let mut parameters = vec![param("emb", &[v, d.dim], Init::Normal { std: 0.02 })];
    let mut nodes = vec![node("Gather", &["emb", "ids"], "h0", &[])];
    let mut h = "h0".to_owned();
    match &d.positions {
        Positions::None => {}
        Positions::Learned => {
            // One vector per position, added to the embedding by broadcasting over the batch.
            parameters.push(param("pos", &[d.seq, d.dim], Init::Normal { std: 0.02 }));
            nodes.push(node("Add", &["h0", "pos"], "h0p", &[]));
            h = "h0p".to_owned();
        }
        Positions::Rotary { base, scaling } => {
            if !dh.is_multiple_of(2) {
                return Err(format!("RoPE requires an even head width, got {dh}"));
            }
            let rotary = rope::rotary(*base, dh, d.seq, scaling)?;
            // Tables shared by all layers, frozen: they are constants of the model.
            let table = [1, 1, d.seq, dh];
            parameters.extend([
                frozen("rope.cos", &table, Init::RotaryCos(rotary.clone())),
                frozen("rope.sin", &table, Init::RotarySin(rotary)),
                frozen("rope.neg", &[1], Init::Constant { value: -1.0 }),
            ]);
        }
    }

    for l in 0..d.layers {
        let p = |s: &str| format!("l{l}.{s}");
        parameters.extend([
            param(&p("wq"), &[d.dim, d.dim], Init::Normal { std: q_std }),
            param(&p("wk"), &[d.dim, d.dim], Init::XavierUniform),
            param(&p("wv"), &[d.dim, d.dim], Init::XavierUniform),
            param(&p("wo"), &[d.dim, d.dim], Init::XavierUniform),
            param(&p("wu"), &[d.dim, ffn], Init::XavierUniform),
            param(&p("wd"), &[ffn, d.dim], Init::XavierUniform),
        ]);
        let n = |s: &str| format!("l{l}.{s}");
        nodes.extend([
            node("RMSNorm", &[&h], &n("n1"), &[]),
            node("MatMul", &[&n("n1"), &p("wq")], &n("q"), &[]),
            node("MatMul", &[&n("n1"), &p("wk")], &n("k"), &[]),
            node("MatMul", &[&n("n1"), &p("wv")], &n("v"), &[]),
            node(
                "Reshape",
                &[&n("q")],
                &n("q4"),
                &[("shape", ints(&[-1, t, nh, dh as i64]))],
            ),
            node(
                "Reshape",
                &[&n("k")],
                &n("k4"),
                &[("shape", ints(&[-1, t, nh, dh as i64]))],
            ),
            node(
                "Reshape",
                &[&n("v")],
                &n("v4"),
                &[("shape", ints(&[-1, t, nh, dh as i64]))],
            ),
            node(
                "Transpose",
                &[&n("q4")],
                &n("qt"),
                &[("perm", ints(&[0, 2, 1, 3]))],
            ),
            // Keys go to [N, heads, d_head, T] for the product; with RoPE, in two steps, so they
            // can be rotated in [N, heads, T, d_head] in between.
            match &d.positions {
                Positions::Rotary { .. } => node(
                    "Transpose",
                    &[&n("k4")],
                    &n("kh"),
                    &[("perm", ints(&[0, 2, 1, 3]))],
                ),
                _ => node(
                    "Transpose",
                    &[&n("k4")],
                    &n("kt"),
                    &[("perm", ints(&[0, 2, 3, 1]))],
                ),
            },
            node(
                "Transpose",
                &[&n("v4")],
                &n("vt"),
                &[("perm", ints(&[0, 2, 1, 3]))],
            ),
        ]);
        // RoPE rotates queries and keys according to position, in [N, heads, T, d_head].
        let q_in = match &d.positions {
            Positions::Rotary { .. } => {
                let q = rotate(&mut nodes, &n("qt"), &n("qr"), dh);
                let k = rotate(&mut nodes, &n("kh"), &n("kr"), dh);
                nodes.push(node(
                    "Transpose",
                    &[&k],
                    &n("kt"),
                    &[("perm", ints(&[0, 1, 3, 2]))],
                ));
                q
            }
            _ => n("qt"),
        };
        nodes.push(node("MatMul", &[&q_in, &n("kt")], &n("scores"), &[]));
        let attended = if d.causal {
            nodes.push(node("CausalMask", &[&n("scores")], &n("masked"), &[]));
            n("masked")
        } else {
            n("scores")
        };
        nodes.extend([
            node(
                "Softmax",
                &[&attended],
                &n("weights"),
                &[("axis", Attr::Int(3))],
            ),
            node("MatMul", &[&n("weights"), &n("vt")], &n("ctx"), &[]),
            node(
                "Transpose",
                &[&n("ctx")],
                &n("ctx_t"),
                &[("perm", ints(&[0, 2, 1, 3]))],
            ),
            node(
                "Reshape",
                &[&n("ctx_t")],
                &n("ctx2"),
                &[("shape", ints(&[-1, t, dim]))],
            ),
            node("MatMul", &[&n("ctx2"), &p("wo")], &n("attn"), &[]),
            node("Add", &[&h, &n("attn")], &n("h1"), &[]),
            node("RMSNorm", &[&n("h1")], &n("n2"), &[]),
            node("MatMul", &[&n("n2"), &p("wu")], &n("up"), &[]),
            node("Gelu", &[&n("up")], &n("act"), &[]),
            node("MatMul", &[&n("act"), &p("wd")], &n("down"), &[]),
            node("Add", &[&n("h1"), &n("down")], &n("h2"), &[]),
        ]);
        h = n("h2");
    }
    parameters.push(param("wlm", &[d.dim, v], Init::XavierUniform));
    nodes.extend([
        node("RMSNorm", &[&h], "nf", &[]),
        node("MatMul", &["nf", "wlm"], "seqlogits", &[]),
        node(
            "Reshape",
            &["seqlogits"],
            "logits",
            &[("shape", ints(&[-1, v as i64]))],
        ),
    ]);

    Ok(ModelGraphV1 {
        format: FORMAT.to_owned(),
        name: format!("transformer-{}x{}", d.layers, d.dim),
        inputs: vec![TensorSpec {
            name: "ids".into(),
            shape: vec![0, d.seq],
            dtype: "i64".into(),
        }],
        parameters,
        nodes,
        loss: Loss::CrossEntropy {
            prediction: "logits".into(),
            target: TensorSpec {
                name: "y".into(),
                shape: vec![0, d.seq],
                dtype: "i64".into(),
            },
        },
        functions: Vec::new(),
    })
}
