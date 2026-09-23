# Describing a model — the `gg-graph/v1` format

A researcher never sends code to the platform. They send **the description of their model**: a
computation graph, in JSON. The executor builds it and trains it with the platform's code, which is
the same for everyone, bounded and verifiable. See [gg-server/docs/adr/0016-computation-graphs.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0016-computation-graphs.md) for the
decision and its reasons.

This document describes the format, the operator profile and how to have a description checked
**before** reserving a single credit.

## The shortest path: PyTorch

If your model is already in PyTorch, you do not have to write any JSON. The driver captures the
module and produces this description for you, rejecting any operator outside the profile before
anything is sent: see [gg-python/docs/PYTHON.md](https://github.com/GPU-Goblin/gg-python/blob/main/docs/PYTHON.md). The rest of this page describes the underlying
format, which is useful for reading it, fixing it or writing it by hand.

## Check before submitting

```bash
gg job check-model my-model.json --batch 128      # offline, no token required
gg job ops                                        # the profile this server accepts
curl -s http://server/v1/catalog/ops | jq .       # the same profile, without the command line
```

`check-model` answers in a second. If it rejects the description, it names the cause: the operator
missing from the profile, the shape that does not match, the tensor used without being produced,
the cycle, the bound exceeded.

```
$ gg job check-model examples/models/transformer-block.json --batch 64
model "transformer-block" accepted by profile gg-ops/8
  nodes                 : 18
  operators             : Add, Gather, Gelu, MatMul, RMSNorm, Reshape, Softmax, Transpose
  parameters            : 17170432 (65.5 MiB), 17170432 trainable (65.5 MiB exchanged)
  compute per example   : 6.949 GFLOP training (2.316 GFLOP forward only)
  memory, batch 64      : 1306.8 MiB, excluding optimizer state
  reproducible within a tolerance, because of: Gelu, MatMul, RMSNorm, Softmax

$ gg job check-model model-with-selu.json
model rejected: operator 'Selu' (node 'a1') is not part of profile gg-ops/8
the full profile is available with `gg job ops` or at /v1/catalog/ops
```

The exit code is 2 when the description is rejected, which makes it easy to chain in a script.

## Structure of a description

```json
{
  "format": "gg-graph/v1",
  "name": "mlp-tabular",
  "inputs":     [{ "name": "x", "shape": [0, 64] }],
  "parameters": [{ "name": "w1", "shape": [64, 256], "init": "xavier_uniform" }],
  "nodes":      [{ "op": "MatMul", "inputs": ["x", "w1"], "output": "h1" }],
  "loss": { "cross_entropy": { "prediction": "logits",
            "target": { "name": "y", "shape": [0], "dtype": "i64" } } }
}
```

- **`inputs`**: the data tensors. A dimension set to `0` marks the batch, unknown in the
  description and set by the scheduler. Accepted types: `f32`, and `i64` for indices.
- **`parameters`**: the trainable tensors. All their dimensions are known. `init` is `zeros`,
  `ones`, `xavier_uniform` or `{"normal": {"std": 0.02}}`; sampling goes through the protocol's
  deterministic generator, so two runs start from the same weights.
- **`nodes`**: the operations. Each node names its operator, its inputs and its single output.
  **The order in which nodes are written does not matter**: the execution order is derived from
  the dependencies, and a cycle is rejected.
- **`attrs`**: a node's attributes, as plain JSON values (`"axis": 1`, `"padding": [1, 1]`). An
  unknown attribute is **rejected**, never silently ignored: a typo does not turn into a wrong
  experiment.
- **`loss`**: `mean_squared_error` or `cross_entropy`, on a tensor produced by the graph.

## Bounds

| Bound | Value |
|---|---|
| Nodes | 4,096 |
| Named tensors | 8,192 |
| Tensor rank | 4 |
| Trainable parameters | 1,000,000,000,000 (feasibility is judged with `gg job plan`, not here) |
| Name length | 64 characters (`a-z A-Z 0-9 _ - . /`) |

One more bound, implicit but decisive: **each stage of the model must fit on a contributor's
machine** (ADR-0017). `check-model` gives the memory of a whole model for a given batch;
`gg job plan` splits it into stages when it does not fit and says what that costs.

## Knowing whether it is feasible before submitting: `gg job plan`

```bash
gg job template transformer --layers 24 --dim 2048 --heads 16 --seq 1024 --out t1b.json
gg job plan t1b.json --examples 97656 --batch 8 --contributors 100 --local-steps 50
gg job plan t1b.json --examples 97656 --batch 8 --contributors 100 --local-steps 50 \
   --uplink-mbit 1000 --downlink-mbit 1000 --gflops 10000      # laboratory machines on fiber
```

The plan splits the graph into stages that each fit in a contributor's memory, at the least loaded
boundary (the residual stream between two layers), then reports:

- how many stages and parallel pipelines, and how many contributors are used;
- the pace, and **what sets it**: the executor or the link;
- for each cut, the bytes per example and the FLOP per byte transferred;
- the total duration, the total Internet traffic, the weight of synchronization;
- **what a single datacenter GPU would do with the same work**, to decide honestly.

On a transformer with 1.3 billion parameters and 100 million tokens, with 100 contributors on home
fiber (300 Mbit/s up): 10.8 h with the current executor, 1.7 h with a 10 TFLOP/s executor, and
1.0 h at 500 Mbit/s; a single datacenter GPU takes 1.5 h. On fiber, the executor sets the pace; on
ADSL (10 Mbit/s up), the link does, and it takes 2.2 days whatever the executor. The cost model's
assumptions, and what it ignores (contributor failures first and foremost), are in
[gg-server/docs/adr/0017-distribution-plan.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0017-distribution-plan.md).

**The route activations take** (`--topology`) changes everything when the model is split:

| | `peer` (default) | `coordinator` |
|---|---|---|
| Route | stage *i* sends to stage *i+1* | every packet returns to the source, which sends the next one |
| 1.3 B on 100 machines on fiber | **1.7 h** | 5.1 days |
| 6.7 B on 200 machines on fiber | **4.1 h** | 20.3 days |

In a star topology, every tensor crosses the coordinator's link twice, and all the cuts of all the
pipelines share it: this is a global ceiling that neither the number of contributors nor their
speed can raise. The details are in [gg-server/docs/adr/0018-dataflow-scheduler.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0018-dataflow-scheduler.md). When the model fits on
one machine, there is no cut and the topology costs nothing.

### An example at scale: 125 billion parameters on Macs

Measured on 2026-09-19 with the planner, on a transformer with 131.7 billion parameters:

```bash
gg job template transformer --layers 72 --dim 12288 --heads 96 --seq 2048 --vocab 50257 --out m125b.json
gg job plan m125b.json --examples 488281 --batch 1 --contributors 500 --gflops 700 \
   --memory-gib 128 --trainable-fraction 0.01 --recompute --local-steps 100
```

**Minimum number of machines**, i.e. the number of stages: below it, the model fits nowhere.

| Unified memory per Mac | Fully trained | Adapters on a frozen base |
|---|---|---|
| 64 GiB | 48 | 9 |
| 96 GiB | 29 | 6 |
| 128 GiB | 21 | 5 |
| 192 GiB | 14 | 3 |
| 256 GiB | 10 | 2 |

**Duration of fine-tuning on one billion tokens**, 128 GiB Macs, adapters:

| Machines | Current executor (700 GFLOP/s) | Optimized executor (7 TFLOP/s) |
|---|---|---|
| 5 | 15.2 years | 1.5 years |
| 50 | 1.5 years | 55.9 days |
| 500 | 55.5 days | 5.6 days |
| 5,000 | 5.6 days | 13.5 h |

Three caveats, which change the figure:

- **The format only knows 32-bit floating point**: 131.7 billion parameters take 490 GiB, where
  half precision would take 245. This doubles the number of machines; it is a limitation of the
  implementation, not of the method.
- **A Mac's unified memory cancels the offloading lever**: host and GPU share the same pool, so
  there is nowhere to offload the parameters to. The tables take this into account.
- **Nothing has ever run on a Mac.** Burn can target Metal, but engine selection only offers
  Vulkan. The 700 GFLOP/s extrapolate our executor's measured efficiency, about 3 % of the hardware
  on an RTX 5060 Ti; the optimized column assumes 30 %.

These figures apply to **fine-tuning**. Pretraining this model from scratch takes on the order of
2,500 billion tokens, 2,500 times more: out of reach, even with 5,000 optimized machines.

### When a stage does not fit on a GPU

The plan refuses, naming the node, saying where its memory goes, and listing what would make it
fit — marking what is enough:

```
no plan: node 'h0' alone needs 31.50 GiB, the machine offers 12.00 GiB
         (31.25 GiB of parameters and optimizer state, 256.0 MiB of activations for a batch of 4)
  [helps]  divide the micro-batch by 4 → 31.31 GiB (more micro-batches, hence more synchronizations)
  [enough] train only adapters (--trainable-fraction 0.01) → 8.30 GiB (the base model is no longer updated)
  [enough] offload to host memory (--host-offload-gib, 31.25 GiB needed) → 256.0 MiB (4.19 s of PCIe transfer per micro-batch)
```

Four levers, from cheapest to most expensive ([gg-server/docs/adr/0019-when-a-stage-does-not-fit.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0019-when-a-stage-does-not-fit.md)):

| Lever | Option | What it costs |
|---|---|---|
| Reduce the micro-batch | `--batch` | more synchronizations |
| Recompute activations | `--recompute` | about a third more compute |
| Freeze the base, train adapters | `--trainable-fraction 0.01` | the base model is no longer updated |
| Offload to host memory | `--host-offload-gib 64` | PCIe traffic, ~1000 times faster than the link |

Offloading is the decisive lever on a contributor's machine, for a reason that is not obvious:
PCIe sustains 16 GB/s versus 37 MB/s for fiber with 300 Mbit/s up. What would be prohibitive in a
datacenter is almost free here.

**The best split is the one you avoid.** On a transformer with 5.8 billion parameters (21.6 GiB)
and a 12 GiB GPU, the four levers combined bring the model down to **a single stage**: no more
cuts, no more hand-offs, and verification once again covers all of a peer's work. Estimated
duration: 2.3 h on 64 contributors, versus 1.7 h on a datacenter GPU.

Options: `--gflops` (default 300, the measurement of this executor), `--memory-gib` (12),
`--uplink-mbit` (300, common home fiber; 10 for ADSL), `--downlink-mbit` (1000), `--local-steps`
(1), `--audit-rate` (0.25), `--topology` (`peer` or `coordinator`), `--coordinator-uplink-mbit`,
`--recompute`, `--trainable-fraction`, `--host-offload-gib`, `--host-link-gbs`, `--json`.

**A weight larger than the GPU is tiled in the description.** A product `x·W` is written as several
products over column tiles of `W`, stitched back together by `Concat` (at most 8 inputs, so they
are chained beyond that). With `--segments` and `--offload`, the parameter peak is divided
**exactly** by the number of tiles, down to 1/64 measured, for the same loss and the same gradient.
At realistic sizes it costs nothing: on an RTX 5060 Ti, a 64 MiB weight in 64 tiles runs **faster**
than whole (0.68×). What remains is a floor, the stitched output, which only the micro-batch
lowers. Details and measurements in [gg-server/docs/adr/0019-when-a-stage-does-not-fit.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0019-when-a-stage-does-not-fit.md).

The other four levers are **executed**, not just priced. A parameter is frozen in the description
itself:

```json
{"name": "w1", "shape": [6, 12], "init": "xavier_uniform", "trainable": false}
```

A frozen parameter keeps its value, receives no gradient, does not travel during synchronization
and has no optimizer state. At submission, `--segments N` and `--offload` request recomputation and
offloading:

```bash
gg job collective --project <id> --graph model.json --dataset x=train_x.ggt --dataset y=train_y.ggt \
   --segments 8 --offload --rounds 20 --batch 256 --tasks 4
```

They do not change the gradient: the measured relative difference is 1.8e-5, which is rounding, and
it decreases with more segments. The validator recomputes with the same levers. Exit codes: 2 if
the description is rejected, 3 if no plan exists (a single node too large for a machine, or fewer
contributors than stages); the message names the cause.

`gg job template transformer` generates a decoder transformer skeleton: the right structure, the
right parameters, the right data flows. It is **causal by default** (`--bidirectional` for an
encoder) and encodes position with **RoPE rotations** by default
(`--positions rotary|learned|none`, `--rope-base 10000`). `--rope-scaling` takes the `rope_scaling`
object of a Hugging Face `config.json` as is — Llama 3.1, YaRN, LongRoPE, linear, dynamic NTK —
inline or as `@file.json`; `--rope-max-positions` gives its `max_position_embeddings` (see "Scaled
RoPE" below).

## Evaluating a description offline: `gg job eval-model`

```bash
gg job eval-model model.json --checkpoint weights.ggb --input ids=ids.ggt --out logits.ggt
```

Forward pass only, on the CPU reference profile, without a server. Without `--checkpoint`, the
initialization drawn from the description's seed is used; without `--output`, the prediction read
by the loss. Useful to evaluate a trained checkpoint, and to check that a PyTorch capture computes
the same function as its module.

## The `gg-ops/8` operator profile

The platform declares exactly what it can execute. Forty-five operators cover tabular data and
perceptrons, convolutional vision, transformers with positional encoding, mixtures of experts,
recurrent networks, time series, speech (vocoders and encoders) and fixed graphs. `gg-ops/2` added `CausalMask`; `gg-ops/3`,
`Slice` and the position tables; `gg-ops/4`, rotary tables with explicit frequencies, without a new
operator; `gg-ops/5`, `ReduceMean` and `Rsqrt`, used by the hand-written normalizations of
`transformers` models (ADR-0023); `gg-ops/6`, the `Formula` custom operator and description
functions (ADR-0025); `gg-ops/7`, the eleven operators and fifteen `Formula` functions that real
models were measured to lack (ADR-0027); `gg-ops/8`, integer tensors computed inside the graph —
`argmax`, `topk`, `sort`, gathers and scatters at a computed index, the routing of a mixture of
experts (ADR-0029). Each version remains compatible with the previous one, with
one exception: since `gg-ops/4`, **an unknown field is rejected** everywhere in the description
instead of being ignored (ADR-0022).

| Operator | ONNX | Attributes | Reproducibility |
|---|---|---|---|
| MatMul | MatMul (opset 13) | — | tolerance |
| Add, Sub, Mul, Div | opset 14 | — | bit for bit |
| Relu | Relu (opset 14) | — | bit for bit |
| Gelu | Gelu (opset 20) | `approximate` (`"none"`, `"tanh"`: the tanh form, since `gg-ops/8`) | tolerance |
| Sigmoid, Tanh | opset 13 | — | tolerance |
| Softmax | Softmax (opset 13) | `axis` | tolerance |
| ReduceMean | ReduceMean (opset 18) | `axis`*, `keepdims` | tolerance |
| ReduceSum | ReduceSum (opset 13) | `axis`*, `keepdims` | tolerance |
| ReduceMax, ReduceMin | opset 18 | `axis`*, `keepdims` | bit for bit |
| ReduceProd | ReduceProd (opset 18) | `axis`*, `keepdims` | tolerance |
| CumSum | CumSum (opset 14) | `axis`* | tolerance |
| Rsqrt | Sqrt + Reciprocal (opset 13) | — | tolerance |
| CausalMask | Trilu (opset 14) + Where (opset 16) | — | bit for bit |
| Slice | Slice (opset 13) | `axis`*, `start`*, `end`* | bit for bit |
| Flip | Slice with step −1 (opset 13) | `axes`* | bit for bit |
| Pad | Pad (opset 18) | `pads`*, `mode`, `value` | bit for bit |
| Resize | Resize (opset 19) | `size`*, `mode`, `align_corners` | tolerance |
| LayerNorm | LayerNormalization (opset 17) | `axis`, `epsilon` | tolerance |
| RMSNorm | RMSNormalization (opset 23) | `axis`, `epsilon` | tolerance |
| Conv2d | Conv (opset 11, 2D) | `stride`, `padding`, `dilation`, `groups` | tolerance |
| ConvTranspose2d | ConvTranspose (opset 11, 2D) | `stride`, `padding`, `output_padding`, `dilation`, `groups` | tolerance |
| MaxPool2d | MaxPool (opset 12, 2D) | `kernel`*, `stride`, `padding` | bit for bit |
| AvgPool2d | AveragePool (opset 11, 2D) | `kernel`*, `stride`, `padding` | tolerance |
| BatchNorm2d | BatchNormalization (opset 15) | `epsilon`, `momentum` | tolerance |
| Reshape | Reshape (opset 14) | `shape`* | bit for bit |
| Transpose | Transpose (opset 13) | `perm`* | bit for bit |
| Flatten | Flatten (opset 13) | `axis` | bit for bit |
| Concat | Concat (opset 13) | `axis`* | bit for bit |
| Gather | Gather (opset 13) | `axis` | bit for bit |
| IndexSelect | Gather (opset 13), constant indices | `axis`*, `index`* | bit for bit |
| IndexAdd | ScatterElements (opset 18, add), constant indices | `axis`*, `index`* | tolerance |
| Dropout | Dropout (opset 13) | `rate`* | deterministic draw |
| Formula | — (element-wise expression) | `expr`* | tolerance |
| ArgMax, ArgMin | ArgMax, ArgMin (opset 13) | `axis`*, `keepdims` | bit for bit |
| ArgTopK | TopK (opset 11), indices | `axis`*, `k`*, `largest` | bit for bit |
| ArgSort | — | `axis`*, `descending` | bit for bit |
| Cast | Cast (opset 13) | `to`* (`"f32"`) | bit for bit |
| GatherElements | GatherElements (opset 13) | `axis`* | bit for bit |
| ScatterElements | ScatterElements (opset 18) | `axis`*, `reduction` (`add`, `none`) | tolerance |

### Operators of `gg-ops/7`

**No operator acts along the batch axis.** The batch is split between workers; a reduction, a
flip, a padding or an index along it would mix examples. The description is rejected with the
reason.

- **`ConvTranspose2d`** takes the kernel in PyTorch's layout `[C_in, C_out/groups, KH, KW]`, an
  optional bias `[C_out]`. Output size: `(i − 1)·stride − 2·padding + dilation·(k − 1) +
  output_padding + 1`, with `output_padding` smaller than the stride or the dilation. A 1D
  transposed convolution is written with a height of 1 (the PyTorch driver does it).
- **`Pad`**: `pads` lists the start of every axis, then the end of every axis (ONNX order), non
  negative (a crop is a `Slice`). `mode` is `constant` (with `value`, 0 by default), `reflect`
  (mirror without the edge; each padding smaller than the axis) or `edge` (repeat the edge).
- **`Resize`** acts on the last axis of `[N, C, L]` or the last two of `[N, C, H, W]`. `size` gives
  the target. `nearest` takes `min(floor(o · in/out), in − 1)`, the ratio in f32, as PyTorch's
  `nearest`; `linear` (bilinear in 2D) uses half-pixel coordinates, or aligned corners with
  `"align_corners": 1`, exactly as PyTorch computes them.
- **`ReduceMax`, `ReduceMin`**: when several elements tie for the extremum, the gradient is shared
  equally between them, as `torch.amax` does (`torch.max(dim)` gives it to one index; at a tie the
  function has no derivative, and both are valid subgradients).
- **`ReduceProd`** is computed by pairwise products: its derivative stays exact when an element is
  zero.
- **`IndexSelect`** and **`IndexAdd`** carry their indices in the description (`"index": [3, 0,
  3, …]`, at most 2²⁰, each inside the axis, repetitions allowed). `IndexSelect` selects the
  listed slices; `IndexAdd(x, src)` adds the slices of `src` at the listed positions of `x`, in list
  order (`torch.index_add`). This is what a relative position table (T5, Swin) or message passing
  over a fixed graph needs. An index **computed in the graph** is not expressible (see below).

Their cost and memory estimates follow the operator: two FLOPs per multiply-add for a transposed
convolution, one per element read for a reduction, the mask and the difference kept by
`ReduceMax`, the interpolation steps of a bilinear `Resize`.

Measured GPU/CPU deviation of the gradient (RTX 5060 Ti, Vulkan; `tests/gpu.rs`), admitted by the
validator at 1e-5 absolute + 1e-3 relative:

| Operator | max abs deviation | max rel deviation |
|---|---|---|
| ConvTranspose2d | 7.6e-6 | 6.4e-4 |
| Pad (reflect / constant / edge) | 1.1e-6 / 3.0e-8 / 3.0e-8 | 3.0e-5 / 5.8e-7 / 6.9e-7 |
| Resize (nearest / linear / aligned) | 1.4e-6 / 1.9e-6 / 4.2e-7 | 4.8e-4 / 5.1e-4 / 1.0e-3 |
| Flip, CumSum | 1.5e-8, 3.0e-8 | 7.3e-7, 1.8e-6 |
| ReduceSum, ReduceMax, ReduceMin, ReduceProd | ≤ 3.0e-8 | ≤ 1.1e-5 |
| IndexSelect, IndexAdd | ≤ 6.0e-8 | ≤ 8.5e-6 |
| `Formula`, the fifteen new functions together | 2.9e-6 | within the tolerance; rounding functions identical byte for byte |

The image cases go through a convolution and a large flattening, which dominate their deviation.

### Integer tensors (`gg-ops/8`)

Every tensor has a type, `f32` or `i64`, inferred by validation like its shape. Integers come from
the i64 inputs and from four operators; they go to `Gather`, `GatherElements`, `ScatterElements`,
`Cast` and the shape operators (`Reshape`, `Transpose`, `Slice`, `Concat`, `Flatten`). Feeding one
to arithmetic is rejected, naming the tensor.

```json
{"op": "ArgTopK",         "inputs": ["scores"],             "output": "idx",  "attrs": {"axis": 2, "k": 2}},
{"op": "GatherElements",  "inputs": ["scores", "idx"],      "output": "top",  "attrs": {"axis": 2}},
{"op": "Softmax",         "inputs": ["top"],                "output": "w",    "attrs": {"axis": 2}},
{"op": "ScatterElements", "inputs": ["zeros", "idx", "w"],  "output": "gate", "attrs": {"axis": 2, "reduction": "none"}}
```

- **Values and indices are two nodes.** The values of a top-k, a sort or a `max(dim)` are
  `GatherElements(x, indices)`: the gradient reaches the selected positions only, as in PyTorch. A
  node keeps a single output; a description without integers is unchanged.
- **Rules that do not depend on the machine.** `ArgMax`/`ArgMin` take the **first** index among
  ties; `ArgTopK` and `ArgSort` are **stable** (ties keep the lower index first); a NaN counts as the
  largest value. Given the same values, CPU and GPU choose the same indices, byte for byte.
- **Integers have no gradient.** `Cast` turns one into f32 (exact: every integer is below 2²⁴).
- **Ranges.** Validation knows the range of every computed index and rejects one that does not fit
  the axis it indexes. An index that comes from the data (an i64 input, the cross-entropy target)
  is checked when the micro-batch is loaded, with a named error: `'ids': value 70000 at position 12
  is outside [0, 64000)`. Nothing is clamped.
- **Scatter.** `reduction: "add"` accumulates. `"none"` writes, and is accepted only when the
  indices are known to be distinct along the axis — the output of `ArgTopK` or `ArgSort` along that
  axis, as in a gating: with repeated positions, the result would depend on the order of the writes.
- **Batch.** No integer operator acts along the batch axis; `Gather` reads the rows of a table
  without batch axis (a parameter, such as the weights of experts `[experts, out, in]`).
- **Stages.** An integer tensor never crosses a stage boundary; the partition does not cut there.
- **Cost.** `ArgTopK`/`ArgSort` sort each line on the host (one read-back per node on the GPU);
  an integer counts 8 bytes in the memory estimate.

Measured GPU/CPU deviation of the gradient through these operators (RTX 5060 Ti, `tests/gpu.rs`):
at most 6.0e-8 absolute, 2.3e-6 relative; the indices are identical, ties and NaNs included; on
8192 tokens routed to 2 of 16 experts, no choice differs. A difference of rounding upstream can
still decide a near-tie differently on another machine: the validator then sees a different
gradient, and a disagreement is settled by a human (ADR-0029).

### The causal mask

`CausalMask` takes attention scores `[batch, …, T, T]` and masks those where the key comes **after**
the query. It sits between the query-key product and the `Softmax`:

```json
{"op": "MatMul",     "inputs": ["q", "kT"],     "output": "scores"},
{"op": "CausalMask", "inputs": ["scores"],      "output": "masked"},
{"op": "Softmax",    "inputs": ["masked"],      "output": "weights", "attrs": {"axis": 3}}
```

Without it, a model that predicts the next token **reads the answer**: on randomly drawn tokens, it
goes below the best possible loss. With it, changing a token does not modify any position before
it, down to the byte, on CPU as on GPU. Details and measurements in [gg-server/docs/adr/0020-causal-mask.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0020-causal-mask.md).

### Positional encoding

Three encodings, without a specialized operator ([gg-server/docs/adr/0021-positional-encoding.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0021-positional-encoding.md)):

- **learned positions**: a `[T, d]` parameter added to the embedding with `Add`, preceded by a
  `Slice` if the table is longer than the sequence;
- **sinusoidal table**: a frozen parameter `{"sinusoidal": {"base": 10000}}` added to the embedding;
- **RoPE**: two frozen tables `{"rotary_cos": …}` and `{"rotary_sin": …}` of shape
  `[1, 1, T, d_head]` and the composition `x·cos + [−x₂, x₁]·sin` (two `Slice`, one `Mul` by −1,
  one `Concat`, two `Mul`, one `Add`), applied to queries and keys.

A position table has the shape `[1, …, 1, T, d]`, with `d` even; its base is finite and greater
than 1. It is computed in f32 as in the reference models. Over 2,048 positions, about 95 % of the
values are byte-identical to those of `transformers`, and the difference is one unit in the last
place for the usual bases; it reaches 1.5e-5 for some bases, where Rust's f32 power and `torch`'s
differ by one ulp. A captured model does not depend on this: its checkpoint carries the module's
exact values.

Why it is necessary: on a task that depends on order (copying the token two positions earlier), a
causal model without encoding plateaus at a loss of 1.36, versus 0.001 with learned positions and
0.25 with RoPE. The mask alone provides part of the order, not the task.

### Scaled RoPE

Large open models extend their context by changing the RoPE frequencies: Llama 3.1 (`llama3`),
Qwen 2.5 and DeepSeek (`yarn`), Phi-3 (`longrope`), `linear` interpolation, `dynamic` NTK. Each one
is just a way of computing the frequencies, plus, for YaRN and LongRoPE, an amplitude that
multiplies cosine and sine. The description therefore carries **the frequencies themselves**, not
the name of the variant ([gg-server/docs/adr/0022-scaled-rope.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0022-scaled-rope.md)):

```json
{"name": "rope.cos", "shape": [1, 1, 4096, 128], "trainable": false,
 "init": {"rotary_cos": {"frequencies": [1.0, 0.8058, …], "amplitude": 1.1386}}}
```

One frequency per pair of dimensions, finite and non-negative; an amplitude that is finite and
strictly positive, 1 by default. `{"base": b}` remains the form of the original law; the two forms
are mutually exclusive. The executor has a single formula, `A·cos(t·ω)`, and knows no variant by
name: the next published scaling will require nothing from the platform.

Three ways to obtain these frequencies:

- **the PyTorch driver** recovers them from the value of the module's tables, whatever the variant
  ([gg-python/docs/PYTHON.md](https://github.com/GPU-Goblin/gg-python/blob/main/docs/PYTHON.md));
- **the skeleton** computes them from a `config.json`: `gg job template transformer --rope-scaling
  '{"rope_type": "llama3", "factor": 8, "low_freq_factor": 1, "high_freq_factor": 4,
  "original_max_position_embeddings": 8192}' --rope-base 500000`;
- **by hand**, for those who build a graph programmatically: `graph::rope::rotary` in Rust.

### Custom operator: `Formula`

An element-wise function missing from the table is written as an expression, without code:

```json
{"op": "Formula", "inputs": ["h", "g"], "output": "act",
 "attrs": {"expr": "a * tanh(softplus(a)) + 0.1 * b"}}
```

The inputs are called `a`, `b`, `c`, `d` in the node's order (one to four, all used) and broadcast
as for `Add`. The grammar is closed:

- `+ - * /`, unary minus, parentheses, decimal numbers;
- power with a **literal** exponent: `a^3`, `a^-0.5` (an integer power is exact, even for a
  negative base);
- functions: `exp`, `log`, `log1p`, `sqrt`, `rsqrt`, `abs`, `tanh`, `sigmoid`, `relu`, `sin`, `cos`,
  `erf`, `softplus` (stable form), `max(x, y)`, `min(x, y)`; since `gg-ops/7`, `tan`, `asin`,
  `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `atan2(y, x)`, `floor`, `ceil`,
  `round` (half to even, as PyTorch, on every backend), `trunc`, `sign`.

`floor`, `ceil`, `round`, `trunc` and `sign` are piecewise constant: their derivative is zero
wherever it exists (and undefined at the jumps), and automatic differentiation returns zero, as
PyTorch does. A formula that uses a function of `gg-ops/7` is only offered to workers at that level.

Bounds: 1024 bytes, 128 terms, 32 levels of nesting. The derivative is the one from automatic
differentiation, on CPU as on GPU; the cost (FLOPs and memory) is read from the expression. An error
is named with its position: `expression, position 4: expected ')'`. From PyTorch, this is
`gpugoblin.formula` ([gg-python/docs/PYTHON.md](https://github.com/GPU-Goblin/gg-python/blob/main/docs/PYTHON.md)). Evidence: ADR-0025.

### Functions: named blocks, called like operators

A repeated block — a layer, a hand-written normalization, a cell — is declared once in `functions`
and called with `fn:<name>`:

```json
"functions": [
  {"name": "layer", "inputs": ["h", "w", "b"], "output": "a",
   "nodes": [
     {"op": "MatMul",  "inputs": ["h", "w"], "output": "m"},
     {"op": "Add",     "inputs": ["m", "b"], "output": "z"},
     {"op": "Formula", "inputs": ["z"],      "output": "a", "attrs": {"expr": "a * sigmoid(a)"}}
   ]}
],
"nodes": [
  {"op": "fn:layer", "inputs": ["x",  "w1", "b1"], "output": "h1"},
  {"op": "fn:layer", "inputs": ["h1", "w2", "b2"], "output": "h2"}
]
```

A function is **closed**: its body only reads its inputs and what it produces, and parameters are
passed to it at the call. It can call the functions declared before it, never itself. The server
**inlines** the calls at submission (the inner tensors of `h1` are named `h1/m`, `h1/z`): workers,
validator and plan only see a flat description, and the submitted description remains the source
of execution. `gg job check-model` inlines in the same way. Bounds: 64 functions, 32 inputs per
function, 4096 nodes once inlined.

### Model constants

A scalar — the 1/√d factor of an attention — is declared as a frozen parameter with a constant
initialization, so that the description computes the right function without a checkpoint:

```json
{"name": "scale", "shape": [1], "init": {"constant": {"value": 0.125}}, "trainable": false}
```

`*` required attribute. The "reproducibility" column says what the validator can require: a graph
that only uses bit-for-bit operators is recomputed down to the byte; otherwise verification is done
within the numeric profile's tolerance, as is done today for `tensor.matmul.v1`.

**An operator missing from the profile makes the description rejected.** It is not approximated,
replaced or ignored. Asking for it to be added is an explicit request: every operator added comes
with its implementation, its verification oracle, its determinism test, its tolerance, its FLOP
estimator and its memory estimator.

## ONNX

ONNX has nearly two hundred operators and several opset versions. The platform does not "support
ONNX": it supports `gg-ops/8`, and ONNX import will be a converter restricted to this profile,
which rejects with the name of the out-of-profile operator. The ONNX column of the table says what
each operator corresponds to, so that exporting from a familiar tool is predictable.

## Bundled examples

`examples/models/` contains the four descriptions used by the tests, and therefore always valid:

| File | What it describes | Parameters |
|---|---|---|
| `mlp-tabular.json` | perceptron 64 → 256 → 128 → 10 | 50,826 |
| `cnn-cifar.json` | two convolution + max-pool blocks, 10 classes | 60,266 |
| `transformer-block.json` | embedding, single-head attention, feed-forward network, language head | 17,170,432 |
| `series-horizon-1.json` | window of 48 steps, 7 channels, horizon-1 regression | 2,528 |

## Submitting a training run

```bash
gg job collective --project <id> --graph my-model.json \
   --dataset x=train_x.ggt --dataset y=train_y.ggt \
   --optimizer adam --lr 0.02 --batch 256 --tasks 4 --rounds 20
gg job collective-status <run>
gg job collective-check <run>          # replays the run in one process and compares with the checkpoint
gg job collective-checkpoint <run> --out weights.ggb
```

The compute type is `graph.grad.v1`: each task computes the mean gradient of an explicit
micro-batch from the current checkpoint, and the server aggregates once per round and publishes the
next checkpoint (ADR-0011, generalized). The checkpoint is a `GGB1` container of all the named
parameters, which `gg data inspect` reads.

`gg job collective-check` downloads the description, the data and the published checkpoint, replays
the training in a single process and compares. The researcher has nothing to keep locally to
verify the grid's work.

## What is not there yet

The implementation status is in [gg-server/docs/STATUS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/STATUS.md). A described model really trains on the grid today.
What is missing:

- **Interleaved RoPE** (GPT-J), **multimodal RoPE** (Qwen2-VL), **ALiBi** and **relative position
  biases** are not translated. Scaled RoPE is (ADR-0022).
- **Constant non-causal masks** (sliding window, arbitrary mask) are not translated either. An
  additive mask computed in the graph (padding) and an empty mask are (ADR-0025).
- **Shapes that depend on the data** (`nonzero`, `torch.where` with one argument, boolean masking,
  `unique`), **assignments by index** (`index_put`), `one_hot` and **integer arithmetic** are
  outside the profile. Integer tensors computed in the graph are in since `gg-ops/8` (ADR-0029).
- **ONNX import** does not exist; the entry point is `torch.export` ([gg-python/docs/PYTHON.md](https://github.com/GPU-Goblin/gg-python/blob/main/docs/PYTHON.md)).
