# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — unreleased

First public version of the shared crates of GPU Goblin.

- `gg-core`: identifiers, states and transitions, integer credits, invariants.
- `gg-protocol`: DAG-CBOR canonical encoding, domain-separated hashes, Ed25519 signatures,
  versioned messages, `GGT1`/`GGB1` tensor formats, deterministic generator, replayable test
  vectors.
- `gg-compute`: approved computation catalog (matrix product, perceptron training, gradients) and
  models described as `gg-graph/v1` graphs: execution on CPU
  and WGPU/Vulkan, staged execution, automatic tiling, sharded checkpoints, memory and
  distribution planning.
- `gg-compute`, operator profile `gg-ops/7` (ADR-0027): ConvTranspose2d, Pad, Resize, Flip,
  ReduceSum/Max/Min/Prod, CumSum, IndexSelect/IndexAdd, and fifteen more `Formula` functions, each
  with an f64 oracle, a determinism test, estimators and a measured GPU tolerance.
- `gg-compute`, operator profile `gg-ops/8` (ADR-0029): integer tensors computed inside the graph
  (ArgMax, ArgMin, ArgTopK, ArgSort, Cast, GatherElements, ScatterElements), identical indices on CPU
  and GPU; mixtures of experts supported; `Gelu(approximate="tanh")` computed with the tanh form.
- `gg-compute`: host-memory limits of model tasks derived from measurement (a stage task is sized
  from its own parameters, not the whole model's); stage partitioning derives each candidate stage
  from an index built once, 108.6 s → 0.4 s for a 1 900-node model, same bounds.
- `gg-compute`: a convolution or transposed convolution whose kernel is shorter than its stride is
  rewritten into an exact equivalent with stride equal to the kernel: Burn's WGPU kernels compute
  it wrongly (a ResNet's 1×1 stride-2 shortcut had a wrong gradient on GPU). A stage task's output
  budget is computed from its boundary tensors.
- `gg-compute`, catalog 2 (ADR-0031): model tasks are compared per tensor by the norm of the
  difference (`recompute-reference` v2), calibrated against float64; `verify_task_on` recomputes a
  model task on the GPU for a GPU validator; Vulkan is initialized once per process and the GPU
  memory pool is released after each computation.
- `gg-compute`, executor: a zero-rate Dropout is the identity (its mask was drawn on the host,
  275 of ViT-tiny's 300 ms forward pass), an unsegmented step runs its forward pass once, and
  parameters, datasets and gradients move to and from the device without host copies; bit-exact.
  A ViT-tiny gradient goes from 800 to 280 ms on Vulkan. `gpu::set_kernel_cache` (ADR-0032).
- `gg-compute`, native CUDA engine (feature `cuda`, ADR-0033): the executor's operators on LibTorch
  2.9 with PyTorch's autograd, strict float32; admitted for model tasks. A ConvNeXt-tiny gradient
  goes from 2360 ms (Vulkan) to 270 ms. `vendor/burn-tch`: burn-tch 0.21 with one change (no
  in-place write into tensors PyTorch's autograd tracks).
- `gg-protocol`: release manifests also signed as an OpenSSH `SSHSIG` over `SHA256SUMS`, verifiable
  with `ssh-keygen -Y verify` and `sha256sum -c`.
- `gg-client`: client of the control plane shared by the worker, the validator and the CLI; plain
  http is refused to another machine (tokens travel on it); SIGTERM and Ctrl-C handling for
  long-running processes.

The full history of the project, before it was split into several repositories, is summarized in
[gg-server/CHANGELOG.md](https://github.com/GPU-Goblin/gg-server/blob/main/CHANGELOG.md).
