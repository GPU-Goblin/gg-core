# burn-tch 0.21.0, patched for GPU Goblin

Copy of `burn-tch` 0.21.0 (MIT OR Apache-2.0, https://github.com/tracel-ai/burn), used through
`[patch.crates-io]` by the native CUDA engine (`gg-compute`, feature `cuda`).

One change, in `src/tensor.rs` (`TchTensor::can_mut`): a tensor that PyTorch's autograd tracks
(`requires_grad`) is never modified in place. The native CUDA engine runs the executor's operators
on this backend with PyTorch's autograd instead of Burn's; burn-tch otherwise reuses the storage of
a uniquely owned tensor for the next operation, and PyTorch's backward pass then refuses the graph
("a variable needed for gradient computation has been modified by an inplace operation").
