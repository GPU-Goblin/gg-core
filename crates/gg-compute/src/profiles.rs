//! Numeric profiles. A profile names the backend family and precision; it is admitted only after
//! real measurements (see gg-server/docs/VALIDATION.md and gg-server/docs/STATUS.md).

/// Burn Flex CPU, f32.
pub const CPU_F32_FLEX_V1: &str = "cpu-f32-flex-v1";
/// Burn WGPU with the Vulkan/SPIR-V compiler, f32.
pub const GPU_F32_WGPU_VULKAN_V1: &str = "gpu-f32-wgpu-vulkan-v1";
/// Native CUDA engine: LibTorch 2.9 with PyTorch's autograd, strict float32 (ADR-0033).
pub const GPU_F32_LIBTORCH_CUDA_V1: &str = "gpu-f32-libtorch-cuda-v1";
/// Profile used by validators to compute references.
pub const REFERENCE_PROFILE: &str = CPU_F32_FLEX_V1;
/// Requested profile family accepted in task definitions.
pub const REQUESTED_F32_V1: &str = "f32-v1";

/// Whether a validator may recompute the reference of `kind` with `profile` (ADR-0031). The CPU
/// profile everywhere; the GPU profile for model tasks only, measured as accurate as the CPU
/// against float64 (ResNet18: 3.9e-3 median relative error, CPU 2.7e-3, PyTorch FP32 CUDA 5.3e-3).
/// A GPU reference shares the GPU workers' software: an engine bug common to both would pass it,
/// which is what the CPU engine check (`engine_check` verifications) is there to catch.
pub fn reference_admitted_for(kind: &str, profile: &str) -> bool {
    profile == REFERENCE_PROFILE
        || (matches!(kind, "graph.grad.v1" | "graph.stage.v1")
            && (profile == GPU_F32_WGPU_VULKAN_V1
                || (GRAPH_CUDA_ADMITTED && profile == GPU_F32_LIBTORCH_CUDA_V1)))
}

/// Profiles admitted for acceptance. The GPU profile was admitted after `tests/gpu.rs` measured the
/// deviation against the CPU reference on the identified hardware (gg-server/docs/STATUS.md).
pub const ADMITTED: &[&str] = &[
    CPU_F32_FLEX_V1,
    GPU_F32_WGPU_VULKAN_V1,
    GPU_F32_LIBTORCH_CUDA_V1,
];

/// Whether a declared execution profile may lead to acceptance (any kind).
pub fn is_admitted(profile: &str) -> bool {
    ADMITTED.contains(&profile)
}

/// Per-kind admission. `mlp.train.v1` admits the GPU profile only once the training deviation has
/// been measured on identified hardware (gg-server/docs/BENCHMARKS.md); until then a GPU-trained result stays
/// `inconclusive`.
pub fn admitted_for(kind: &str, profile: &str) -> bool {
    match kind {
        // The native CUDA engine runs model tasks only (ADR-0033).
        "tensor.matmul.v1" => profile == CPU_F32_FLEX_V1 || profile == GPU_F32_WGPU_VULKAN_V1,
        "mlp.train.v1" => {
            profile == CPU_F32_FLEX_V1 || (MLP_GPU_ADMITTED && profile == GPU_F32_WGPU_VULKAN_V1)
        }
        "mlp.grad.v1" => {
            profile == CPU_F32_FLEX_V1 || (GRAD_GPU_ADMITTED && profile == GPU_F32_WGPU_VULKAN_V1)
        }
        "graph.grad.v1" | "graph.stage.v1" => {
            profile == CPU_F32_FLEX_V1
                || (GRAPH_GPU_ADMITTED && profile == GPU_F32_WGPU_VULKAN_V1)
                || (GRAPH_CUDA_ADMITTED && profile == GPU_F32_LIBTORCH_CUDA_V1)
        }
        // The comparison is the same CPU code whatever the worker's engine.
        "replica.compare.v1" => profile == CPU_F32_FLEX_V1,
        _ => false,
    }
}

/// NOT admitted (2026-09-16). `tests/gpu.rs::mlp_training_gpu_vs_cpu` on an RTX 5060 Ti measured the
/// demo grid (12 Adam configurations + 1 SGD, 300 steps): 12/13 runs deviate by < 3e-6 on the final
/// weights, but one (Adam, lr 0.02, hidden 64, seed 2) diverges to 6.9e-2 (a ReLU boundary flip
/// amplified by the optimizer) while its evaluation loss stays within 5e-5. A CPU replay therefore
/// cannot certify GPU-trained weights under a tight tolerance; GPU results stay `inconclusive`
/// until a same-profile (GPU) validator exists (gg-server/docs/BENCHMARKS.md, gg-server/docs/VALIDATION.md).
pub const MLP_GPU_ADMITTED: bool = false;

/// Admitted on 2026-09-17: `tests/gpu.rs::graph_ops_on_gpu` measured every operator of `gg-ops/1`
/// on an RTX 5060 Ti against the Flex CPU reference (max abs deviation 1.2e-5, on Gelu), and
/// `tests/gpu.rs::graph_on_gpu` a CNN (bitwise identical loss, max abs 0) and a two-layer
/// transformer (max abs 1.2e-5 on an embedding gradient). The GPU is also bitwise reproducible
/// with itself on both. Under the `graph.grad.v1` rule (by norm since catalog 2, ADR-0031) a CPU
/// validator accepts all of them, and the gradients of 12 described models (ResNet18, a
/// transformer, speech models, mixtures of experts); the tests assert it, so a regression flips
/// this back (gg-server/docs/BENCHMARKS.md).
pub const GRAPH_GPU_ADMITTED: bool = true;

/// Admitted on 2026-09-25 (ADR-0033): the native CUDA engine in strict float32
/// (`NVIDIA_TF32_OVERRIDE=0`), measured by `tests/cuda.rs` against the CPU reference under the
/// model-task rule (by norm, 2e-2): every trained tensor of ResNet18, ViT-tiny, ConvNeXt-tiny and
/// SmolLM2-135M within it (worst 5.9e-3, ResNet18), and the fixtures' gradients against PyTorch's.
/// With TF32 allowed, cuDNN put ResNet18 7 % away: the engine refuses to run without the override.
pub const GRAPH_CUDA_ADMITTED: bool = true;

/// Admitted on 2026-09-16: `tests/gpu.rs::grad_gpu_vs_cpu` on an RTX 5060 Ti measured bitwise
/// identical micro-batch gradients (max abs diff 0) and losses within 1.4e-7 on 4 configurations;
/// a single gradient has no trajectory to diverge (gg-server/docs/BENCHMARKS.md).
pub const GRAD_GPU_ADMITTED: bool = true;

#[cfg(test)]
mod tests {
    use super::*;

    /// A validator may recompute on the GPU only model tasks (ADR-0031); anything else, and any
    /// unknown profile, only on the CPU reference.
    #[test]
    fn the_gpu_is_a_reference_for_model_tasks_only() {
        for kind in ["graph.grad.v1", "graph.stage.v1"] {
            assert!(reference_admitted_for(kind, REFERENCE_PROFILE));
            assert!(reference_admitted_for(kind, GPU_F32_WGPU_VULKAN_V1));
            assert!(!reference_admitted_for(kind, "gpu-anything"));
        }
        for kind in [
            "tensor.matmul.v1",
            "mlp.train.v1",
            "mlp.grad.v1",
            "unknown.v1",
        ] {
            assert!(reference_admitted_for(kind, REFERENCE_PROFILE));
            assert!(!reference_admitted_for(kind, GPU_F32_WGPU_VULKAN_V1));
        }
    }
}
