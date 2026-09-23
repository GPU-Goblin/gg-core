//! Engine selection and reporting. GPU support is compiled behind the `gpu` feature and requires an
//! explicit device; there is never a silent CPU fallback.

use gg_protocol::messages::ExecutionProfileV1;

use crate::error::ComputeError;
use crate::profiles;

/// Which engine to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineSelection {
    /// Burn Flex CPU backend.
    FlexCpu,
    /// Burn WGPU backend with Vulkan on the discrete GPU at this index.
    WgpuVulkan {
        /// Discrete GPU index as enumerated by wgpu.
        device_index: usize,
    },
    /// Native CUDA engine: LibTorch with PyTorch's autograd (ADR-0033), NVIDIA only.
    TorchCuda {
        /// CUDA device index.
        device_index: usize,
    },
}

impl EngineSelection {
    /// Parse `cpu`, `gpu`, `gpu:<index>`, `cuda`, `cuda:<index>`.
    pub fn parse(s: &str) -> Result<Self, ComputeError> {
        match s {
            "cpu" | "flex" => Ok(Self::FlexCpu),
            "gpu" | "vulkan" => Ok(Self::WgpuVulkan { device_index: 0 }),
            "cuda" => Ok(Self::TorchCuda { device_index: 0 }),
            other if other.starts_with("cuda:") => {
                let device_index = other["cuda:".len()..].parse::<usize>().map_err(|_| {
                    ComputeError::EngineUnavailable(format!("bad engine spec '{other}'"))
                })?;
                Ok(Self::TorchCuda { device_index })
            }
            other => {
                if let Some(idx) = other
                    .strip_prefix("gpu:")
                    .or_else(|| other.strip_prefix("vulkan:"))
                {
                    let device_index = idx.parse::<usize>().map_err(|_| {
                        ComputeError::EngineUnavailable(format!("bad engine spec '{other}'"))
                    })?;
                    Ok(Self::WgpuVulkan { device_index })
                } else {
                    Err(ComputeError::EngineUnavailable(format!(
                        "unknown engine '{other}'"
                    )))
                }
            }
        }
    }

    /// Numeric profile produced by this engine.
    pub const fn numeric_profile(self) -> &'static str {
        match self {
            Self::FlexCpu => profiles::CPU_F32_FLEX_V1,
            Self::WgpuVulkan { .. } => profiles::GPU_F32_WGPU_VULKAN_V1,
            Self::TorchCuda { .. } => profiles::GPU_F32_LIBTORCH_CUDA_V1,
        }
    }

    /// Engine identifier.
    pub const fn engine_name(self) -> &'static str {
        match self {
            Self::FlexCpu => "burn-flex",
            Self::WgpuVulkan { .. } => "burn-wgpu-vulkan",
            Self::TorchCuda { .. } => "libtorch-cuda",
        }
    }
}

/// Information about the engine that actually ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineInfo {
    /// Engine identifier.
    pub engine: &'static str,
    /// Numeric profile.
    pub numeric_profile: &'static str,
    /// Device name as reported locally (not attested).
    pub device_name: String,
}

impl EngineInfo {
    /// Build the declared execution profile for a submission.
    pub fn to_profile(&self, duration_ms: u64) -> ExecutionProfileV1 {
        ExecutionProfileV1 {
            engine: self.engine.to_owned(),
            engine_version: crate::ENGINE_VERSION.to_owned(),
            numeric_profile: self.numeric_profile.to_owned(),
            device_name: self.device_name.chars().take(128).collect(),
            duration_ms,
        }
    }
}

/// CPU engine info.
pub fn cpu_info() -> EngineInfo {
    EngineInfo {
        engine: EngineSelection::FlexCpu.engine_name(),
        numeric_profile: EngineSelection::FlexCpu.numeric_profile(),
        device_name: format!(
            "cpu ({} threads)",
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        ),
    }
}

/// GPU support.
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::Vulkan, init_setup};

    /// Initialize the Vulkan device at `device_index` and return its info. Fails (does not fall
    /// back) if the adapter is a CPU implementation (e.g. llvmpipe) or is not Vulkan.
    ///
    /// `init_setup` panics when no adapter exists; callers run this in a dedicated process and
    /// treat a panic as an explicit failure (see gg-worker `doctor` / `probe-gpu`).
    ///
    /// Initialized once per process and device: CubeCL refuses a second initialization of the same
    /// device, and a validator recomputes many tasks in one process (ADR-0031), keeping its
    /// compiled kernels from one to the next.
    pub fn init_vulkan(device_index: usize) -> Result<(WgpuDevice, EngineInfo), ComputeError> {
        use std::collections::BTreeMap;
        use std::sync::Mutex;
        static READY: Mutex<BTreeMap<usize, (WgpuDevice, EngineInfo)>> =
            Mutex::new(BTreeMap::new());
        let mut ready = READY.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = ready.get(&device_index) {
            return Ok(r.clone());
        }
        let r = init_vulkan_once(device_index)?;
        ready.insert(device_index, r.clone());
        Ok(r)
    }

    fn init_vulkan_once(device_index: usize) -> Result<(WgpuDevice, EngineInfo), ComputeError> {
        let device = WgpuDevice::DiscreteGpu(device_index);
        let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
        let info = setup.adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(ComputeError::EngineUnavailable(format!(
                "adapter '{}' is a CPU implementation, refusing to report it as a GPU",
                info.name
            )));
        }
        if info.backend != wgpu::Backend::Vulkan {
            return Err(ComputeError::EngineUnavailable(format!(
                "adapter '{}' uses backend {:?}, expected Vulkan",
                info.name, info.backend
            )));
        }
        Ok((
            device,
            EngineInfo {
                engine: EngineSelection::WgpuVulkan { device_index }.engine_name(),
                numeric_profile: EngineSelection::WgpuVulkan { device_index }.numeric_profile(),
                device_name: format!(
                    "{} ({:?}, driver {})",
                    info.name, info.device_type, info.driver
                ),
            },
        ))
    }

    /// Keep the compiled GPU kernels in `dir`, from one process to the next (ADR-0032). Each task
    /// runs in a fresh executor, which otherwise compiled its kernels again: 5.1 s of a 7.2 s
    /// ViT-tiny task, 0.84 s instead of 6.0 s for the first call with the cache. Must be called
    /// before the GPU is initialized; fails if the configuration was already read.
    pub fn set_kernel_cache(dir: &std::path::Path) -> Result<(), ComputeError> {
        use cubecl_runtime::config::cache::CacheConfig;
        use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
        std::fs::create_dir_all(dir).map_err(|e| {
            ComputeError::EngineUnavailable(format!("kernel cache {}: {e}", dir.display()))
        })?;
        let mut config = CubeClRuntimeConfig::default();
        config.compilation.cache = Some(CacheConfig::File(dir.to_path_buf()));
        std::panic::catch_unwind(|| CubeClRuntimeConfig::set(config)).map_err(|_| {
            ComputeError::EngineUnavailable(
                "the GPU configuration was already read: the kernel cache must be set first".into(),
            )
        })
    }

    /// Give the device memory a computation used back to the driver, once its results are on the
    /// host. CubeCL keeps its memory pool at its peak: a long-lived process (a GPU validator,
    /// ADR-0031) otherwise held 6.9 GB after recomputing a staged SmolLM2, and a worker sharing the
    /// card failed for lack of memory.
    pub fn release_memory(device: &WgpuDevice) {
        use burn::tensor::backend::Backend;
        <burn::backend::Vulkan as Backend>::memory_cleanup(device);
    }

    /// Enumerate Vulkan adapters without initializing a compute runtime (for `worker doctor`).
    pub fn list_vulkan_adapters() -> Vec<String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        adapters
            .into_iter()
            .map(|a| {
                let i = a.get_info();
                format!(
                    "{} [{:?}] driver={} {}",
                    i.name, i.device_type, i.driver, i.driver_info
                )
            })
            .collect()
    }
}

/// Native CUDA engine (ADR-0033): LibTorch with PyTorch's autograd.
#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;

    /// NVIDIA libraries must not compute in TF32: strict float32 for the whole process.
    pub const TF32_OVERRIDE: (&str, &str) = ("NVIDIA_TF32_OVERRIDE", "0");

    /// Check that CUDA device `index` is usable in strict float32 and describe it. Fails, never
    /// falls back.
    pub fn check(index: usize) -> Result<EngineInfo, ComputeError> {
        if std::env::var(TF32_OVERRIDE.0).as_deref() != Ok(TF32_OVERRIDE.1) {
            return Err(ComputeError::EngineUnavailable(format!(
                "the native CUDA engine computes in strict float32: run it with {}={}",
                TF32_OVERRIDE.0, TF32_OVERRIDE.1
            )));
        }
        if !tch::Cuda::is_available() {
            return Err(ComputeError::EngineUnavailable(
                "CUDA is not available to LibTorch (driver, libraries or device access)".into(),
            ));
        }
        let count = tch::Cuda::device_count();
        if index as i64 >= count {
            return Err(ComputeError::EngineUnavailable(format!(
                "CUDA device {index} requested, {count} present"
            )));
        }
        // cuDNN tries its algorithms on the first call for each tensor shape and keeps the
        // fastest: other float32 algorithms, not a lower precision (TF32 stays off). In PyTorch it
        // takes ConvNeXt-tiny from 219 to 165 ms per step; the executor already lands near the
        // faster algorithms, and gains 4 % (two stages' backward passes, 340 → 326 ms). The trial
        // costs about 20 ms once per shape, per executor session.
        tch::Cuda::cudnn_set_benchmark(true);
        let selection = EngineSelection::TorchCuda {
            device_index: index,
        };
        Ok(EngineInfo {
            engine: selection.engine_name(),
            numeric_profile: selection.numeric_profile(),
            device_name: format!("CUDA device {index} (LibTorch {})", LIBTORCH_VERSION),
        })
    }

    /// The LibTorch version the engine is built against.
    pub const LIBTORCH_VERSION: &str = "2.9.0";
}

/// Refusal for a kind the native CUDA engine does not run.
pub fn cuda_models_only() -> ComputeError {
    ComputeError::EngineUnavailable(
        "the native CUDA engine runs model tasks only (graph.grad.v1, graph.stage.v1)".into(),
    )
}
