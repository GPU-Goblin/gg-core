//! `tensor.matmul.v1`: execution on the selected engine and verification by reference recomputation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::messages::{ArtifactRef, CheckV1, TaskParamsV1};
use gg_protocol::tensor_file::TensorF32;

use crate::catalog::{ValidatedTask, tensor_limits_for};
use crate::compare::compare_f32;
use crate::engine::{EngineInfo, EngineSelection, cpu_info};
use crate::error::ComputeError;

/// A produced output file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputFile {
    /// Path on disk.
    pub path: PathBuf,
    /// SHA-256 of the file bytes.
    pub hash: Sha256Digest,
    /// Size in bytes.
    pub size: u64,
}

impl OutputFile {
    /// Artifact reference.
    pub fn artifact_ref(&self) -> ArtifactRef {
        ArtifactRef {
            hash: self.hash,
            size: self.size,
        }
    }
}

/// Result of an execution.
#[derive(Clone, Debug)]
pub struct ExecutionReport {
    /// Outputs by name.
    pub outputs: BTreeMap<String, OutputFile>,
    /// Engine that ran.
    pub engine: EngineInfo,
    /// Wall-clock duration in milliseconds (declared, not attested).
    pub duration_ms: u64,
    /// Declared metrics (never used for acceptance).
    pub metrics: BTreeMap<String, f64>,
}

/// Verdict of a verification.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// Checks performed.
    pub checks: Vec<CheckV1>,
    /// Outcome.
    pub outcome: VerificationOutcome,
    /// Artifacts whose bytes were checked.
    pub artifacts_checked: Vec<ArtifactRef>,
}

fn to_tensor<B: Backend>(device: &B::Device, t: &TensorF32) -> Tensor<B, 2> {
    let shape = [t.shape[0] as usize, t.shape[1] as usize];
    Tensor::<B, 2>::from_data(TensorData::new(t.data.clone(), shape), device)
}

/// Matrix product on any Burn backend, returning a decoded f32 tensor (synchronizes the device).
pub fn matmul_on<B: Backend>(
    device: &B::Device,
    a: &TensorF32,
    b: &TensorF32,
) -> Result<TensorF32, ComputeError> {
    let c = to_tensor::<B>(device, a).matmul(to_tensor::<B>(device, b));
    let data = c.into_data();
    let shape: Vec<u64> = data.shape.iter().map(|d| *d as u64).collect();
    let values = data.to_vec::<f32>().map_err(|e| {
        ComputeError::EngineUnavailable(format!("backend returned unexpected dtype: {e:?}"))
    })?;
    TensorF32::new(shape, values).map_err(|e| ComputeError::EngineUnavailable(e.to_string()))
}

/// Reference computation on the CPU reference profile (Burn Flex).
pub fn matmul_reference(a: &TensorF32, b: &TensorF32) -> Result<TensorF32, ComputeError> {
    let device = burn::backend::flex::FlexDevice;
    matmul_on::<burn::backend::Flex>(&device, a, b)
}

/// Naive f64-accumulated product, used by tests as an independent oracle.
pub fn matmul_naive_f64(a: &TensorF32, b: &TensorF32) -> TensorF32 {
    let (m, k) = (a.shape[0] as usize, a.shape[1] as usize);
    let n = b.shape[1] as usize;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a.data[i * k + p] as f64 * b.data[p * n + j] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    TensorF32::new(vec![m as u64, n as u64], out).expect("shape")
}

fn load_input(
    task: &ValidatedTask,
    name: &str,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<TensorF32, ComputeError> {
    let shape = task
        .input_shapes
        .get(name)
        .ok_or_else(|| ComputeError::Inputs(format!("no expected shape for '{name}'")))?;
    let path = inputs
        .get(name)
        .ok_or_else(|| ComputeError::Inputs(format!("input file '{name}' not provided")))?;
    let t = TensorF32::read_from_path(path, &tensor_limits_for(shape)).map_err(|source| {
        ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        }
    })?;
    t.check_shape(shape)
        .map_err(|source| ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        })?;
    t.check_finite()
        .map_err(|source| ComputeError::InputTensor {
            name: name.to_owned(),
            source,
        })?;
    Ok(t)
}

fn write_output(
    out_dir: &Path,
    name: &str,
    t: &TensorF32,
    max_bytes: u64,
) -> Result<OutputFile, ComputeError> {
    let bytes = t.encode();
    if bytes.len() as u64 > max_bytes {
        return Err(ComputeError::OutputLimit(format!(
            "output '{name}' is {} bytes > {max_bytes}",
            bytes.len()
        )));
    }
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join(format!("{name}.ggt"));
    let tmp = out_dir.join(format!(".{name}.ggt.tmp"));
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(OutputFile {
        path,
        hash: Sha256Digest::of_bytes(&bytes),
        size: bytes.len() as u64,
    })
}

/// Execute a validated matmul task. Inputs are files named by input name; outputs are written to
/// `out_dir/<name>.ggt`.
pub fn execute(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
) -> Result<ExecutionReport, ComputeError> {
    let TaskParamsV1::Matmul { .. } = task.params else {
        return Err(ComputeError::UnknownKind(
            task.params.compute_kind().to_owned(),
        ));
    };
    let a = load_input(task, "a", inputs)?;
    let b = load_input(task, "b", inputs)?;
    let start = Instant::now();
    let (c, info) = match engine {
        EngineSelection::FlexCpu => (matmul_reference(&a, &b)?, cpu_info()),
        #[cfg(feature = "gpu")]
        EngineSelection::WgpuVulkan { device_index } => {
            let (device, info) = crate::engine::gpu::init_vulkan(device_index)?;
            (matmul_on::<burn::backend::Vulkan>(&device, &a, &b)?, info)
        }
        EngineSelection::TorchCuda { .. } => return Err(crate::engine::cuda_models_only()),
        #[cfg(not(feature = "gpu"))]
        EngineSelection::WgpuVulkan { .. } => {
            return Err(ComputeError::EngineUnavailable(
                "this build has no GPU support (feature `gpu`)".into(),
            ));
        }
    };
    let duration_ms = start.elapsed().as_millis() as u64;
    let shape = &task.output_shapes["c"];
    c.check_shape(shape)
        .map_err(|e| ComputeError::OutputLimit(e.to_string()))?;
    let file = write_output(out_dir, "c", &c, task.limits.max_output_bytes)?;
    let mut outputs = BTreeMap::new();
    outputs.insert("c".to_owned(), file);
    let mut metrics = BTreeMap::new();
    metrics.insert("duration_ms".to_owned(), duration_ms as f64);
    metrics.insert("flops".to_owned(), task.estimate.flops as f64);
    Ok(ExecutionReport {
        outputs,
        engine: info,
        duration_ms,
        metrics,
    })
}

fn fail(name: &str, detail: String) -> CheckV1 {
    CheckV1 {
        name: name.to_owned(),
        passed: false,
        measured: None,
        tolerance: None,
        detail: detail.chars().take(200).collect(),
    }
}

/// Verify candidate outputs against a fresh reference computation. Never trusts declared metrics.
/// `candidates` maps output name → file path; `claimed` maps output name → claimed artifact ref.
pub fn verify(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
) -> Result<Verdict, ComputeError> {
    let TaskParamsV1::Matmul { .. } = task.params else {
        return Err(ComputeError::UnknownKind(
            task.params.compute_kind().to_owned(),
        ));
    };
    let mut checks = Vec::new();
    let mut artifacts_checked = Vec::new();

    // 1. Output set and byte integrity.
    let expected_names: Vec<&String> = task.output_shapes.keys().collect();
    let claimed_names: Vec<&String> = claimed.keys().collect();
    if expected_names != claimed_names {
        checks.push(fail(
            "outputs.names",
            format!("expected {expected_names:?}, got {claimed_names:?}"),
        ));
        return Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Rejected,
            artifacts_checked,
        });
    }
    let shape = &task.output_shapes["c"];
    let claimed_c = claimed["c"];
    let path = match candidates.get("c") {
        Some(p) => p,
        None => {
            checks.push(fail("outputs.c.present", "candidate file missing".into()));
            return Ok(Verdict {
                checks,
                outcome: VerificationOutcome::Inconclusive,
                artifacts_checked,
            });
        }
    };
    let max_bytes = crate::catalog::expected_rgt_size(shape);
    let meta = std::fs::metadata(path)?;
    if meta.len() != claimed_c.size || meta.len() > max_bytes {
        checks.push(fail(
            "outputs.c.size",
            format!(
                "claimed {}, file {}, expected {max_bytes}",
                claimed_c.size,
                meta.len()
            ),
        ));
        return Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Rejected,
            artifacts_checked,
        });
    }
    let bytes = std::fs::read(path)?;
    let hash = Sha256Digest::of_bytes(&bytes);
    let hash_ok = hash == claimed_c.hash;
    checks.push(CheckV1 {
        name: "outputs.c.sha256".into(),
        passed: hash_ok,
        measured: None,
        tolerance: None,
        detail: if hash_ok {
            String::new()
        } else {
            "hash mismatch".into()
        },
    });
    if !hash_ok {
        return Ok(Verdict {
            checks,
            outcome: VerificationOutcome::Rejected,
            artifacts_checked,
        });
    }
    artifacts_checked.push(claimed_c);
    let candidate = match TensorF32::decode(&bytes, &tensor_limits_for(shape)) {
        Ok(t) => t,
        Err(e) => {
            checks.push(fail("outputs.c.format", e.to_string()));
            return Ok(Verdict {
                checks,
                outcome: VerificationOutcome::Rejected,
                artifacts_checked,
            });
        }
    };

    // 2. Reference recomputation from verified inputs.
    let a = load_input(task, "a", inputs)?;
    let b = load_input(task, "b", inputs)?;
    let reference = matmul_reference(&a, &b)?;
    checks.push(CheckV1 {
        name: "reference.recomputed".into(),
        passed: true,
        measured: None,
        tolerance: None,
        detail: format!("profile {}", crate::profiles::REFERENCE_PROFILE),
    });

    // 3. Tolerance comparison.
    let cmp = compare_f32("outputs.c", &candidate, &reference, task.tolerance);
    checks.extend(cmp.checks);
    let outcome = if cmp.passed {
        VerificationOutcome::Accepted
    } else {
        VerificationOutcome::Rejected
    };
    Ok(Verdict {
        checks,
        outcome,
        artifacts_checked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gg_protocol::drbg::Drbg;

    #[test]
    fn flex_matches_naive_oracle() {
        let mut d = Drbg::new("matmul-test", 1);
        let a = TensorF32::new(vec![17, 33], d.sym_f32_vec(17 * 33)).unwrap();
        let b = TensorF32::new(vec![33, 9], d.sym_f32_vec(33 * 9)).unwrap();
        let flex = matmul_reference(&a, &b).unwrap();
        let naive = matmul_naive_f64(&a, &b);
        let cmp = compare_f32(
            "c",
            &flex,
            &naive,
            gg_protocol::messages::ToleranceV1 {
                abs: 1e-5,
                rel: 1e-5,
            },
        );
        assert!(cmp.passed, "{:?}", cmp.checks);
    }
}
