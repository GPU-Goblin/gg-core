//! Approved compute catalog and execution/verification adapters built on Burn.
//!
//! Researchers never submit code: only parameters of catalog kinds compiled into this crate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod catalog;
pub mod compare;
pub mod engine;
pub mod error;
pub mod grad;
pub mod graph;
pub mod matmul;
pub mod mlp;
pub mod profiles;
pub mod synthetic;

pub use catalog::{CATALOG_VERSION, ResourceEstimate, ValidatedTask, validate_definition};
pub use engine::{EngineInfo, EngineSelection};
pub use error::ComputeError;
pub use matmul::{ExecutionReport, OutputFile, Verdict};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use gg_protocol::messages::{ArtifactRef, TaskParamsV1};

/// Execute any catalog task on the selected engine.
pub fn execute_task(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
) -> Result<ExecutionReport, ComputeError> {
    match task.params {
        TaskParamsV1::Matmul { .. } => matmul::execute(engine, task, inputs, out_dir),
        TaskParamsV1::MlpTrain(_) => mlp::execute(engine, task, inputs, out_dir),
        TaskParamsV1::MlpGrad(_) => grad::execute(engine, task, inputs, out_dir),
        TaskParamsV1::GraphGrad(_) => graph::task::execute(engine, task, inputs, out_dir),
        TaskParamsV1::GraphStage(_) => {
            graph::stage_task::execute(engine, task, inputs, out_dir, None)
        }
    }
}

/// Execute any catalog task, with the accelerator memory the worker declared (C1). Only
/// `graph.grad.v1` estimates its working set today; the other kinds are bounded by their resource
/// limits as before.
pub fn execute_task_limited(
    engine: EngineSelection,
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &Path,
    device_memory: Option<u64>,
) -> Result<ExecutionReport, ComputeError> {
    match task.params {
        TaskParamsV1::GraphGrad(_) => {
            graph::task::execute_limited(engine, task, inputs, out_dir, device_memory)
        }
        TaskParamsV1::GraphStage(_) => {
            graph::stage_task::execute(engine, task, inputs, out_dir, device_memory)
        }
        _ => execute_task(engine, task, inputs, out_dir),
    }
}

/// Verify the outputs of any catalog task by recomputing the reference on the CPU. `declared_profile`
/// is the numeric profile declared by the worker; profiles not admitted for the kind yield
/// `inconclusive`.
pub fn verify_task(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
    declared_profile: &str,
) -> Result<Verdict, ComputeError> {
    verify_task_on(
        task,
        inputs,
        candidates,
        claimed,
        declared_profile,
        EngineSelection::FlexCpu,
    )
    .map(|(v, _)| v)
}

/// Like [`verify_task`], recomputing on `reference` where the kind admits it
/// ([`profiles::reference_admitted_for`]: model tasks on the GPU, ADR-0031); every other kind is
/// recomputed on the CPU. Returns the verdict and the profile the reference was computed with.
pub fn verify_task_on(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
    declared_profile: &str,
    reference: EngineSelection,
) -> Result<(Verdict, String), ComputeError> {
    let kind = task.params.compute_kind();
    let reference = if profiles::reference_admitted_for(kind, reference.numeric_profile()) {
        reference
    } else {
        EngineSelection::FlexCpu
    };
    let profile = reference.numeric_profile().to_owned();
    verify_with(
        task,
        inputs,
        candidates,
        claimed,
        declared_profile,
        reference,
    )
    .map(|v| (v, profile))
}

fn verify_with(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
    declared_profile: &str,
    reference: EngineSelection,
) -> Result<Verdict, ComputeError> {
    if !profiles::admitted_for(task.params.compute_kind(), declared_profile) {
        return Ok(Verdict {
            checks: vec![gg_protocol::messages::CheckV1 {
                name: "profile.admitted".into(),
                passed: false,
                measured: None,
                tolerance: None,
                detail: format!(
                    "profile '{declared_profile}' is not admitted for {}",
                    task.params.compute_kind()
                ),
            }],
            outcome: gg_core::VerificationOutcome::Inconclusive,
            artifacts_checked: Vec::new(),
        });
    }
    match task.params {
        TaskParamsV1::Matmul { .. } => matmul::verify(task, inputs, candidates, claimed),
        TaskParamsV1::MlpTrain(_) => {
            mlp::verify(task, inputs, candidates, claimed, declared_profile)
        }
        TaskParamsV1::MlpGrad(_) => {
            grad::verify(task, inputs, candidates, claimed, declared_profile)
        }
        TaskParamsV1::GraphGrad(_) => graph::task::verify(
            task,
            inputs,
            candidates,
            claimed,
            declared_profile,
            reference,
        ),
        TaskParamsV1::GraphStage(_) => graph::stage_task::verify(
            task,
            inputs,
            candidates,
            claimed,
            declared_profile,
            reference,
        ),
    }
}

/// Version string of the engine reported in execution profiles.
pub const ENGINE_VERSION: &str = "burn-0.21.0";
