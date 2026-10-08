//! Versioned protocol messages. Field order in structs is irrelevant to the encoding (DAG-CBOR
//! sorts keys); the *set* of fields and their types define the version.

use std::collections::BTreeMap;

use gg_core::{
    AttemptId, AttemptRole, Credits, DecisionId, ExperimentId, JobId, LeaseId, NetworkId,
    ProjectId, ReaderId, ReceiptId, ResourceLimits, Sha256Digest, TaskId, UnixMillis, ValidatorId,
    VerificationOutcome, WorkerId,
};
use serde::{Deserialize, Serialize};

use crate::ProtocolError;
use crate::keys::PublicKey;

/// Bound on free-text fields carried in signed messages.
pub const MAX_TEXT: usize = 256;
/// Bound on the number of named inputs/outputs.
pub const MAX_NAMED_ARTIFACTS: usize = 64;
/// Bound on the number of checks in a decision.
pub const MAX_CHECKS: usize = 256;
/// Bound on declared metrics.
pub const MAX_METRICS: usize = 64;

/// Reference to an immutable artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    /// SHA-256 of the raw bytes.
    pub hash: Sha256Digest,
    /// Size in bytes.
    pub size: u64,
}

/// Parameters of a catalog compute kind. The tag is the compute kind identifier; unknown kinds fail
/// to deserialize, which the API maps to an explicit "unknown compute kind" error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TaskParamsV1 {
    /// Bounded matrix product `C[m,n] = A[m,k] · B[k,n]` in f32.
    #[serde(rename = "tensor.matmul.v1")]
    Matmul {
        /// Rows of A and C.
        m: u32,
        /// Columns of A, rows of B.
        k: u32,
        /// Columns of B and C.
        n: u32,
    },
    /// Training of a small approved dense network (phase P2).
    #[serde(rename = "mlp.train.v1")]
    MlpTrain(MlpTrainParamsV1),
    /// Mean gradient of one explicit micro-batch from a given checkpoint (phase P4).
    #[serde(rename = "mlp.grad.v1")]
    MlpGrad(MlpGradParamsV1),
    /// Mean gradient of one explicit micro-batch of a model described by a graph (ADR-0016/0017).
    #[serde(rename = "graph.grad.v1")]
    GraphGrad(GraphGradParamsV1),
    /// One stage of a model split between machines, forward or backward (P11 D2, ADR-0024).
    #[serde(rename = "graph.stage.v1")]
    GraphStage(GraphStageParamsV1),
    /// Comparison of two computations of one task by two contributors (ADR-0042), computed by a
    /// third contributor the server drew: the check of a comparison a lab node reported.
    #[serde(rename = "replica.compare.v1")]
    ReplicaCompare(ReplicaCompareParamsV1),
}

impl TaskParamsV1 {
    /// Compute kind identifier.
    pub const fn compute_kind(&self) -> &'static str {
        match self {
            TaskParamsV1::Matmul { .. } => "tensor.matmul.v1",
            TaskParamsV1::MlpTrain(_) => "mlp.train.v1",
            TaskParamsV1::MlpGrad(_) => "mlp.grad.v1",
            TaskParamsV1::GraphGrad(_) => "graph.grad.v1",
            TaskParamsV1::GraphStage(_) => "graph.stage.v1",
            TaskParamsV1::ReplicaCompare(_) => "replica.compare.v1",
        }
    }
}

/// One dataset tensor of a `graph.grad.v1` task: a graph input or the loss target, with its full
/// shape (first dimension = number of examples).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetSpecV1 {
    /// Name of the graph input (or loss target) it feeds.
    pub name: String,
    /// Full shape, `[total_examples, ...]`.
    pub shape: Vec<u64>,
    /// `f32` or `i64` (indices are shipped as integral f32 values).
    pub dtype: String,
}

/// Parameters of `graph.grad.v1`. The model is the `graph` input (a `gg-graph/v1` description);
/// what the executor needs before opening it is declared here and cross-checked against it, so a
/// task that lies about its model is refused, never silently run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphGradParamsV1 {
    /// Hash of the `graph` input, as 64 hex characters. A string on purpose: inside the
    /// internally tagged `TaskParamsV1`, serde buffers the payload and loses the encoder's
    /// human-readability flag, so a binary digest would not round-trip through canonical CBOR.
    pub graph_hash: String,
    /// Size of the `graph` input.
    pub graph_bytes: u64,
    /// Parameters of the graph, frozen ones included.
    pub parameters: u64,
    /// Exact size of the `checkpoint` input: a `GGB1` bundle of **every** parameter.
    pub checkpoint_bytes: u64,
    /// Exact size of the `gradient` output: a `GGB1` bundle of the **trained** parameters only.
    /// Equal to `checkpoint_bytes` when nothing is frozen; much smaller with adapters (ADR-0019).
    pub gradient_bytes: u64,
    /// Segments of activation recomputation. `1` keeps every activation. Above that, the forward
    /// pass keeps only segment boundaries and each segment is recomputed during the backward pass:
    /// less memory, about a third more compute, same gradient.
    pub segments: u32,
    /// Upload only the current segment's parameters to the device instead of all of them.
    pub offload_parameters: bool,
    /// Training FLOPs per example, from the validation report; the task is priced on it.
    pub training_flops_per_example: u64,
    /// Activation bytes per example, from the validation report; bounds the memory limit.
    pub activation_bytes_per_example: u64,
    /// Dataset tensors, one per graph input plus the loss target.
    pub datasets: Vec<DatasetSpecV1>,
    /// Seed of the round's index stream (and of the Dropout masks).
    pub round_seed: u64,
    /// First position in the index stream.
    pub example_start: u64,
    /// Number of examples in the micro-batch.
    pub example_count: u32,
    /// Number of training examples.
    pub total_examples: u64,
    /// Target size of a checkpoint shard (P11, C2). `0`: one `checkpoint` input and one `gradient`
    /// output, as before. Otherwise inputs `checkpoint.0 …` and outputs `gradient.i`, grouped by a
    /// deterministic function of the graph and this size that every party recomputes
    /// (`gg_compute::graph::shards::plan`). Absent from the encoding when zero, so earlier tasks keep
    /// their exact bytes and hashes.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub shard_bytes: u64,
    /// Dataset inputs are the micro-batch itself (P11, C4): `[example_count, …]` rows already
    /// selected by the coordinator from the round's index stream, in stream order. A task then
    /// downloads its micro-batch, not the whole dataset. `false`: each dataset input is the whole
    /// dataset and the executor selects the rows, as before. Absent from the encoding when false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub batch_inputs: bool,
    /// Gradients of the trained parameters travel in bf16, rounded to nearest (ADR-0035): half the
    /// bytes, and a rounding error (≤ 2⁻⁹ of each element, about 0.16 % of the norm) far inside the
    /// validators' tolerance. The server averages and keeps weights and optimizer state in f32.
    /// Absent from the encoding when false, so earlier tasks keep their exact bytes and hashes.
    #[serde(default, skip_serializing_if = "is_false")]
    pub gradient_bf16: bool,
}

impl GraphGradParamsV1 {
    /// Element type of the `gradient` outputs.
    pub fn gradient_dtype(&self) -> crate::tensor_file::Dtype {
        gradient_dtype(self.gradient_bf16)
    }
}

impl GraphStageParamsV1 {
    /// Element type of the `gradient.j` outputs of a backward pass. Activations (`out.k`) and
    /// their gradients (`din.k`) stay f32.
    pub fn gradient_dtype(&self) -> crate::tensor_file::Dtype {
        gradient_dtype(self.gradient_bf16)
    }
}

fn gradient_dtype(bf16: bool) -> crate::tensor_file::Dtype {
    if bf16 {
        crate::tensor_file::Dtype::Bf16
    } else {
        crate::tensor_file::Dtype::F32
    }
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

fn is_false(v: &bool) -> bool {
    !*v
}

/// Parameters of `replica.compare.v1` (ADR-0042). Inputs: `a.<name>` and `b.<name>` for every name
/// of `outputs`, the two computations' outputs. Output: `verdict`, JSON `{outcome, checks}`, the
/// rule of `gg_compute::replica` applied to them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplicaCompareParamsV1 {
    /// Kind of the compared task (its comparison rule).
    pub compared_kind: String,
    /// The compared task's tolerance.
    pub tolerance: ToleranceV1,
    /// The compared task's bound on one output, bytes.
    pub max_output_bytes: u64,
    /// The compared task's output names.
    pub outputs: Vec<String>,
    /// Numeric profile side `a` declared.
    pub profile_a: String,
    /// Numeric profile side `b` declared.
    pub profile_b: String,
}

/// Parameters of `graph.stage.v1` (P11 D2, ADR-0024): one stage of a graph split at
/// `stage_bounds` (indices in the execution order, `0 = b₀ < … < b_K = n`), forward or backward, over
/// a micro-batch supplied by the coordinator. The stage's parameters, inputs and outputs, and the
/// checkpoint shards it reads, follow deterministically from the graph and the bounds
/// (`gg_compute::graph::stages`).
///
/// Inputs: `graph`; `checkpoint.j` for each shard holding a parameter of the stage; the micro-batch
/// rows of each dataset the stage reads, under the dataset's name; `in.k` for the k-th input of the
/// stage that is not a dataset; backward only, `dout.k.c` for the gradient of output k from its c-th
/// consumer, consumers taken in decreasing stage order.
/// Outputs: forward `out.k` per stage output; backward `gradient.j` per shard holding a trained
/// parameter of the stage, and `din.k` per non-dataset input; both `stage_meta`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphStageParamsV1 {
    /// Hash of the `graph` input, 64 hex characters.
    pub graph_hash: String,
    /// Size of the `graph` input.
    pub graph_bytes: u64,
    /// Parameters of the whole graph, frozen ones included.
    pub parameters: u64,
    /// Stage boundaries in the execution order.
    pub stage_bounds: Vec<u32>,
    /// This stage.
    pub stage: u32,
    /// Backward pass (recomputes the stage from its inputs) rather than forward.
    pub backward: bool,
    /// Dataset tensors of the whole run (full shapes), for their types and trailing dimensions.
    pub datasets: Vec<DatasetSpecV1>,
    /// Seed of the round (Dropout masks).
    pub round_seed: u64,
    /// First position of the micro-batch in the round's index stream.
    pub example_start: u64,
    /// Examples in the micro-batch.
    pub example_count: u32,
    /// Training examples.
    pub total_examples: u64,
    /// Training FLOPs per example of this stage, from the graph; the task is priced on it.
    pub training_flops_per_example: u64,
    /// Activation bytes per example of this stage, from the graph; bounds the outputs.
    pub activation_bytes_per_example: u64,
    /// Gradients of the trained parameters (`gradient.j`, not `din.k`) travel in bf16, rounded to nearest (ADR-0035): half the
    /// bytes, and a rounding error (≤ 2⁻⁹ of each element, about 0.16 % of the norm) far inside the
    /// validators' tolerance. The server averages and keeps weights and optimizer state in f32.
    /// Absent from the encoding when false, so earlier tasks keep their exact bytes and hashes.
    #[serde(default, skip_serializing_if = "is_false")]
    pub gradient_bf16: bool,
}

/// Parameters of `mlp.grad.v1`: the micro-batch is the slice `[example_start, example_start +
/// example_count)` of the index stream `mlp.grad.v1/order` seeded by `round_seed` over
/// `total_examples` training examples (ADR-0011).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlpGradParamsV1 {
    /// Input width.
    pub d_input: u32,
    /// Hidden width.
    pub d_hidden: u32,
    /// Output width.
    pub d_output: u32,
    /// Seed of the round's index stream.
    pub round_seed: u64,
    /// First position in the index stream.
    pub example_start: u64,
    /// Number of examples in the micro-batch.
    pub example_count: u32,
    /// Number of training examples (size of `train_x`).
    pub total_examples: u64,
}

/// Hyperparameters of `mlp.train.v1` (bounded by the catalog).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlpTrainParamsV1 {
    /// Input width.
    pub d_input: u32,
    /// Hidden width.
    pub d_hidden: u32,
    /// Output width.
    pub d_output: u32,
    /// Optimizer: `sgd` or `adam`.
    pub optimizer: String,
    /// Learning rate (finite, bounded).
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: u32,
    /// Number of optimizer steps.
    pub steps: u32,
    /// Seed for protocol-fixed initial weights and example order.
    pub seed: u64,
}

/// How randomness is fixed. Backend RNGs are never trusted for reproducibility.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum SeedStrategyV1 {
    /// No randomness involved.
    None,
    /// All randomness derives from the protocol DRBG with this seed.
    ProtocolDrbg {
        /// Seed.
        seed: u64,
    },
}

/// Element-wise tolerance: `|a - b| <= abs + rel * |b|`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToleranceV1 {
    /// Absolute tolerance.
    pub abs: f64,
    /// Relative tolerance.
    pub rel: f64,
}

/// Validation policy fixed before execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidationPolicyV1 {
    /// Method identifier, e.g. `recompute-reference`.
    pub method: String,
    /// Method version.
    pub method_version: u32,
    /// Numeric profile used to compute the reference, e.g. `cpu-f32-flex-v1`.
    pub reference_profile: String,
    /// Tolerance applied to compared outputs.
    pub tolerance: ToleranceV1,
}

/// Reward schedule announced in advance, in credits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardScheduleV1 {
    /// Reward for an accepted primary attempt.
    pub primary: Credits,
    /// Reward for an accepted authorized replica.
    pub replica: Credits,
    /// Reward for an accepted audit attempt.
    pub audit: Credits,
    /// Cost reserved for verification of one attempt.
    pub verification_cost: Credits,
}

impl RewardScheduleV1 {
    /// Reward for a role.
    pub fn for_role(&self, role: AttemptRole) -> Credits {
        match role {
            AttemptRole::Primary => self.primary,
            AttemptRole::Replica => self.replica,
            AttemptRole::Audit => self.audit,
        }
    }

    /// Amount to reserve when assigning an attempt with this role.
    pub fn reservation_for(&self, role: AttemptRole) -> Result<Credits, gg_core::CreditsError> {
        self.for_role(role).checked_add(self.verification_cost)
    }
}

/// Scientific definition of a task, free of instance identifiers. Hashed in domain `task-definition`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskDefinitionV1 {
    /// Catalog version the parameters were validated against.
    pub catalog_version: u32,
    /// Typed parameters (the tag is the compute kind).
    pub params: TaskParamsV1,
    /// Named inputs.
    pub inputs: BTreeMap<String, ArtifactRef>,
    /// Requested numeric profile family (the execution reports the actual profile).
    pub numeric_profile: String,
    /// Randomness strategy.
    pub seed_strategy: SeedStrategyV1,
    /// Resource limits.
    pub resource_limits: ResourceLimits,
    /// Validation policy.
    pub validation_policy: ValidationPolicyV1,
    /// Reward schedule.
    pub reward_schedule: RewardScheduleV1,
    /// Maximum number of attempts before the task fails.
    pub max_attempts: u32,
}

/// Task specification: definition plus instance identifiers. Hashed in domain `task-spec`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskSpecV1 {
    /// Specification version (1).
    pub spec_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Project.
    pub project_id: ProjectId,
    /// Experiment.
    pub experiment_id: ExperimentId,
    /// Job.
    pub job_id: JobId,
    /// Task instance identifier.
    pub task_id: TaskId,
    /// Index of the task within its job.
    pub index: u32,
    /// Scientific definition.
    pub definition: TaskDefinitionV1,
}

/// Assignment envelope, signed by the control server. Does not alter the scientific definition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssignmentV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Task.
    pub task_id: TaskId,
    /// Hash of the task specification.
    pub spec_hash: Sha256Digest,
    /// Attempt.
    pub attempt_id: AttemptId,
    /// Worker.
    pub worker_id: WorkerId,
    /// Lease.
    pub lease_id: LeaseId,
    /// Monotonic generation of the task; older generations can no longer publish.
    pub generation: u64,
    /// Role of the attempt.
    pub role: AttemptRole,
    /// Server time of assignment.
    pub assigned_at: UnixMillis,
    /// Current lease expiry (server time).
    pub lease_expires_at: UnixMillis,
    /// Hard bound after which the lease can no longer be renewed.
    pub lease_max_until: UnixMillis,
}

/// Declared execution profile. Not attested; stored as a declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProfileV1 {
    /// Engine identifier, e.g. `burn-flex`, `burn-wgpu-vulkan`.
    pub engine: String,
    /// Engine version.
    pub engine_version: String,
    /// Numeric profile actually used, e.g. `cpu-f32-flex-v1`.
    pub numeric_profile: String,
    /// Device name as reported locally.
    pub device_name: String,
    /// Declared wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Result submission, signed by the worker agent (the executor never holds the key).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultSubmissionV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Hash of the assignment envelope body.
    pub assignment_hash: Sha256Digest,
    /// Task.
    pub task_id: TaskId,
    /// Attempt.
    pub attempt_id: AttemptId,
    /// Specification hash.
    pub spec_hash: Sha256Digest,
    /// Generation of the assignment.
    pub generation: u64,
    /// Named outputs.
    pub outputs: BTreeMap<String, ArtifactRef>,
    /// Declared execution profile.
    pub execution_profile: ExecutionProfileV1,
    /// Declared metrics (never used for acceptance).
    pub declared_metrics: BTreeMap<String, f64>,
    /// Agent time of submission.
    pub submitted_at: UnixMillis,
}

/// One verification check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckV1 {
    /// Check name, e.g. `shape`, `finite`, `max_abs_diff`.
    pub name: String,
    /// Whether it passed.
    pub passed: bool,
    /// Measured value, if numeric.
    pub measured: Option<f64>,
    /// Tolerance applied, if any.
    pub tolerance: Option<f64>,
    /// Bounded detail.
    pub detail: String,
}

/// Verification decision, signed by a validator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerificationDecisionV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Decision id.
    pub decision_id: DecisionId,
    /// Hash of the verified submission body.
    pub submission_hash: Sha256Digest,
    /// Task.
    pub task_id: TaskId,
    /// Attempt.
    pub attempt_id: AttemptId,
    /// Validator identity record.
    pub validator_id: ValidatorId,
    /// Method identifier.
    pub method: String,
    /// Method version.
    pub method_version: u32,
    /// Profile used for the reference computation.
    pub reference_profile: String,
    /// Checks performed.
    pub checks: Vec<CheckV1>,
    /// Artifacts whose integrity was checked.
    pub artifacts_checked: Vec<ArtifactRef>,
    /// Outcome.
    pub outcome: VerificationOutcome,
    /// Validator time of decision.
    pub decided_at: UnixMillis,
}

/// Contribution receipt, signed by the control server. Attributes a statement to a key; it does not
/// prove physical hardware or consumed time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionReceiptV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Receipt id (global reward identifier).
    pub receipt_id: ReceiptId,
    /// Project.
    pub project_id: ProjectId,
    /// Experiment.
    pub experiment_id: ExperimentId,
    /// Job.
    pub job_id: JobId,
    /// Task.
    pub task_id: TaskId,
    /// Attempt.
    pub attempt_id: AttemptId,
    /// Role.
    pub role: AttemptRole,
    /// Specification hash.
    pub spec_hash: Sha256Digest,
    /// Definition hash.
    pub definition_hash: Sha256Digest,
    /// Input artifact hashes.
    pub input_hashes: Vec<Sha256Digest>,
    /// Submission hash.
    pub submission_hash: Sha256Digest,
    /// Output artifact hashes.
    pub output_hashes: Vec<Sha256Digest>,
    /// Decision hash.
    pub decision_hash: Sha256Digest,
    /// Hashes of every decision recorded on this submission, in verification order (primary then
    /// audit). With federated validation (ADR-0014) a payment is backed by administratively distinct
    /// validators, and the receipt says which signed decisions supported it.
    pub decision_hashes: Vec<Sha256Digest>,
    /// Worker record.
    pub worker_id: WorkerId,
    /// Worker signing key at the time of acceptance.
    pub worker_public_key: PublicKey,
    /// Payout address (separate from the technical key); absent until phase P7.
    pub beneficiary: Option<String>,
    /// Amount.
    pub amount: Credits,
    /// Asset label (`TC`).
    pub asset: String,
    /// Server time of issuance.
    pub issued_at: UnixMillis,
}

/// Login challenge signed by a worker or validator key. Bound to network and action.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoginChallengeV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Action: `worker-login` or `validator-login`.
    pub action: String,
    /// Server nonce (32 bytes), single use. Hex in JSON, byte string in DAG-CBOR.
    #[serde(with = "hex_or_bytes")]
    pub nonce: Vec<u8>,
    /// Key being authenticated.
    pub public_key: PublicKey,
    /// Client time.
    pub issued_at: UnixMillis,
}

/// Largest number of chunks in a manifest (64 GiB at 1 MiB chunks).
pub const MAX_CHUNKS: usize = 65_536;
/// Smallest and largest chunk size accepted in a manifest.
pub const MIN_CHUNK_BYTES: u32 = 64 * 1024;
/// Largest chunk size accepted in a manifest.
pub const MAX_CHUNK_BYTES: u32 = 16 * 1024 * 1024;

/// Chunk manifest of an immutable artifact, signed by the server (ADR-0012). It binds the artifact
/// identity (SHA-256 of the whole bytes) to the SHA-256 of each transport chunk, so bytes obtained
/// from any peer are verified chunk by chunk before they are written, and the whole-file hash is
/// still checked at the end.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkManifestV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Artifact identity: SHA-256 of the complete bytes.
    pub artifact_hash: Sha256Digest,
    /// Total size in bytes.
    pub size: u64,
    /// Size of every chunk but the last.
    pub chunk_size: u32,
    /// SHA-256 of each chunk, in order.
    pub chunks: Vec<Sha256Digest>,
    /// Server time.
    pub issued_at: UnixMillis,
}

impl ChunkManifestV1 {
    /// Structural validation: bounds, chunk count consistent with the size.
    pub fn validate_structure(&self) -> Result<(), ProtocolError> {
        if !(MIN_CHUNK_BYTES..=MAX_CHUNK_BYTES).contains(&self.chunk_size) {
            return Err(ProtocolError::Validation("chunk_size out of range".into()));
        }
        if self.chunks.len() > MAX_CHUNKS {
            return Err(ProtocolError::Validation("too many chunks".into()));
        }
        let expected = self.size.div_ceil(self.chunk_size as u64) as usize;
        if self.chunks.len() != expected {
            return Err(ProtocolError::Validation(format!(
                "manifest has {} chunks, size {} with chunk_size {} needs {expected}",
                self.chunks.len(),
                self.size,
                self.chunk_size
            )));
        }
        Ok(())
    }

    /// Byte range of chunk `index` within the artifact.
    pub fn range_of(&self, index: u32) -> Option<(u64, u64)> {
        if index as usize >= self.chunks.len() {
            return None;
        }
        let start = index as u64 * self.chunk_size as u64;
        let len = (self.size - start).min(self.chunk_size as u64);
        Some((start, len))
    }
}

/// Short-lived permission, signed by the server, allowing `reader_key` to fetch chunks of one
/// artifact from any holder. The holder verifies it offline against the server key: peers never
/// decide authorization themselves and never see the reader's session token.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerTicketV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Artifact this ticket allows reading.
    pub artifact_hash: Sha256Digest,
    /// Total size, so a holder computes the last chunk length from signed data.
    pub size: u64,
    /// Transport chunk size, likewise signed by the server.
    pub chunk_size: u32,
    /// Record of the reader: a worker, or a lab node (ADR-0041). Same encoding as before: a
    /// UUID.
    pub reader_id: ReaderId,
    /// Key the reader must sign its requests with.
    pub reader_key: PublicKey,
    /// Server time.
    pub issued_at: UnixMillis,
    /// Expiry (server time).
    pub expires_at: UnixMillis,
}

impl PeerTicketV1 {
    /// Byte range of chunk `index`, or `None` when the index is past the end.
    pub fn range_of(&self, index: u32) -> Option<(u64, u64)> {
        if self.chunk_size == 0 {
            return None;
        }
        let start = index as u64 * self.chunk_size as u64;
        if start >= self.size && !(self.size == 0 && index == 0) {
            return None;
        }
        Some((start, (self.size - start).min(self.chunk_size as u64)))
    }
}

/// Short-lived permission, signed by the server, for a holder to push the chunks of one artifact
/// to one reader over a connection the holder opens (ADR-0041, A8: the reversed connection, when
/// the reader can be reached and the holder cannot). The reader accepts a push only with a ticket
/// naming its own transport key and an artifact it is waiting for, and checks every chunk against
/// the signed manifest: the ticket keeps strangers from filling its connection, the manifest keeps
/// the bytes honest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushTicketV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Artifact to push.
    pub artifact_hash: Sha256Digest,
    /// Total size.
    pub size: u64,
    /// Transport chunk size.
    pub chunk_size: u32,
    /// SHA-256 of the reader's TLS transport key: the holder pins it, the reader checks it is its
    /// own.
    pub reader_fingerprint: Sha256Digest,
    /// Server time.
    pub issued_at: UnixMillis,
    /// Expiry (server time).
    pub expires_at: UnixMillis,
}

/// One side of a [`ComparisonV1`]: a computation of the task, by the submission it made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparedV1 {
    /// Its task (the first computation's, or a replica's).
    pub task_id: TaskId,
    /// Its attempt.
    pub attempt_id: AttemptId,
    /// Hash of its signed result submission.
    pub submission_hash: Sha256Digest,
}

/// The comparison of two computations of one task by two contributors (ADR-0042): whether their
/// outputs agree within the task's tolerance, with the checks measured. Signed by whoever compared
/// them; anyone holding both outputs can replay it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// The first computation's task: the family both belong to.
    pub family: TaskId,
    /// One side.
    pub a: ComparedV1,
    /// The other.
    pub b: ComparedV1,
    /// `accepted`: they agree; `rejected`: they differ; `inconclusive`: they could not be compared.
    pub outcome: gg_core::VerificationOutcome,
    /// The checks.
    pub checks: Vec<CheckV1>,
    /// When.
    pub compared_at: UnixMillis,
}

/// One chunk request, signed by the reader with the key named in its ticket.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRequestV1 {
    /// Protocol version.
    pub protocol_version: u32,
    /// Network.
    pub network: NetworkId,
    /// Artifact requested.
    pub artifact_hash: Sha256Digest,
    /// Chunk index.
    pub chunk_index: u32,
    /// Reader time (holders refuse stale requests).
    pub issued_at: UnixMillis,
}

/// Serde helper: hex text in human-readable formats, byte strings otherwise.
pub mod hex_or_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    /// Serialize.
    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            s.serialize_str(&hex::encode(v))
        } else {
            s.serialize_bytes(v)
        }
    }

    /// Deserialize.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        if d.is_human_readable() {
            let s = String::deserialize(d)?;
            hex::decode(s).map_err(serde::de::Error::custom)
        } else {
            serde_bytes::ByteBuf::deserialize(d).map(|b| b.into_vec())
        }
    }
}

fn check_text(name: &str, s: &str) -> Result<(), ProtocolError> {
    if s.len() > MAX_TEXT {
        return Err(ProtocolError::Validation(format!(
            "{name} exceeds {MAX_TEXT} bytes"
        )));
    }
    if s.chars().any(|c| c.is_control()) {
        return Err(ProtocolError::Validation(format!(
            "{name} contains control characters"
        )));
    }
    Ok(())
}

fn check_name(name: &str) -> Result<(), ProtocolError> {
    if name.is_empty() || name.len() > 64 {
        return Err(ProtocolError::Validation(format!(
            "artifact name '{name}' length out of range"
        )));
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
        || name.starts_with('.')
    {
        return Err(ProtocolError::Validation(format!(
            "artifact name '{name}' has invalid characters"
        )));
    }
    Ok(())
}

impl TaskDefinitionV1 {
    /// Structural validation independent of the catalog (bounds on names, counts, finiteness).
    pub fn validate_structure(&self) -> Result<(), ProtocolError> {
        if self.inputs.len() > MAX_NAMED_ARTIFACTS {
            return Err(ProtocolError::Validation("too many inputs".into()));
        }
        for name in self.inputs.keys() {
            check_name(name)?;
        }
        check_text("numeric_profile", &self.numeric_profile)?;
        check_text("validation_policy.method", &self.validation_policy.method)?;
        check_text(
            "validation_policy.reference_profile",
            &self.validation_policy.reference_profile,
        )?;
        let t = self.validation_policy.tolerance;
        if !(t.abs.is_finite() && t.rel.is_finite() && t.abs >= 0.0 && t.rel >= 0.0) {
            return Err(ProtocolError::Validation(
                "tolerance must be finite and non-negative".into(),
            ));
        }
        self.resource_limits
            .validate()
            .map_err(|e| ProtocolError::Validation(e.into()))?;
        if self.max_attempts == 0 || self.max_attempts > 16 {
            return Err(ProtocolError::Validation(
                "max_attempts must be in 1..=16".into(),
            ));
        }
        if let TaskParamsV1::MlpTrain(p) = &self.params {
            if !p.learning_rate.is_finite() {
                return Err(ProtocolError::Validation(
                    "learning_rate must be finite".into(),
                ));
            }
            check_text("optimizer", &p.optimizer)?;
        }
        Ok(())
    }
}

impl ResultSubmissionV1 {
    /// Structural validation.
    pub fn validate_structure(&self) -> Result<(), ProtocolError> {
        if self.protocol_version != crate::PROTOCOL_VERSION {
            return Err(ProtocolError::Validation(
                "unsupported protocol version".into(),
            ));
        }
        if self.outputs.len() > MAX_NAMED_ARTIFACTS {
            return Err(ProtocolError::Validation("too many outputs".into()));
        }
        for name in self.outputs.keys() {
            check_name(name)?;
        }
        if self.declared_metrics.len() > MAX_METRICS {
            return Err(ProtocolError::Validation("too many metrics".into()));
        }
        for (k, v) in &self.declared_metrics {
            check_text("metric name", k)?;
            if !v.is_finite() {
                return Err(ProtocolError::Validation("metric must be finite".into()));
            }
        }
        check_text("engine", &self.execution_profile.engine)?;
        check_text("engine_version", &self.execution_profile.engine_version)?;
        check_text("numeric_profile", &self.execution_profile.numeric_profile)?;
        check_text("device_name", &self.execution_profile.device_name)?;
        Ok(())
    }
}

impl VerificationDecisionV1 {
    /// Structural validation.
    pub fn validate_structure(&self) -> Result<(), ProtocolError> {
        if self.protocol_version != crate::PROTOCOL_VERSION {
            return Err(ProtocolError::Validation(
                "unsupported protocol version".into(),
            ));
        }
        if self.checks.len() > MAX_CHECKS {
            return Err(ProtocolError::Validation("too many checks".into()));
        }
        for c in &self.checks {
            check_text("check name", &c.name)?;
            check_text("check detail", &c.detail)?;
            if c.measured.is_some_and(|m| !m.is_finite())
                || c.tolerance.is_some_and(|t| !t.is_finite())
            {
                return Err(ProtocolError::Validation(
                    "check values must be finite".into(),
                ));
            }
        }
        check_text("method", &self.method)?;
        check_text("reference_profile", &self.reference_profile)?;
        Ok(())
    }
}

impl LoginChallengeV1 {
    /// Structural validation.
    pub fn validate_structure(&self) -> Result<(), ProtocolError> {
        if self.protocol_version != crate::PROTOCOL_VERSION {
            return Err(ProtocolError::Validation(
                "unsupported protocol version".into(),
            ));
        }
        if self.nonce.len() != 32 {
            return Err(ProtocolError::Validation("nonce must be 32 bytes".into()));
        }
        if !matches!(
            self.action.as_str(),
            "worker-login" | "validator-login" | "node-login"
        ) {
            return Err(ProtocolError::Validation("unknown login action".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_kind_is_rejected_explicitly() {
        let j = r#"{"kind":"shell.exec.v1","cmd":"rm -rf /"}"#;
        let err = serde_json::from_str::<TaskParamsV1>(j)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown variant"), "{err}");
        let ok: TaskParamsV1 =
            serde_json::from_str(r#"{"kind":"tensor.matmul.v1","m":2,"k":3,"n":4}"#).unwrap();
        assert_eq!(ok, TaskParamsV1::Matmul { m: 2, k: 3, n: 4 });
        assert_eq!(ok.compute_kind(), "tensor.matmul.v1");
    }

    #[test]
    fn artifact_names_are_bounded() {
        assert!(check_name("a").is_ok());
        assert!(check_name("out.c").is_ok());
        assert!(check_name("../x").is_err());
        assert!(check_name("/abs").is_err());
        assert!(check_name(".hidden").is_err());
        assert!(check_name("").is_err());
        assert!(check_name(&"a".repeat(65)).is_err());
    }
}
