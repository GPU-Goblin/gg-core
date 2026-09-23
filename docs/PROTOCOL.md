# Signed message protocol

Protocol version: `1`. Network: `NetworkId` (`[a-z0-9-]{1,32}`), e.g. `gg-dev-local`.

## Deterministic encoding (ADR-0003)

All signed or hashed metadata is encoded as **DAG-CBOR** (a strict subset of CBOR,
RFC 8949 §4.2 + IPLD constraints) via `serde_ipld_dagcbor`:

- integers in minimal form, range `[-2^64, 2^64-1]`;
- floats always 64-bit and **finite** (NaN/Inf refused on both encoding and decoding);
- map keys sorted (by length, then by lexicographic byte order), duplicates refused;
- no indefinite lengths, no tags other than CID (unused);
- decoding is strict: a non-canonical message is rejected before any signature check.

Serde conventions: structs → maps with text keys; unit enums → text; enums with data → a
`{variant: value}` map; bytes → CBOR byte string (`serde_bytes`); in JSON (`is_human_readable`),
hashes, keys, signatures and nonces are hexadecimal and amounts are strings.

## Hashing and domain separation

```
tagged(network, domain, bytes) = "gpu-goblin/v1\n" || network || "\n" || domain || "\n" || bytes
hash(network, domain, msg)      = SHA-256(tagged(network, domain, dagcbor(msg)))
sign(key, network, domain, msg) = Ed25519(key, tagged(network, domain, dagcbor(msg)))
```

Domains: `task-definition`, `task-spec`, `assignment`, `result`, `verification`, `receipt`,
`worker-login`, `chunk-manifest`, `peer-ticket`, `chunk-request`, `merkle-leaf`, `merkle-node`. Artifacts
are addressed by `SHA-256(raw bytes)` with no tag, so that standard tools can verify a file.

A signature is therefore bound to the protocol version, the network and the domain; a message signed for
`gg-dev-local` is invalid on any other network.

JSON transport: the server re-encodes the received body as DAG-CBOR before checking the signature. Any
JSON converter must therefore preserve f64 values exactly (serde_json with `float_roundtrip`, see
[gg-server/docs/DEPENDENCIES.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/DEPENDENCIES.md)); a property test checks that no finite f64 changes by even one ulp.

## Signed envelope

```
Signed<T> { body: T, signer: PublicKey(32 bytes), signature: Signature(64 bytes) }
```

`verify(network, domain)` recomputes `tagged(...)` from the re-encoded body and checks the Ed25519
signature (`verify_strict`). The server then maps `signer` to an authorized identity; cryptographic
validity is not an authorization.

## Messages (v1)

| Message | Signer | Main content |
|---|---|---|
| `TaskDefinitionV1` | — (hashed) | `compute_kind`, `catalog_version`, `inputs{name→hash,size}`, typed `params`, `numeric_profile`, `seed_strategy`, `resource_limits`, `validation_policy`, `reward_schedule` |
| `TaskSpecV1` | — (hashed) | `spec_version`, `network`, `project_id`, `experiment_id`, `job_id`, `task_id`, `index`, `definition` |
| `AssignmentV1` | server | `task_id`, `spec_hash`, `attempt_id`, `worker_id`, `lease_id`, `generation`, `role`, `assigned_at`, `lease_expires_at`, `lease_max_until` |
| `ResultSubmissionV1` | worker agent | `assignment_hash`, ids, `generation`, `outputs{name→hash,size}`, `execution_profile` (declared), `declared_metrics`, `submitted_at` |
| `VerificationDecisionV1` | validator | `submission_hash`, ids, `method`, `method_version`, `checks[]`, `artifacts_checked[]`, `outcome`, `decided_at` |
| `ContributionReceiptV1` | server | ids, `role`, `spec_hash`, `input_hashes`, `submission_hash`, `output_hashes`, `decision_hash`, `worker_public_key`, `beneficiary`, `amount`, `asset`, `issued_at` |
| `WorkerLoginV1` | worker/validator | `nonce`, `public_key`, `action`, `issued_at` |
| `ChunkManifestV1` | server | `artifact_hash`, `size`, `chunk_size`, `chunks[]` (SHA-256 of each chunk, computed by the server from its own bytes — or, for an activation that has no durable copy yet, declared by its producer, ADR-0026), `issued_at` — binds the artifact's identity to the transport's chunks (ADR-0012) |
| `PeerTicketV1` | server | `artifact_hash`, `size`, `chunk_size`, `reader_id`, `reader_key`, `issued_at`, `expires_at` — a read authorization that a holder can verify offline |
| `ChunkRequestV1` | reading worker | `artifact_hash`, `chunk_index`, `issued_at` — proves possession of `reader_key`; 120 s freshness |

Timestamps are `unix_millis` integers (i64). Amounts are `Credits` (u64, unit `TC`).
Floating-point hyperparameters are finite f64 values bounded by the catalog.

**Optional `shard_bytes` field of `graph.grad.v1`** (2026-09-23, [gg-server/docs/ROADMAP.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/ROADMAP.md) P11 C2): the target size
of a checkpoint shard. It is omitted from the encoding when zero, so earlier tasks keep exactly the same
bytes and digests (vectors unchanged). When non-zero, the task takes the inputs `checkpoint.0 …` and
returns the outputs `gradient.i`, following a deterministic partition of the description
(`gg_compute::graph::shards::plan`) that every party recomputes; nothing about the partition travels over
the wire. At most 40 shards, each of at most 1 GiB.

**Capabilities announced at claim time** (P11 C1): `POST /v1/worker/claim` accepts `ram_bytes`,
`device_memory_bytes`, `profile` (`gg-ops/N`) and `engine`. These are not signed messages: the server
uses them to offer only tasks that fit, and the worker enforces the limit it declares on itself.

**Activation handed off before its durable copy** (ADR-0026, the bounded exception to invariant I7 in
[gg-server/docs/SPEC.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/SPEC.md) §4). The signed message does not change: `ResultSubmissionV1` carries the hash and size of
each output. Three API rules are added, only for the `out.*` and `din.*` outputs of a `graph.stage.v1`
task:

- `POST /v1/worker/attempts/{id}/outputs/{hash}/declare {size, chunks[]}`: during the lease, the
  producer declares the chunk digests, computed with `p2p_chunk_bytes` (published by
  `GET /v1/info`, `0` when peer-to-peer distribution is disabled);
- the submission accepts an output declared by this attempt in place of an uploaded output;
  the durable copy (`PUT …/outputs/{hash}`) may still be uploaded after submission;
- `GET /v1/worker/artifacts/{hash}` returns `409` as long as the durable copy is missing. The manifest
  is then signed over the declared digests, and the producer may announce itself as a holder.

No verification is offered while any input or output of the task lacks a durable copy. After
`GG_PENDING_OUTPUT_SECS` (600 s by default), the staged round that passed on the activation is
reissued.

## `GGT1` tensor format

```
magic "GGT1" | u8 dtype (1 = f32 little-endian, 2 = bf16 little-endian) | u8 ndim (1..=4) | u64 LE shape[ndim] | data
```

Constraints: the product of the dimensions is ≤ 2^26 elements, the file size is exactly the header
plus `n × 4` bytes (`n × 2` in bf16), and no non-finite value is allowed as catalog input (checked on
load).

bf16 is the upper half of an f32, rounded to nearest even. It carries only the parameter gradients a
task sends when its parameters say `gradient_bf16` (ADR-0035). Everything else stays f32: checkpoints,
datasets, activations and their gradients. A reader always states the type it expects, and a file of
the other type is refused. `GGB1` bundles hold tensors of a single type.

## Test vectors

`tests/fixtures/protocol/*.json` contains, for each message, the readable JSON, the expected DAG-CBOR
bytes (hex) and the domain-separated hash. The `gg-protocol/tests/vectors.rs` test fails if the bytes
produced change. Deliberate regeneration: `GG_UPDATE_VECTORS=1 cargo test -p gg-protocol`.
