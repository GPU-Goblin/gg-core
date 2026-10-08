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
`worker-login`, `chunk-manifest`, `peer-ticket`, `chunk-request`, `push-ticket`, `comparison`,
`merkle-leaf`, `merkle-node`. Artifacts
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

**Artifact held by a lab node** (ADR-0041, milestone A3; the second exception to invariant I7). No
signed message changes. A node signs in with the `node-login` action (`POST /v1/node/challenge`,
`POST /v1/node/session`), with a key its researcher registered (`POST /v1/researcher/nodes`), and:

- `POST /v1/node/artifacts/{hash} {project_id, size, chunks[], schema}` declares an artifact it holds,
  in chunks of `p2p_chunk_bytes`; the researcher must be a member of the project. The same declaration
  again is accepted; a hash recorded with another holder or other chunks is refused (`409`);
- `POST /v1/node/artifacts/{hash}/announce {addr, tls_fingerprint}` records where it serves the
  chunks, for the holding node only, after the same dial-back as a worker's announcement;
- `GET /v1/worker/artifacts/{hash}` (and the validator's) returns `409` for such an artifact: the
  control plane has no copy. The peer list (`GET …/peers`) lists the node first, with `node_id` in
  place of `worker_id`, and says `server_holds: false`; a reader then asks the peers again, for up to
  its request timeout, before trying the control plane. Both fields are new; a reader that does not
  know them keeps working, falling back to the control plane and getting `409`.

**Outputs pulled by a lab node** (ADR-0041, milestone A4).

- `PeerTicketV1.reader_id` names the reader's record, a worker's or a lab node's (`ReaderId`). Its
  encoding is unchanged (a UUID), and so are the test vectors.
- A job submission may carry `output_holder`: an active lab node of the submitting researcher.
- A worker announces `serves_peers: true` at claim time to be given such a job. The claim answer
  then says `outputs_to_node: true`.
- `POST /v1/worker/attempts/{id}/outputs/{hash}/declare` also accepts the outputs of such a job, and
  the submission accepts them declared in place of uploaded. An upload above `node_relay_max_bytes`
  (published by `GET /v1/info`, with `pending_output_secs`) is refused (`422`).
- The node lists what waits for it (`GET /v1/node/pulls`), and reads it through
  `GET /v1/node/artifacts/{hash}/manifest` and `…/peers`, with a ticket issued to its own key.
  `GET /v1/node/artifacts/{hash}` serves a small output the server stored, `409` otherwise.
- After checking the whole hash, the node calls `POST /v1/node/artifacts/{hash}/pulled`. The output
  is recorded as held by the node, stops being pending, and its verification becomes claimable.
- Past `pending_output_secs`, the server closes a submitted attempt with an unpulled output as
  inconclusive (no payment, no sanction) and queues the task again, for another worker. The node
  may bring that moment forward with `POST /v1/node/artifacts/{hash}/unavailable`.

**A shared lab node** (ADR-0041, milestone A7, step 1).

- A node works for the projects attached to it, whoever registered it.
- Its owner (the researcher who registered it) grants others the use of it with
  `POST /v1/researcher/nodes/{id}/users {researcher_id}`, lists them with `GET …/users`, and
  withdraws the grant with `POST …/users/{researcher}/remove`. Withdrawing leaves attached projects
  attached.
- A project member allowed to use the node attaches it with
  `POST /v1/researcher/projects/{id}/nodes {node_id}`. A node the researcher may not use answers
  `404`, as an unknown one does.
- A member detaches it with `POST …/nodes/{node}/detach`. This is refused (`409`) while the node
  steps a running run or pulls an unfinished job's outputs there.
- `GET /v1/researcher/nodes` lists the nodes a researcher owns or may use. `GET /v1/node/me` lists the
  projects the node works for.
- Holding an artifact, a job's `output_holder` and a run's `holder_node` require the node to be
  attached to the project. They no longer require the node to belong to the researcher.

**The node's own interfaces** (ADR-0041, milestone A7).

- **List of servers.** `servers.json` (`format: gg-servers/v1`, `issued_at`, `expires_at`, `servers`:
  `name`, `url`, `network`, `server_public_key`) and `servers.json.sig`, an armored SSHSIG signature
  (`ssh-keygen -Y sign -n gpugoblin-servers`) by the release key. The node refuses:
  - an unsigned list, a list signed by another key or in another namespace;
  - an expired list, or one older than the last it accepted;
  - a server that is not `https`, or one whose `/v1/info` presents another network or key.
- **Local API.** Every request carries the researcher's bearer token, which the node checks with
  `GET /v1/researcher/me`. With several servers, `X-GG-Network` names one. The routes:
  - `GET /v1/status`;
  - `PUT /v1/projects/{id}/data?schema=`, which returns `{hash, size}`;
  - `POST /v1/projects/{id}/runs`, the control plane's run request, with `holder_node` added;
  - `GET /v1/runs/{id}`, which returns `run`, `current_round_tasks` and `checkpoint_on_this_node`;
  - `GET /v1/runs/{id}/checkpoint/{name}`, the bytes, with the `x-gg-sha256` header;
  - `POST /v1/runs/{id}/cancel`.
- **Control plane.** A node may read the artifacts of the projects it is attached to
  (`GET /v1/node/artifacts/{hash}`). A graph run with `holder_node` may carry an `initial_checkpoint`
  held by that node; the server checks its size only.

**Reachability of a lab node** (ADR-0041, milestone A8).

- `GET /v1/node/me` gives `observed_ip`, the source address the server sees: one of the node's
  candidates, checked like the others by the dial-back.
- A node nobody can reach waits on `GET /v1/node/relay?wait_secs=N`. The server answers with the
  chunks readers ask for (`request_id`, `hash`, `index`, `start`, `len`). The node sends each chunk
  with `PUT /v1/node/relay/{request_id}` (raw bytes, at most 16 MiB, of exactly the length asked).
- Readers read as for an artifact the server stores (`GET /v1/worker/artifacts/{hash}`, the
  validator's, ranges included), for an artifact of at most `node_relay_max_bytes`. The server relays
  only when the node has no live announcement for that artifact. A node that can be reached is read
  directly, through peers. The answer
  carries `x-gg-relayed: lab-node`. If the node does not answer within `node_relay_wait_secs`, the
  reader gets a 503. The peer list says `server_holds: true` for such an artifact.
  `node_relay_max_bytes` is 0 by default: the server carries no data, and only a private
  deployment turns the relay on.

**The reversed connection** (ADR-0041, milestone A8, step 3). A reader that cannot reach the holder
of an artifact, but can itself be reached, has the holder connect to it. The server only puts the
two in touch; no byte of the artifact passes through it.

- The reader waits for the artifact on its own chunk server (the one it serves peers with; for a
  validator, `push_listen`), then asks
  `POST /v1/{worker|node|validator}/artifacts/{hash}/reverse` with `{addr, tls_fingerprint}`. The
  server checks the reader may read the artifact (as for a download), dials `addr` back (the same
  rules as an announcement), and answers 404 when no holder can be asked.
- It signs a `PushTicketV1` (domain `push-ticket`): network, artifact hash, size, chunk size, the
  reader's transport-key fingerprint, issue and expiry times (`p2p_ticket_ttl_secs`).
- The holders that may be unreachable wait on `GET /v1/node/pushes?wait_secs=N` (the lab node that
  holds it) and `GET /v1/worker/pushes?wait_secs=N` (the worker that produced a pending output). Each
  request carries `hash`, `size`, `chunk_size`, `reader_addr`, `reader_fingerprint` and `ticket`
  (hex of the signed ticket). At most 256 requests wait per holder; the same reader asking twice
  for the same artifact is queued once.
- The holder opens a TLS connection pinned to `reader_fingerprint` and sends each chunk with
  `PUT /p2p/v1/artifacts/{hash}/push/{index}`, header `x-gg-push-ticket`. The reader takes a chunk
  only under a ticket signed by the server, for this network and artifact, naming its own
  transport key and still valid (else 403); for an artifact it waits for, with the same size and
  chunk size as its manifest (else 409); whose SHA-256 is the manifest's for that index (else 422).
  It answers 204 when it took the chunk, and 409 once it no longer waits; the holder then stops.
- The reader assembles the chunks and checks the whole against the artifact hash, as for any peer.

**Verification by other contributors** (ADR-0042, `replica_verification`, off by default).

- A task is computed a second time as a task of its own (`replica_of`, `replica_seq` 1), in a twin
  job, created when the first computation is assigned. Its spec differs only by `task_id` and
  `job_id`; its assignment says `role: primary` like any other: a worker cannot tell.
- A contributor account (a worker without one is its own) computes one computation of a task
  only. The replica is offered first to one worker drawn by the server (the smallest
  `SHA-256("gg/pairing/v1" ‖ secret ‖ task_id ‖ worker_id)` among the workers of other accounts
  that announced, in the last 2 minutes, what the task needs), alone for `pairing_reserve_secs`;
  to any worker of another account afterwards.
- No validator verification is opened for such a task. Once two computations are submitted, the
  comparer signs a `ComparisonV1` (domain `comparison`): the family (first computation's task), each
  side's task, attempt and submission hash, the outcome, the checks, the time. The server is the
  comparer of what it stores. The rule: same output names; each file matching its declared hash
  and size; `GGT1`/`GGB1` values within the task's tolerance both ways (the norm rule for
  `graph.*`, element-wise otherwise); JSON outputs of the same structure with numbers within
  tolerance (`numeric_profile` and `engine` ignored); anything else byte-identical.
- Two that agree are accepted (receipts with roles `primary` and `replica`). Two that differ call a
  third computation (`replica_seq` 2) by a third account; the majority is accepted, the odd one
  rejected; a replica no longer needed is cancelled. Three that all differ open one dispute per
  computation (`outcomes: {"comparisons": [hashes]}`); an operator settles each on any outcome.
- Spot checks (`spot_check_rate`, 5 %) and probation (every task of an account with fewer than
  `probation_tasks`, 20, accepted computations): a third computation (`replica_seq` 2) is created
  at once and the family (`tasks.spot_check`) waits for it. A pair that agreed and that the check
  contradicts has both accounts suspended pending a human, and one dispute per computation.
- A lab node compares the computations of the jobs it holds. `GET /v1/node/comparisons?wait_secs=N`
  lists `family`, `definition` and the two sides (`task_id`, `attempt_id`, `submission_hash`,
  `outputs`, `profile`), once the node holds every output. The node posts the signed
  `ComparisonV1` to `POST /v1/node/comparisons`: 422 when not signed by the node's key or not
  naming the computations as submitted, 403 when the node is suspended, 409 when a computation is
  no longer current. An agreement counts as it is. A difference counts once a comparer confirms it:
  a `replica.compare.v1` task (catalog kind; inputs `a.<name>`, `b.<name>`, the two computations'
  outputs, read through peers; output `verdict`, JSON `{outcome, checks}`), computed by a worker of
  an account that took no part in the task, drawn by the server, claimable whatever its engine (it
  runs on the CPU), outside the project's `allowed_kinds`. A comparer that contradicts the node is
  followed by a second: two against the node suspend it; the second siding with the node suspends
  the first comparer's account. The check is stored with the comparison (`replayed_outcome`,
  `replayed_hash`: the comparer's submission, `replayed_by`). The server reads no output of a
  node's job.

**Runs stepped by a lab node** (ADR-0041, milestone A6).

- A graph run may carry `holder_node` at creation. The server keeps no weights; it records the
  node on the run and its round jobs (`output_holder`).
- `GET /v1/node/steps?wait_secs=N` lists what is due: step 0 (the initial weights, drawn from the
  seed) or a round whose tasks are all accepted, with the run's configuration, its current
  checkpoint and the accepted outputs in slice order. `GET /v1/node/pulls` takes the same
  `wait_secs`. Both answer as soon as something is due, and wait on a server signal otherwise.
- `POST /v1/node/runs/{id}/steps {step, round_id, checkpoint}` reports the new checkpoint. The server
  checks that it has every shard of the plan at its size, held by this node; that untrained shards
  are unchanged; that the moments are there for Adam; and that the step count is right. It then
  records the checkpoint and opens the next round.
- `POST /v1/node/artifacts/{hash}` accepts a `header` (hex, at most 64 bytes): the first bytes of the
  artifact. The server reads a dataset's shape from it without holding the dataset.
- **Rows cut by the node.** For a run whose tasks receive their micro-batch rows (`batch_inputs`),
  `GET /v1/node/rows` lists the steps whose rows are due (the open round's and the next):
  `run_id`, `project_id`, `step`, `round_seed`, `total_examples`, `slices`
  (`example_start`, `example_count`, one per task) and `datasets` (`name`, full `shape`, `parts`).
  The node cuts, for each task, the rows `round_indices(round_seed, total_examples, start, count)`
  of every dataset (GGT1 f32, shape `[count, …]`), holds them, and declares them with
  `POST /v1/node/runs/{id}/rows {step, rows}`, `rows` being one map per task of dataset name to
  artifact. The server answers 204 after checking the task count, the datasets, each size, and that
  the node holds each artifact for the run's project (422 otherwise); 409 for a step not due, or
  other rows than those already declared for it. A round opens only once its rows are declared;
  a round replayed at the same step reuses them.
- **Tiled runs.** A node's run may carry `tiling` and `source_graph`: `graph` is then the
  description the node tiled. The server checks that applying `tiling` to `source_graph`
  (functions inlined) gives `graph` exactly, and refuses `tile_max_param_bytes` for a node's run.
  The run's configuration records both, as for a run the server tiled.
- **Staged runs.** A node's run may carry `stage_bounds` (computed by the node); the server refuses
  `stage_memory_bytes` for it. Every stage job names the node as output holder. A step of a staged
  round is due once every wave is recorded and accepted; each item of `parts` then carries
  `partials`, the micro-batch's `[gradient.j, artifact]` pairs in production order, instead of
  `outputs`.

**Validators read through peers** (ADR-0041, milestone A5). `GET /v1/validator/artifacts/{hash}` answers
`409` for an artifact a lab node holds. The validator then reads it with
`GET /v1/validator/artifacts/{hash}/manifest` and `…/peers`, for the same artifacts it may download
(inputs and outputs of the verifications it holds, inputs of upcoming tasks), with a ticket issued
to its key (`ReaderId` from its record).

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
