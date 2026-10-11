# gg-core

The shared foundation of [GPU Goblin](https://github.com/GPU-Goblin/gg-server): the domain model,
the signed wire protocol, the approved computations and the client of the control plane. The
server, the validator, the command line and the contributor agent are all built on these crates.

GPU Goblin trains models and runs large computations on volunteer machines. No researcher code
ever runs on a contributor's machine: what crosses the network is a model description and tensors,
and every result is recomputed by a validator before anything is accepted or paid.

| Crate | Role |
|---|---|
| `gg-core` | Pure domain: identifiers, states and transitions, credits, invariants. No I/O. |
| `gg-protocol` | Deterministic DAG-CBOR encoding, domain-separated hashes, Ed25519 signatures, versioned messages, the `GGT1`/`GGB1` tensor formats, the deterministic generator. |
| `gg-compute` | The catalog of approved computations and their Burn implementations (CPU, optional WGPU/Vulkan): matrix products, perceptron training, gradients, and models described as `gg-graph/v1` graphs — execution, stages, tiling, memory and distribution planning. |
| `gg-client` | Client of the control plane: authenticated HTTP API with backpressure, resumable and verified transfers, the SHA-256 artifact cache. |

## Documentation

- [docs/MODELS.md](docs/MODELS.md) — the `gg-graph/v1` model description, the operator profile
  (`gg-ops/8`) and how a description is checked before anything is reserved.
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — the signed messages, their encoding and the test vectors in
  [tests/fixtures/protocol](tests/fixtures/protocol).
- [examples/models](examples/models) — four descriptions used by the tests, and therefore always
  valid.

The specification, the architecture decisions (`ADR-nnnn` in the code) and the project status live
in [gg-server/docs](https://github.com/GPU-Goblin/gg-server/tree/main/docs); references to
`gg-server/docs/...` in the code point there.

## Using the crates

The crates are not published on crates.io. The other repositories depend on a tagged version:

```toml
[workspace.dependencies]
gg-core = { git = "https://github.com/GPU-Goblin/gg-core", tag = "v0.1.0" }
gg-protocol = { git = "https://github.com/GPU-Goblin/gg-core", tag = "v0.1.0" }
gg-compute = { git = "https://github.com/GPU-Goblin/gg-core", tag = "v0.1.0", default-features = false }
gg-client = { git = "https://github.com/GPU-Goblin/gg-core", tag = "v0.1.0" }
```

A change to the protocol or to a computation changes what every other component computes or
verifies: it is released as a new tag, and the other repositories move to it explicitly.

## Building and testing

Prerequisites: Rust 1.96 (pinned by `rust-toolchain.toml`) and `ssh-keygen` (OpenSSH), against which
the release signatures are tested. No database, no network access.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

GPU variant (WGPU/Vulkan, hardware explicitly selected, clear failure otherwise; an ignored GPU
test is not a success):

```bash
cargo test -p gg-compute --features gpu --test gpu -- --nocapture
```

The protocol vectors are regenerated only for a deliberate protocol change, documented in
[docs/PROTOCOL.md](docs/PROTOCOL.md):

```bash
GG_UPDATE_VECTORS=1 cargo test -p gg-protocol --test vectors
```

## Related repositories

| Repository | Contents |
|---|---|
| [gg-server](https://github.com/GPU-Goblin/gg-server) | Control plane, validator, `gg` command line, ledger, reward contract, design documents |
| [gg-worker](https://github.com/GPU-Goblin/gg-worker) | The agent contributors install to share their GPU, with its sandboxed executor |
| [gg-python](https://github.com/GPU-Goblin/gg-python) | Python package and PyTorch driver for researchers |

## Status

Research software, version 0.1.0, used by the public beta at grid.gpugoblin.com. What works, what
does not, and the risks not yet addressed:
[gg-server/docs/STATUS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/STATUS.md) and
[gg-server/docs/RISKS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/RISKS.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free for research and any noncommercial use,
source-available, not open source in the OSI sense. See [LICENSING.md](LICENSING.md).
