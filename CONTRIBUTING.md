# Contributing to GPU Goblin

Thank you for your interest. This document describes what is expected of a contribution and,
above all, what will not pass review. It is the same in every GPU Goblin repository, apart from the
commands to run.

## Before writing code

Read [gg-server/docs/STATUS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/STATUS.md) (what actually works, measured on a real machine), [gg-server/docs/ROADMAP.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/ROADMAP.md) (where the project is
going) and [gg-server/docs/RISKS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/RISKS.md) (**the platform is not ready for public deployment**: four blockers, and
decisions that belong to a human). Open an issue before any large change.

## Non-negotiable rules

They come from the threat model, not from anyone's taste:

- **No code submitted by a researcher runs on a contributor's machine.** What crosses the network
  is a description (`gg-graph/v1`) and tensors. No Wasm, no container, no script, no binary.
- **No acceptance relies on what the worker reports**: the validator recomputes.
- **No `todo!()`, no simulated success, no fake validation and no hidden in-memory backend** in a
  path advertised as working. An ignored test is not a success.
- **Never hide an error** by disabling a test or relaxing a security rule.
- **An operator added to the profile** comes, without exception, with: its implementation, its
  verification oracle, its determinism test, its tolerance measured per numeric profile, its FLOP
  estimator and its memory estimator (ADR-0016).
- **No secrets or tokens** in the repository, the logs or the status document.
- **No real funds, no mainnet transactions.** The rewards contract is tested on a simulated chain.

## Before opening a pull request

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p gg-compute --features gpu --test gpu   # if a computation changes, on identified GPU hardware
```

These must pass. Clippy warnings are rejected: this is not a formality, it is what keeps the code
readable by someone else.

## What a review looks at

1. **The evidence.** A change in behavior comes with a test that would fail without it. A number
   that is claimed comes with the command that measures it.
2. **The bounds.** Every network input, file or tensor is untrusted: size, shape, NaN, infinity,
   name length, element count.
3. **The error message.** A rejection names what is wrong (the missing operator, the offending
   shape); it does not just say "invalid".
4. **The status document updated** ([gg-server/docs/STATUS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/STATUS.md)): what works, what is simulated, the commands that
   were run.
5. **Documents, code and identifiers in English**, like the rest of the project.

## Architecture decisions

A structural choice is recorded as an ADR in [gg-server/docs/adr](https://github.com/GPU-Goblin/gg-server/tree/main/docs/adr), numbered, with its context, the decision and
its consequences — including what it makes impossible. An ADR that turns out to be wrong is
corrected by a new ADR that says so (see ADR-0017 correcting ADR-0016), never by a silent rewrite.

## Contribution license

By submitting a contribution, you agree that it will be published under the project's license
(PolyForm Noncommercial 1.0.0; see `LICENSE` and `LICENSING.md`). You declare that you have the
right to submit it.
