# Security

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private vulnerability reporting on this repository
(the **Security** tab, then **Report a vulnerability**). We aim to respond within 7 days.

Coordinated disclosure is preferred: time to fix the issue and publish a signed release
([RELEASE.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/RELEASE.md)), followed by joint publication.

## Security status

**GPU Goblin is not ready for public deployment.** [RISKS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/RISKS.md) lists 14 residual risks,
**4 of which are blockers**:

| | Blocking risk |
|---|---|
| R1 | Collusion between validation operators |
| R2 | Server key compromise: rotation is not tooled |
| R3 | Non-reproducible binaries: a signed release cannot be rebuilt independently |
| R4 | No independent review has taken place |

The consequences are described in [THREAT_MODEL.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/THREAT_MODEL.md) and [RISKS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/RISKS.md). A public deployment
before they are resolved would be a decision taken against the project's own advice.

## What the platform protects, and what it does not

**Protected:**

- the contributor runs no third-party code: what crosses the network is a description and tensors
  (ADR-0016);
- the executor runs under bubblewrap, Landlock and seccomp (ADR-0009), without network access;
- no acceptance relies on what the worker reports: a validator recomputes;
- bytes received from a peer are verified against a server-signed manifest before being written;
- signed metadata uses a canonical encoding and domain separation
  ([gg-core/docs/PROTOCOL.md](https://github.com/GPU-Goblin/gg-core/blob/main/docs/PROTOCOL.md)).

**Not protected, and stated as such:**

- **data confidentiality with respect to contributors.** The owner of the computing machine sees
  what it processes. The deployment's default rule only admits synthetic or already public data
  ([DECISIONS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/DECISIONS.md), D4);
- availability in the face of a single control plane;
- contributor anonymity with respect to the control plane operator.

## Versions

The project is at `0.1.0` and does not yet have a version support policy. Releases are signed
offline and verifiable (`gg release verify`, [RELEASE.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/RELEASE.md)).
