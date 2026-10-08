//! Comparison of two computations of the same task by two contributors (ADR-0042).
//!
//! Neither side is the reference: the two outputs must agree within the task's tolerance, in both
//! directions, under the rule the kind is verified with (the norm rule of ADR-0031 for model
//! tasks, element by element otherwise). The comparison reads only the two outputs, never the
//! inputs, so it is one pass over the tensors; it is deterministic, so anyone holding both outputs
//! (the lab node, the server on a contestation) finds the same verdict.
//!
//! What is compared, output by output:
//! - the same output names on both sides, each file matching its declared hash and size;
//! - identical bytes agree at once;
//! - `GGT1` tensors and `GGB1` bundles: same dtype, same names and shapes, values within tolerance;
//! - bounded JSON outputs (the kinds' metadata): same structure, strings equal (except the
//!   engine-dependent `numeric_profile` and `engine`), numbers within tolerance;
//! - anything else must be byte-identical.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gg_core::{Sha256Digest, VerificationOutcome};
use gg_protocol::messages::{ArtifactRef, CheckV1, TaskParamsV1, ToleranceV1};

use crate::engine::cpu_info;
use crate::{ExecutionReport, OutputFile};
use gg_protocol::tensor_file::{BUNDLE_MAGIC, BundleF32, Dtype, MAGIC, TensorF32, TensorLimits};

use crate::compare::{Comparison, compare_f32, compare_f32_norm};
use crate::{ComputeError, ValidatedTask, Verdict};

/// One side: the outputs it declared, where their bytes are, and the profile it declared.
pub struct Side<'a> {
    /// The outputs it declared in its signed submission.
    pub outputs: &'a BTreeMap<String, ArtifactRef>,
    /// Where their bytes are.
    pub files: &'a BTreeMap<String, PathBuf>,
    /// The numeric profile it declared.
    pub profile: &'a str,
}

/// JSON keys whose values depend on the machine, not on the computation.
const MACHINE_KEYS: &[&str] = &["numeric_profile", "engine"];

/// Checks kept in a verdict (the protocol bounds them).
const MAX_CHECKS: usize = 64;

fn check(name: &str, passed: bool, detail: String) -> CheckV1 {
    CheckV1 {
        name: name.to_owned(),
        passed,
        measured: None,
        tolerance: None,
        detail: detail.chars().take(200).collect(),
    }
}

/// Compare `a` and `b`, two computations of `task`. `Accepted`: they agree. `Rejected`: they
/// differ, or a side's outputs do not match what it declared. `Inconclusive`: a side's profile is
/// not admitted for the kind, or a file is missing.
pub fn compare_replicas(
    task: &ValidatedTask,
    a: &Side<'_>,
    b: &Side<'_>,
) -> Result<Verdict, ComputeError> {
    compare_sides(
        task.params.compute_kind(),
        task.tolerance,
        task.limits.max_output_bytes,
        a,
        b,
    )
}

/// The same, for a task of `kind` with `tolerance` and a bound of `max_output_bytes` per output.
pub fn compare_sides(
    kind: &str,
    tolerance: ToleranceV1,
    max_output_bytes: u64,
    a: &Side<'_>,
    b: &Side<'_>,
) -> Result<Verdict, ComputeError> {
    let mut checks = Vec::new();
    let mut artifacts_checked = Vec::new();
    let verdict = |checks: Vec<CheckV1>, outcome, artifacts_checked| {
        Ok(Verdict {
            checks,
            outcome,
            artifacts_checked,
        })
    };
    for (who, s) in [("a", a), ("b", b)] {
        if !crate::profiles::admitted_for(kind, s.profile) {
            checks.push(check(
                &format!("{who}.profile.admitted"),
                false,
                format!("profile '{}' is not admitted for {kind}", s.profile),
            ));
            return verdict(checks, VerificationOutcome::Inconclusive, artifacts_checked);
        }
    }
    let names_a: Vec<&String> = a.outputs.keys().collect();
    let names_b: Vec<&String> = b.outputs.keys().collect();
    if names_a != names_b {
        checks.push(check(
            "outputs.names",
            false,
            format!("{names_a:?} against {names_b:?}"),
        ));
        return verdict(checks, VerificationOutcome::Rejected, artifacts_checked);
    }
    let norm = matches!(kind, "graph.grad.v1" | "graph.stage.v1");
    let mut agree = true;
    for name in a.outputs.keys() {
        let mut sides = Vec::with_capacity(2);
        for (who, s) in [("a", a), ("b", b)] {
            let r = s.outputs[name];
            let Some(path) = s.files.get(name) else {
                checks.push(check(
                    &format!("{who}.{name}.present"),
                    false,
                    "file missing".into(),
                ));
                return verdict(checks, VerificationOutcome::Inconclusive, artifacts_checked);
            };
            if std::fs::metadata(path)?.len() != r.size || r.size > max_output_bytes {
                checks.push(check(
                    &format!("{who}.{name}.size"),
                    false,
                    format!("declared {}", r.size),
                ));
                return verdict(checks, VerificationOutcome::Rejected, artifacts_checked);
            }
            let bytes = std::fs::read(path)?;
            if Sha256Digest::of_bytes(&bytes) != r.hash {
                checks.push(check(
                    &format!("{who}.{name}.sha256"),
                    false,
                    "hash mismatch".into(),
                ));
                return verdict(checks, VerificationOutcome::Rejected, artifacts_checked);
            }
            artifacts_checked.push(r);
            sides.push(bytes);
        }
        let (x, y) = (&sides[0], &sides[1]);
        let c = if x == y {
            Comparison {
                checks: vec![check(&format!("outputs.{name}"), true, "identical".into())],
                passed: true,
            }
        } else {
            compare_bytes(&format!("outputs.{name}"), x, y, tolerance, norm)
        };
        agree &= c.passed;
        if !c.passed || checks.len() < MAX_CHECKS {
            checks.extend(c.checks);
        }
    }
    checks.truncate(MAX_CHECKS.max(1));
    let outcome = if agree {
        VerificationOutcome::Accepted
    } else {
        VerificationOutcome::Rejected
    };
    verdict(checks, outcome, artifacts_checked)
}

type Refs = BTreeMap<String, ArtifactRef>;
type Files = BTreeMap<String, PathBuf>;

/// Name of the output of `replica.compare.v1`.
pub const VERDICT: &str = "verdict";

/// What a comparer writes: the outcome and its checks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompareVerdict {
    /// `accepted`: the two computations agree; `rejected`: they differ; `inconclusive`: they
    /// could not be compared.
    pub outcome: VerificationOutcome,
    /// The checks.
    pub checks: Vec<CheckV1>,
}

/// Run `replica.compare.v1`: compare the two computations named in its inputs and write the
/// verdict. The comparison is the same code whatever the engine: it runs on the CPU.
pub fn execute(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    out_dir: &std::path::Path,
) -> Result<ExecutionReport, ComputeError> {
    let started = std::time::Instant::now();
    let TaskParamsV1::ReplicaCompare(p) = &task.params else {
        return Err(ComputeError::InvalidParams(
            "not a replica.compare.v1 task".into(),
        ));
    };
    let side = |prefix: &str| -> Result<(Refs, Files), ComputeError> {
        let mut refs = BTreeMap::new();
        let mut files = BTreeMap::new();
        for name in &p.outputs {
            let path = inputs
                .get(&format!("{prefix}.{name}"))
                .ok_or_else(|| ComputeError::Inputs(format!("missing {prefix}.{name}")))?;
            let bytes = std::fs::read(path)?;
            refs.insert(
                name.clone(),
                ArtifactRef {
                    hash: Sha256Digest::of_bytes(&bytes),
                    size: bytes.len() as u64,
                },
            );
            files.insert(name.clone(), path.clone());
        }
        Ok((refs, files))
    };
    let (ra, fa) = side("a")?;
    let (rb, fb) = side("b")?;
    let v = compare_sides(
        &p.compared_kind,
        p.tolerance,
        p.max_output_bytes,
        &Side {
            outputs: &ra,
            files: &fa,
            profile: &p.profile_a,
        },
        &Side {
            outputs: &rb,
            files: &fb,
            profile: &p.profile_b,
        },
    )?;
    std::fs::create_dir_all(out_dir)?;
    let mut bytes = serde_json::to_vec(&CompareVerdict {
        outcome: v.outcome,
        checks: v.checks.clone(),
    })
    .map_err(|e| ComputeError::InvalidParams(e.to_string()))?;
    if bytes.len() as u64 > crate::catalog::MAX_VERDICT_BYTES {
        bytes = serde_json::to_vec(&CompareVerdict {
            outcome: v.outcome,
            checks: v
                .checks
                .into_iter()
                .filter(|c| !c.passed)
                .take(16)
                .collect(),
        })
        .map_err(|e| ComputeError::InvalidParams(e.to_string()))?;
    }
    let path = out_dir.join(VERDICT);
    std::fs::write(&path, &bytes)?;
    let mut outputs = BTreeMap::new();
    outputs.insert(
        VERDICT.to_owned(),
        OutputFile {
            path,
            hash: Sha256Digest::of_bytes(&bytes),
            size: bytes.len() as u64,
        },
    );
    Ok(ExecutionReport {
        outputs,
        engine: cpu_info(),
        duration_ms: started.elapsed().as_millis() as u64,
        metrics: BTreeMap::new(),
    })
}

/// Verify a comparer's verdict by comparing again: the same outcome is accepted.
pub fn verify(
    task: &ValidatedTask,
    inputs: &BTreeMap<String, PathBuf>,
    candidates: &BTreeMap<String, PathBuf>,
    claimed: &BTreeMap<String, ArtifactRef>,
) -> Result<Verdict, ComputeError> {
    let fail = |name: &str, detail: String| Verdict {
        checks: vec![check(name, false, detail)],
        outcome: VerificationOutcome::Rejected,
        artifacts_checked: Vec::new(),
    };
    let (Some(r), Some(path)) = (claimed.get(VERDICT), candidates.get(VERDICT)) else {
        return Ok(fail("outputs.verdict.present", "no verdict".into()));
    };
    let bytes = std::fs::read(path)?;
    if Sha256Digest::of_bytes(&bytes) != r.hash || bytes.len() as u64 != r.size {
        return Ok(fail("outputs.verdict.sha256", "hash mismatch".into()));
    }
    let Ok(given) = read_verdict(&bytes) else {
        return Ok(fail("outputs.verdict.format", "not a verdict".into()));
    };
    let dir = std::env::temp_dir().join(format!("gg-compare-{}", uuid_like(&r.hash)));
    let again = execute(task, inputs, &dir)
        .and_then(|rep| Ok(std::fs::read(&rep.outputs[VERDICT].path)?))
        .and_then(|b| read_verdict(&b));
    let _ = std::fs::remove_dir_all(&dir);
    let again = again?;
    let same = again.outcome == given.outcome;
    Ok(Verdict {
        checks: vec![check(
            "outputs.verdict.outcome",
            same,
            format!("declared {}, recomputed {}", given.outcome, again.outcome),
        )],
        outcome: if same {
            VerificationOutcome::Accepted
        } else {
            VerificationOutcome::Rejected
        },
        artifacts_checked: vec![*r],
    })
}

fn uuid_like(h: &Sha256Digest) -> String {
    h.to_hex()[..16].to_owned()
}

/// The verdict a comparer wrote, bounded and parsed.
pub fn read_verdict(bytes: &[u8]) -> Result<CompareVerdict, ComputeError> {
    if bytes.len() as u64 > crate::catalog::MAX_VERDICT_BYTES {
        return Err(ComputeError::InvalidParams("verdict too large".into()));
    }
    serde_json::from_slice(bytes).map_err(|e| ComputeError::InvalidParams(format!("verdict: {e}")))
}

fn dtype_of_tensor(bytes: &[u8]) -> Option<Dtype> {
    match bytes.get(4)? {
        1 => Some(Dtype::F32),
        2 => Some(Dtype::Bf16),
        _ => None,
    }
}

fn limits(bytes: &[u8]) -> TensorLimits {
    TensorLimits {
        max_elements: bytes.len() as u64,
    }
}

/// Two tensors, both ways: neither side is the reference.
fn compare_tensors(
    name: &str,
    x: &TensorF32,
    y: &TensorF32,
    tol: ToleranceV1,
    norm: bool,
) -> Comparison {
    let one = |c: &TensorF32, r: &TensorF32| {
        if norm {
            compare_f32_norm(name, c, r, tol)
        } else {
            compare_f32(name, c, r, tol)
        }
    };
    let ab = one(x, y);
    if !ab.passed {
        return ab;
    }
    let ba = one(y, x);
    Comparison {
        passed: ba.passed,
        checks: if ba.passed { ab.checks } else { ba.checks },
    }
}

fn differ(name: &str, why: String) -> Comparison {
    Comparison {
        checks: vec![check(name, false, why)],
        passed: false,
    }
}

fn compare_bytes(name: &str, x: &[u8], y: &[u8], tol: ToleranceV1, norm: bool) -> Comparison {
    if x.starts_with(MAGIC) {
        let (Some(dx), Some(dy)) = (dtype_of_tensor(x), dtype_of_tensor(y)) else {
            return differ(name, "not two tensors of a known type".into());
        };
        if dx != dy {
            return differ(name, "tensors of different types".into());
        }
        return match (
            TensorF32::decode_as(x, &limits(x), dx),
            TensorF32::decode_as(y, &limits(y), dy),
        ) {
            (Ok(tx), Ok(ty)) => compare_tensors(name, &tx, &ty, tol, norm),
            (Err(e), _) | (_, Err(e)) => differ(name, format!("tensor: {e}")),
        };
    }
    if x.starts_with(BUNDLE_MAGIC) {
        let decode = |b: &[u8]| {
            BundleF32::decode_as(b, &limits(b), Dtype::F32)
                .map(|v| (Dtype::F32, v))
                .or_else(|_| {
                    BundleF32::decode_as(b, &limits(b), Dtype::Bf16).map(|v| (Dtype::Bf16, v))
                })
        };
        return match (decode(x), decode(y)) {
            (Ok((dx, bx)), Ok((dy, by))) => {
                if dx != dy {
                    return differ(name, "bundles of different types".into());
                }
                let kx: Vec<&String> = bx.entries.keys().collect();
                let ky: Vec<&String> = by.entries.keys().collect();
                if kx != ky {
                    return differ(name, format!("entries {kx:?} against {ky:?}"));
                }
                let mut out = Comparison {
                    checks: Vec::new(),
                    passed: true,
                };
                for (k, tx) in &bx.entries {
                    let c = compare_tensors(&format!("{name}.{k}"), tx, &by.entries[k], tol, norm);
                    out.passed &= c.passed;
                    if !c.passed || out.checks.len() < MAX_CHECKS {
                        out.checks.extend(c.checks);
                    }
                }
                out
            }
            (Err(e), _) | (_, Err(e)) => differ(name, format!("bundle: {e}")),
        };
    }
    match (
        serde_json::from_slice::<serde_json::Value>(x),
        serde_json::from_slice::<serde_json::Value>(y),
    ) {
        (Ok(jx), Ok(jy)) => match json_agree("", &jx, &jy, tol) {
            Ok(()) => Comparison {
                checks: vec![check(
                    name,
                    true,
                    "same structure, numbers within tolerance".into(),
                )],
                passed: true,
            },
            Err(why) => differ(name, why),
        },
        _ => differ(name, "different bytes".into()),
    }
}

fn number_agrees(a: f64, b: f64, tol: ToleranceV1) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= tol.abs + tol.rel * a.abs().max(b.abs())
}

fn json_agree(
    path: &str,
    x: &serde_json::Value,
    y: &serde_json::Value,
    tol: ToleranceV1,
) -> Result<(), String> {
    use serde_json::Value as J;
    match (x, y) {
        (J::Object(mx), J::Object(my)) => {
            let kx: Vec<&String> = mx.keys().collect();
            let ky: Vec<&String> = my.keys().collect();
            if kx != ky {
                return Err(format!("{path}: keys {kx:?} against {ky:?}"));
            }
            for (k, vx) in mx {
                if MACHINE_KEYS.contains(&k.as_str()) {
                    continue;
                }
                json_agree(&format!("{path}.{k}"), vx, &my[k], tol)?;
            }
            Ok(())
        }
        (J::Array(ax), J::Array(ay)) => {
            if ax.len() != ay.len() {
                return Err(format!("{path}: {} against {} items", ax.len(), ay.len()));
            }
            for (i, (vx, vy)) in ax.iter().zip(ay).enumerate() {
                json_agree(&format!("{path}[{i}]"), vx, vy, tol)?;
            }
            Ok(())
        }
        (J::Number(nx), J::Number(ny)) => match (nx.as_f64(), ny.as_f64()) {
            (Some(a), Some(b)) if number_agrees(a, b, tol) => Ok(()),
            _ => Err(format!("{path}: {nx} against {ny}")),
        },
        _ if x == y => Ok(()),
        _ => Err(format!("{path}: {x} against {y}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tol() -> ToleranceV1 {
        ToleranceV1 {
            abs: 1e-5,
            rel: 1e-4,
        }
    }

    #[test]
    fn tensors_agree_within_tolerance_and_both_ways() {
        let a = TensorF32::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = TensorF32::new(vec![3], vec![1.0, 2.00001, 3.0]).unwrap();
        let far = TensorF32::new(vec![3], vec![1.0, 2.1, 3.0]).unwrap();
        assert!(compare_bytes("t", &a.encode(), &b.encode(), tol(), false).passed);
        assert!(!compare_bytes("t", &a.encode(), &far.encode(), tol(), false).passed);
        assert!(!compare_bytes("t", &far.encode(), &a.encode(), tol(), false).passed);
        let other = TensorF32::new(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!compare_bytes("t", &a.encode(), &other.encode(), tol(), false).passed);
    }

    #[test]
    fn a_comparer_writes_a_verdict_a_validator_can_check_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let t = |v: f32| {
            TensorF32::new(vec![2, 2], vec![1.0, 2.0, 3.0, v])
                .unwrap()
                .encode()
        };
        let file = |name: &str, bytes: Vec<u8>| {
            let p = dir.path().join(name);
            std::fs::write(&p, &bytes).unwrap();
            (
                p,
                ArtifactRef {
                    hash: Sha256Digest::of_bytes(&bytes),
                    size: bytes.len() as u64,
                },
            )
        };
        let (pa, ra) = file("a", t(4.0));
        let (pb, rb) = file("b", t(4.0000001));
        let (pc, rc) = file("c", t(5.0));
        let any = ArtifactRef {
            hash: Sha256Digest::of_bytes(b"x"),
            size: 38,
        };
        let compared = crate::catalog::matmul_definition(2, 2, 2, any, any).unwrap();
        let run = |other: (&PathBuf, ArtifactRef)| {
            let outs = |r: ArtifactRef| BTreeMap::from([("c".to_owned(), r)]);
            let def = crate::catalog::replica_compare_definition(
                &compared,
                (&outs(ra), crate::profiles::CPU_F32_FLEX_V1),
                (&outs(other.1), crate::profiles::GPU_F32_WGPU_VULKAN_V1),
            )
            .unwrap();
            let task = crate::validate_definition(&def).unwrap();
            let inputs = BTreeMap::from([
                ("a.c".to_owned(), pa.clone()),
                ("b.c".to_owned(), other.0.clone()),
            ]);
            let out = dir.path().join(format!("out-{}", other.1.hash));
            let rep = execute(&task, &inputs, &out).unwrap();
            let v = read_verdict(&std::fs::read(&rep.outputs[VERDICT].path).unwrap()).unwrap();
            (task, inputs, rep, v.outcome)
        };
        let (task, inputs, rep, outcome) = run((&pb, rb));
        assert_eq!(outcome, VerificationOutcome::Accepted);
        let claimed = BTreeMap::from([(VERDICT.to_owned(), rep.outputs[VERDICT].artifact_ref())]);
        let files = BTreeMap::from([(VERDICT.to_owned(), rep.outputs[VERDICT].path.clone())]);
        assert_eq!(
            verify(&task, &inputs, &files, &claimed).unwrap().outcome,
            VerificationOutcome::Accepted
        );
        let (_, _, _, outcome) = run((&pc, rc));
        assert_eq!(outcome, VerificationOutcome::Rejected);
        // A verdict that says otherwise than the comparison is rejected.
        let lie = serde_json::to_vec(&CompareVerdict {
            outcome: VerificationOutcome::Rejected,
            checks: Vec::new(),
        })
        .unwrap();
        let (pl, rl) = file("lie", lie);
        let claimed = BTreeMap::from([(VERDICT.to_owned(), rl)]);
        let files = BTreeMap::from([(VERDICT.to_owned(), pl)]);
        assert_eq!(
            verify(&task, &inputs, &files, &claimed).unwrap().outcome,
            VerificationOutcome::Rejected
        );
    }

    #[test]
    fn metadata_ignores_the_machine_but_not_the_numbers() {
        let a = br#"{"kind":"k","loss":0.5,"numeric_profile":"cpu-f32"}"#;
        let b = br#"{"kind":"k","loss":0.50000001,"numeric_profile":"cuda-f32"}"#;
        let c = br#"{"kind":"k","loss":0.7,"numeric_profile":"cpu-f32"}"#;
        let d = br#"{"kind":"j","loss":0.5,"numeric_profile":"cpu-f32"}"#;
        assert!(compare_bytes("m", a, b, tol(), false).passed);
        assert!(!compare_bytes("m", a, c, tol(), false).passed);
        assert!(!compare_bytes("m", a, d, tol(), false).passed);
        assert!(!compare_bytes("m", b"\x00\x01", b"\x00\x02", tol(), false).passed);
    }
}
