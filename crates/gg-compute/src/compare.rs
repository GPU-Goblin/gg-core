//! Tolerance-based comparison of tensors, producing auditable checks.

use gg_protocol::messages::{CheckV1, ToleranceV1};
use gg_protocol::tensor_file::TensorF32;

/// Outcome of a comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    /// Checks performed, in order.
    pub checks: Vec<CheckV1>,
    /// All checks passed.
    pub passed: bool,
}

fn check(
    name: &str,
    passed: bool,
    measured: Option<f64>,
    tolerance: Option<f64>,
    detail: String,
) -> CheckV1 {
    CheckV1 {
        name: name.to_owned(),
        passed,
        measured,
        tolerance,
        detail: detail.chars().take(200).collect(),
    }
}

/// Compare `candidate` against `reference` with `tol`: shape, finiteness, then element-wise
/// `|a - b| <= abs + rel * |b|`. Reports max absolute and max relative deviations.
pub fn compare_f32(
    name: &str,
    candidate: &TensorF32,
    reference: &TensorF32,
    tol: ToleranceV1,
) -> Comparison {
    let mut checks = Vec::new();
    let shape_ok = candidate.shape == reference.shape;
    checks.push(check(
        &format!("{name}.shape"),
        shape_ok,
        None,
        None,
        format!("expected {:?}, got {:?}", reference.shape, candidate.shape),
    ));
    if !shape_ok {
        return Comparison {
            checks,
            passed: false,
        };
    }
    let finite = candidate.check_finite();
    let finite_ok = finite.is_ok();
    checks.push(check(
        &format!("{name}.finite"),
        finite_ok,
        None,
        None,
        finite.err().map(|e| e.to_string()).unwrap_or_default(),
    ));
    if !finite_ok {
        return Comparison {
            checks,
            passed: false,
        };
    }
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut violations = 0u64;
    let mut first_violation = None;
    for (i, (a, b)) in candidate.data.iter().zip(reference.data.iter()).enumerate() {
        let a = *a as f64;
        let b = *b as f64;
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        if b != 0.0 {
            max_rel = max_rel.max(diff / b.abs());
        }
        if diff > tol.abs + tol.rel * b.abs() {
            violations += 1;
            if first_violation.is_none() {
                first_violation = Some((i, a, b));
            }
        }
    }
    let ok = violations == 0;
    checks.push(check(
        &format!("{name}.max_abs_diff"),
        true,
        Some(max_abs),
        Some(tol.abs),
        String::new(),
    ));
    checks.push(check(
        &format!("{name}.max_rel_diff"),
        true,
        Some(max_rel),
        Some(tol.rel),
        String::new(),
    ));
    checks.push(check(
        &format!("{name}.within_tolerance"),
        ok,
        Some(violations as f64),
        Some(0.0),
        first_violation
            .map(|(i, a, b)| format!("first violation at index {i}: got {a}, reference {b}"))
            .unwrap_or_default(),
    ));
    Comparison { checks, passed: ok }
}

/// Compare `candidate` against `reference` as one vector (`recompute-reference` v2, ADR-0031):
/// shape, finiteness, then `‖a − b‖ ≤ rel·‖b‖ + abs·√n`. Rounding differences between engines
/// spread over a tensor and stay far below this bound; a computation that differs does not. The
/// largest element gap is reported, for the record.
pub fn compare_f32_norm(
    name: &str,
    candidate: &TensorF32,
    reference: &TensorF32,
    tol: ToleranceV1,
) -> Comparison {
    let mut checks = Vec::new();
    let shape_ok = candidate.shape == reference.shape;
    checks.push(check(
        &format!("{name}.shape"),
        shape_ok,
        None,
        None,
        format!("expected {:?}, got {:?}", reference.shape, candidate.shape),
    ));
    if !shape_ok {
        return Comparison {
            checks,
            passed: false,
        };
    }
    let finite = candidate.check_finite();
    if let Err(e) = finite {
        checks.push(check(
            &format!("{name}.finite"),
            false,
            None,
            None,
            e.to_string(),
        ));
        return Comparison {
            checks,
            passed: false,
        };
    }
    checks.push(check(
        &format!("{name}.finite"),
        true,
        None,
        None,
        String::new(),
    ));
    let (mut diff2, mut ref2, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in candidate.data.iter().zip(&reference.data) {
        let (a, b) = (*a as f64, *b as f64);
        let d = (a - b).abs();
        diff2 += d * d;
        ref2 += b * b;
        max_abs = max_abs.max(d);
    }
    let (diff, norm) = (diff2.sqrt(), ref2.sqrt());
    let bound = tol.rel * norm + tol.abs * (reference.data.len() as f64).sqrt();
    let ok = diff <= bound;
    checks.push(check(
        &format!("{name}.max_abs_diff"),
        true,
        Some(max_abs),
        None,
        String::new(),
    ));
    checks.push(check(
        &format!("{name}.norm_rel_diff"),
        ok,
        Some(if norm > 0.0 { diff / norm } else { diff }),
        Some(tol.rel),
        format!("‖Δ‖ {diff:.3e} against a bound of {bound:.3e} (‖ref‖ {norm:.3e})"),
    ));
    Comparison { checks, passed: ok }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[u64], data: &[f32]) -> TensorF32 {
        TensorF32::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    #[test]
    fn exact_passes_and_deviation_fails() {
        let tol = ToleranceV1 {
            abs: 1e-4,
            rel: 1e-4,
        };
        let r = t(&[2], &[1.0, 100.0]);
        assert!(compare_f32("c", &t(&[2], &[1.0, 100.0]), &r, tol).passed);
        assert!(compare_f32("c", &t(&[2], &[1.0 + 5e-5, 100.0 + 1e-3]), &r, tol).passed);
        assert!(!compare_f32("c", &t(&[2], &[1.0 + 5e-4, 100.0]), &r, tol).passed);
        assert!(!compare_f32("c", &t(&[2], &[1.0, 100.5]), &r, tol).passed);
        let c = compare_f32("c", &t(&[2], &[f32::NAN, 100.0]), &r, tol);
        assert!(!c.passed);
        assert!(c.checks.iter().any(|k| k.name == "c.finite" && !k.passed));
        let c = compare_f32("c", &t(&[1, 2], &[1.0, 100.0]), &r, tol);
        assert!(!c.passed);
        assert_eq!(c.checks.len(), 1);
    }

    /// The norm rule (ADR-0031) passes rounding spread over a tensor, even where an element-wise
    /// rule fails on near-zero values, and refuses a computation that differs: a deviation of a
    /// few percent of the norm, whether spread or concentrated on one element.
    #[test]
    fn norm_rule_passes_rounding_and_refuses_a_different_computation() {
        let tol = ToleranceV1 {
            abs: 1e-7,
            rel: 2e-2,
        };
        let n = 10_000usize;
        let reference: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 1e-2).collect();
        let r = t(&[n as u64], &reference);
        let norm = reference
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        // Rounding: 0.5 % of each element, alternating signs; element-wise v1 refuses it where
        // the value crosses zero.
        let rounded: Vec<f32> = reference
            .iter()
            .enumerate()
            .map(|(i, v)| v * (1.0 + if i % 2 == 0 { 5e-3 } else { -5e-3 }) + 1e-6)
            .collect();
        assert!(compare_f32_norm("g", &t(&[n as u64], &rounded), &r, tol).passed);
        assert!(
            !compare_f32(
                "g",
                &t(&[n as u64], &rounded),
                &r,
                ToleranceV1 {
                    abs: 1e-9,
                    rel: 1e-3
                }
            )
            .passed
        );
        // A gradient scaled by 3 %: refused.
        let scaled: Vec<f32> = reference.iter().map(|v| v * 1.03).collect();
        assert!(!compare_f32_norm("g", &t(&[n as u64], &scaled), &r, tol).passed);
        // The same energy on one element: refused.
        let mut spike = reference.clone();
        spike[17] += (0.03 * norm) as f32;
        let c = compare_f32_norm("g", &t(&[n as u64], &spike), &r, tol);
        assert!(!c.passed);
        assert!(
            c.checks
                .iter()
                .any(|k| k.name == "g.norm_rel_diff" && !k.passed)
        );
        // Shape and finiteness first.
        assert!(!compare_f32_norm("g", &t(&[1, n as u64], &reference), &r, tol).passed);
        let mut nan = reference.clone();
        nan[3] = f32::NAN;
        assert!(!compare_f32_norm("g", &t(&[n as u64], &nan), &r, tol).passed);
    }
}
