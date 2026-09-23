//! RoPE scalings, as Hugging Face configurations describe them (ADR-0022).
//!
//! A scaling is only a way of computing the frequencies of a rotary table, plus, for YaRN and
//! LongRoPE, an amplitude. The executor knows none of them: it receives explicit frequencies
//! ([`Rotary`]). This module serves whoever writes a description without going through PyTorch
//! (the `gg job template transformer --rope-scaling` skeleton) and reproduces the formulas of
//! `transformers` (`modeling_rope_utils.py`) **in their f32 order of operations**: `torch` converts
//! each Python scalar to f32 before the operation, and `s / x` there equals `reciprocal(x) · s`.
//! What remains is the f32 power, which `torch` evaluates with SLEEF and Rust with the system libm:
//! the two sometimes differ by one unit in the last place. Measured against `transformers`
//! (`gg-python/tests/test_rope_scaling.py`): 14 variants out of 16 byte-identical, the other two
//! within 1 and 2 ulp.
//!
//! The PyTorch driver does not use these formulas: it recognizes tables by their values and
//! recovers their frequencies, whatever the variant that produced them.

use serde_json::Value;

use super::Rotary;

/// Scaling variant, with resolved parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    /// Original law, `ω_i = 1 / base^(2i/d)`.
    None,
    /// Position interpolation: all frequencies divided by `factor`.
    Linear {
        /// Context extension factor.
        factor: f64,
    },
    /// Dynamic NTK: the base grows when the sequence exceeds `max_positions`, otherwise nothing.
    Dynamic {
        /// Factor.
        factor: f64,
        /// The model's `max_position_embeddings`.
        max_positions: u64,
    },
    /// YaRN: ramp between extrapolated (high) and interpolated (low) frequencies, and amplitude.
    Yarn {
        /// Factor, resolved (`max_position_embeddings / original_max_position_embeddings` when
        /// missing).
        factor: f64,
        /// Original context.
        original_max_positions: u64,
        /// Number of rotations above which a frequency is not interpolated (32 by default).
        beta_fast: f64,
        /// Number of rotations below which it is fully interpolated (1 by default).
        beta_slow: f64,
        /// Imposed amplitude; otherwise derived from `factor`, `mscale` and `mscale_all_dim`.
        attention_factor: Option<f64>,
        /// Numerator of the derived amplitude (DeepSeek).
        mscale: Option<f64>,
        /// Denominator of the derived amplitude (DeepSeek).
        mscale_all_dim: Option<f64>,
        /// Round the ramp bounds to integers (true by default).
        truncate: bool,
    },
    /// LongRoPE (Phi-3): one factor per frequency, short or long by sequence length, and amplitude.
    LongRope {
        /// Factors for a sequence that fits in the original context.
        short_factor: Vec<f64>,
        /// Factors beyond it.
        long_factor: Vec<f64>,
        /// Factor, resolved as for YaRN.
        factor: f64,
        /// Original context.
        original_max_positions: u64,
        /// Imposed amplitude; otherwise `sqrt(1 + ln(factor) / ln(original_max_positions))`.
        attention_factor: Option<f64>,
    },
    /// Llama 3.1: high frequencies untouched, low ones divided by `factor`, smoothed transition.
    Llama3 {
        /// Factor applied to low frequencies (8 for Llama 3.1).
        factor: f64,
        /// Low-frequency threshold, as a fraction of the original context (1).
        low_freq_factor: f64,
        /// High-frequency threshold (4).
        high_freq_factor: f64,
        /// Original context (8,192).
        original_max_positions: u64,
    },
}

/// Recognized keys of a `rope_scaling` / `rope_parameters` dictionary. An unknown key is rejected
/// rather than ignored: `mrope_section`, for example, changes the layout of the tables.
const KNOWN_KEYS: &[&str] = &[
    "rope_type",
    "type",
    "rope_theta",
    "factor",
    "original_max_position_embeddings",
    "low_freq_factor",
    "high_freq_factor",
    "beta_fast",
    "beta_slow",
    "mscale",
    "mscale_all_dim",
    "attention_factor",
    "truncate",
    "short_factor",
    "long_factor",
    "partial_rotary_factor",
];

fn number(map: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() => Ok(Some(x)),
            _ => Err(format!("'{key}' must be a finite number, got {v}")),
        },
    }
}

fn positive(map: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    match number(map, key)? {
        Some(x) if x <= 0.0 => Err(format!("'{key}' must be strictly positive, got {x}")),
        other => Ok(other),
    }
}

fn required(map: &serde_json::Map<String, Value>, key: &str, kind: &str) -> Result<f64, String> {
    positive(map, key)?.ok_or_else(|| format!("variant '{kind}' requires '{key}'"))
}

fn positions(map: &serde_json::Map<String, Value>, key: &str, kind: &str) -> Result<u64, String> {
    let x = required(map, key, kind)?;
    if x.fract() != 0.0 || x > u32::MAX as f64 {
        return Err(format!("'{key}' must be an integer, got {x}"));
    }
    Ok(x as u64)
}

fn factors(map: &serde_json::Map<String, Value>, key: &str) -> Result<Vec<f64>, String> {
    let list = map
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("variant 'longrope' requires the list '{key}'"))?;
    list.iter()
        .map(|v| match v.as_f64() {
            Some(x) if x.is_finite() && x > 0.0 => Ok(x),
            _ => Err(format!(
                "'{key}' must contain only finite numbers > 0, got {v}"
            )),
        })
        .collect()
}

impl RopeScaling {
    /// Reads a `rope_scaling` (transformers 4) or `rope_parameters` (transformers 5) dictionary, as
    /// found in a model's `config.json`. `max_positions` is the model's `max_position_embeddings`,
    /// which some variants consult. Returns the variant and the `rope_theta` base if the dictionary
    /// carries it.
    pub fn from_hugging_face(v: &Value, max_positions: u64) -> Result<(Self, Option<f64>), String> {
        let map = v
            .as_object()
            .ok_or("the scaling must be a JSON object, like the `rope_scaling` of a config.json")?;
        if let Some(key) = map.keys().find(|k| !KNOWN_KEYS.contains(&k.as_str())) {
            return Err(format!(
                "unknown key '{key}': it could change the computed function, so it is rejected \
rather than ignored"
            ));
        }
        let kind = map
            .get("rope_type")
            .or_else(|| map.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        let theta = positive(map, "rope_theta")?;
        if let Some(p) = number(map, "partial_rotary_factor")?
            && p != 1.0
        {
            return Err(format!(
                "partial rotation ({p}): the skeleton rotates the whole head; the PyTorch driver \
captures a model that rotates only part of it"
            ));
        }
        let original = |kind: &str| positions(map, "original_max_position_embeddings", kind);
        // `factor` missing: YaRN and LongRoPE derive it from the ratio of the contexts.
        let resolved_factor = |kind: &str| -> Result<f64, String> {
            match positive(map, "factor")? {
                Some(f) => Ok(f),
                None => Ok(max_positions as f64 / original(kind)? as f64),
            }
        };
        let scaling = match kind {
            "default" => RopeScaling::None,
            "linear" => RopeScaling::Linear {
                factor: required(map, "factor", kind)?,
            },
            "dynamic" => RopeScaling::Dynamic {
                factor: required(map, "factor", kind)?,
                max_positions,
            },
            "yarn" => RopeScaling::Yarn {
                factor: resolved_factor(kind)?,
                original_max_positions: original(kind)?,
                // `or 32` in the reference: zero also means the default value.
                beta_fast: number(map, "beta_fast")?
                    .filter(|b| *b != 0.0)
                    .unwrap_or(32.0),
                beta_slow: number(map, "beta_slow")?
                    .filter(|b| *b != 0.0)
                    .unwrap_or(1.0),
                attention_factor: positive(map, "attention_factor")?,
                mscale: number(map, "mscale")?,
                mscale_all_dim: number(map, "mscale_all_dim")?,
                truncate: match map.get("truncate") {
                    None | Some(Value::Null) => true,
                    Some(Value::Bool(b)) => *b,
                    Some(other) => {
                        return Err(format!("'truncate' must be a boolean, got {other}"));
                    }
                },
            },
            "longrope" => RopeScaling::LongRope {
                short_factor: factors(map, "short_factor")?,
                long_factor: factors(map, "long_factor")?,
                factor: resolved_factor(kind)?,
                original_max_positions: original(kind)?,
                attention_factor: positive(map, "attention_factor")?,
            },
            "llama3" => {
                let low = required(map, "low_freq_factor", kind)?;
                let high = required(map, "high_freq_factor", kind)?;
                if high <= low {
                    return Err(format!(
                        "'high_freq_factor' ({high}) must exceed 'low_freq_factor' ({low})"
                    ));
                }
                RopeScaling::Llama3 {
                    factor: required(map, "factor", kind)?,
                    low_freq_factor: low,
                    high_freq_factor: high,
                    original_max_positions: original(kind)?,
                }
            }
            other => {
                return Err(format!(
                    "unknown variant '{other}': default, linear, dynamic, yarn, longrope or llama3"
                ));
            }
        };
        Ok((scaling, theta))
    }
}

/// Exponents `2i/d` in f32, like `arange(0, d, 2).float() / d`.
fn exponents(d: usize) -> impl Iterator<Item = f32> {
    (0..d / 2).map(move |i| (2 * i) as f32 / d as f32)
}

/// Original law, `1 / base^(2i/d)` in f32.
fn default_frequencies(base: f64, d: usize) -> Vec<f32> {
    let b = base as f32;
    exponents(d).map(|e| 1.0 / b.powf(e)).collect()
}

fn mscale(scale: f64, m: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * m * scale.ln() + 1.0
    }
}

/// Frequencies of a rotary table of width `d` for a sequence of `seq` positions. Returns the
/// original law when there is no scaling (or when dynamic NTK does not apply), explicit
/// frequencies otherwise.
pub fn rotary(base: f64, d: u64, seq: u64, scaling: &RopeScaling) -> Result<Rotary, String> {
    if !(base.is_finite() && base > 1.0) {
        return Err(format!(
            "invalid frequency base {base}, expected finite and > 1"
        ));
    }
    if d < 2 || !d.is_multiple_of(2) {
        return Err(format!("rotation width {d}: expected even and non-zero"));
    }
    let dim = d as usize;
    let half = dim / 2;
    let explicit = |freqs: Vec<f32>, amplitude: f64| {
        Rotary::explicit(freqs.into_iter().map(f64::from).collect(), amplitude)
    };
    let rotary = match scaling {
        RopeScaling::None => Rotary::from_base(base),
        RopeScaling::Linear { factor } => {
            let f = *factor as f32;
            explicit(
                default_frequencies(base, dim)
                    .into_iter()
                    .map(|w| w / f)
                    .collect(),
                1.0,
            )
        }
        RopeScaling::Dynamic {
            factor,
            max_positions,
        } => {
            // Within the context, the reference keeps the original law. Beyond it, it recomputes
            // the base during the forward pass, with a tensor length: the computation is in f32.
            if seq <= *max_positions {
                Rotary::from_base(base)
            } else {
                let stretch =
                    (seq as f32 * *factor as f32) / *max_positions as f32 - (factor - 1.0) as f32;
                let grown = base as f32 * stretch.powf((d as f64 / (d as f64 - 2.0)) as f32);
                Rotary::from_base(f64::from(grown))
            }
        }
        RopeScaling::Yarn {
            factor,
            original_max_positions,
            beta_fast,
            beta_slow,
            attention_factor,
            mscale: m,
            mscale_all_dim,
            truncate,
        } => {
            let amplitude = match (attention_factor, m, mscale_all_dim) {
                (Some(a), _, _) => *a,
                (None, Some(m), Some(all)) if *m != 0.0 && *all != 0.0 => {
                    mscale(*factor, *m) / mscale(*factor, *all)
                }
                _ => mscale(*factor, 1.0),
            };
            let correction = |rotations: f64| {
                (d as f64
                    * (*original_max_positions as f64 / (rotations * 2.0 * std::f64::consts::PI))
                        .ln())
                    / (2.0 * base.ln())
            };
            let (mut low, mut high) = (correction(*beta_fast), correction(*beta_slow));
            if *truncate {
                low = low.floor();
                high = high.ceil();
            }
            let low = low.max(0.0);
            let mut high = high.min(d as f64 - 1.0);
            if low == high {
                high += 0.001;
            }
            let b = base as f32;
            let f = *factor as f32;
            let (low32, span32) = (low as f32, (high - low) as f32);
            let freqs = exponents(dim)
                .enumerate()
                .map(|(i, e)| {
                    let pos = b.powf(e);
                    let extrapolation = (1.0 / pos) * 1.0;
                    let interpolation = (1.0 / (f * pos)) * 1.0;
                    let ramp = ((i as f32 - low32) / span32).clamp(0.0, 1.0);
                    let keep = 1.0 - ramp;
                    interpolation * (1.0 - keep) + extrapolation * keep
                })
                .collect();
            explicit(freqs, amplitude)
        }
        RopeScaling::LongRope {
            short_factor,
            long_factor,
            factor,
            original_max_positions,
            attention_factor,
        } => {
            let ext = if seq > *original_max_positions {
                long_factor
            } else {
                short_factor
            };
            if ext.len() != half {
                return Err(format!(
                    "LongRoPE: {} factors for {half} frequencies",
                    ext.len()
                ));
            }
            let amplitude = attention_factor.unwrap_or_else(|| {
                if *factor <= 1.0 {
                    1.0
                } else {
                    (1.0 + factor.ln() / (*original_max_positions as f64).ln()).sqrt()
                }
            });
            let b = base as f32;
            let freqs = exponents(dim)
                .zip(ext)
                .map(|(e, x)| (1.0 / (*x as f32 * b.powf(e))) * 1.0)
                .collect();
            explicit(freqs, amplitude)
        }
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_positions,
        } => {
            let old = *original_max_positions as f64;
            let low_wavelen = (old / low_freq_factor) as f32;
            let high_wavelen = (old / high_freq_factor) as f32;
            let two_pi = (2.0 * std::f64::consts::PI) as f32;
            let (f, lo, span, old32) = (
                *factor as f32,
                *low_freq_factor as f32,
                (high_freq_factor - low_freq_factor) as f32,
                old as f32,
            );
            let freqs = default_frequencies(base, dim)
                .into_iter()
                .map(|w| {
                    let wavelen = (1.0 / w) * two_pi;
                    let llama = if wavelen > low_wavelen { w / f } else { w };
                    let smooth = ((1.0 / wavelen) * old32 - lo) / span;
                    let smoothed = (1.0 - smooth) * llama / f + smooth * llama;
                    // `!(a < b) && !(a > c)` in the reference; equivalent here, since a
                    // wavelength is never NaN.
                    let medium = wavelen >= high_wavelen && wavelen <= low_wavelen;
                    if medium { smoothed } else { llama }
                })
                .collect();
            explicit(freqs, 1.0)
        }
    };
    rotary.check(d)?;
    Ok(rotary)
}
