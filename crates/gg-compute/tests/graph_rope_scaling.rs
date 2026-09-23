//! RoPE scaling (ADR-0022): explicit frequencies and amplitude.
//!
//! The oracles do not copy the `transformers` formulas — those are compared against the reference
//! itself in `gg-python/tests/test_rope_scaling.py`. They check what each variant **promises**: linear
//! interpolation slows the rotation by an exact factor, Llama 3.1 leaves high frequencies intact
//! and divides low ones, YaRN amplifies by a known factor, and all of them keep the property that
//! defines RoPE.

use std::collections::BTreeMap;

use gg_compute::graph::exec::{self, Prepared, position_frequencies, position_table};
use gg_compute::graph::rope::{self, RopeScaling};
use gg_compute::graph::templates::{Positions, TransformerDims, rotary, transformer};
use gg_compute::graph::{
    self, FORMAT, GraphError, Init, Loss, ModelGraphV1, Node, ParamSpec, Rotary, TensorSpec,
};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;
use serde_json::json;

fn table_param(name: &str, shape: Vec<u64>, init: Init) -> ParamSpec {
    ParamSpec {
        name: name.into(),
        shape,
        init,
        trainable: false,
    }
}

/// Minimal graph: one input, one neutral trained weight, and the given table added.
fn with_table(shape: Vec<u64>, init: Init) -> ModelGraphV1 {
    ModelGraphV1 {
        format: FORMAT.into(),
        name: "table".into(),
        inputs: vec![TensorSpec {
            name: "x".into(),
            shape: vec![0, 4],
            dtype: "f32".into(),
        }],
        parameters: vec![
            ParamSpec {
                name: "w".into(),
                shape: vec![4, 4],
                init: Init::XavierUniform,
                trainable: true,
            },
            table_param("tab", shape, init),
        ],
        nodes: vec![Node {
            op: "MatMul".into(),
            inputs: vec!["x".into(), "w".into()],
            output: "y".into(),
            attrs: BTreeMap::new(),
        }],
        functions: Vec::new(),
        loss: Loss::MeanSquaredError {
            prediction: "y".into(),
            target: TensorSpec {
                name: "t".into(),
                shape: vec![0, 4],
                dtype: "f32".into(),
            },
        },
    }
}

fn shape_error(graph: &ModelGraphV1) -> String {
    match graph::validate(graph) {
        Err(GraphError::Shape(m)) => m,
        other => panic!("expected a shape refusal, got {other:?}"),
    }
}

#[test]
fn explicit_frequencies_are_validated() {
    let ok = |r: Rotary| graph::validate(&with_table(vec![1, 1, 16, 8], Init::RotaryCos(r)));
    assert!(ok(Rotary::explicit(vec![1.0, 0.1, 0.01, 0.001], 1.2)).is_ok());
    // A zero frequency is allowed: "proportional" RoPE leaves some pairs unrotated.
    assert!(ok(Rotary::explicit(vec![1.0, 0.1, 0.0, 0.0], 1.0)).is_ok());
    // The amplitude is optional.
    assert!(
        ok(Rotary {
            frequencies: Some(vec![1.0, 0.1, 0.01, 0.001]),
            ..Rotary::default()
        })
        .is_ok()
    );

    let refused = [
        (
            Rotary {
                base: Some(10000.0),
                frequencies: Some(vec![1.0; 4]),
                amplitude: None,
            },
            "not both",
        ),
        (Rotary::default(), "are expected"),
        (
            Rotary {
                amplitude: Some(1.1),
                ..Rotary::from_base(10000.0)
            },
            "only goes with",
        ),
        (Rotary::explicit(vec![1.0, 0.1, 0.01], 1.0), "3 frequencies"),
        (
            Rotary::explicit(vec![1.0, f64::NAN, 0.01, 0.0], 1.0),
            "frequency 1",
        ),
        (
            Rotary::explicit(vec![1.0, 0.1, -0.01, 0.0], 1.0),
            "frequency 2",
        ),
        (
            Rotary::explicit(vec![1.0, 0.1, 0.01, f64::INFINITY], 1.0),
            "frequency 3",
        ),
        (Rotary::explicit(vec![1.0; 4], 0.0), "amplitude"),
        (Rotary::explicit(vec![1.0; 4], f64::NAN), "amplitude"),
        (Rotary::explicit(vec![1.0; 4], -1.0), "amplitude"),
    ];
    for (rotary, needle) in refused {
        let m = shape_error(&with_table(
            vec![1, 1, 16, 8],
            Init::RotarySin(rotary.clone()),
        ));
        assert!(
            m.contains(needle) && m.contains("'tab'"),
            "{rotary:?}: \"{m}\" does not name \"{needle}\""
        );
    }
}

/// Adds an unknown field somewhere in a description.
type Patch = fn(&mut serde_json::Value);

/// A field the worker does not understand is refused, never ignored: otherwise a worker running an
/// older version would silently compute a different function from the one described.
#[test]
fn an_unknown_field_is_refused_everywhere() {
    let base = serde_json::to_value(with_table(
        vec![1, 1, 16, 8],
        Init::RotaryCos(Rotary::explicit(vec![1.0, 0.5, 0.25, 0.125], 1.0)),
    ))
    .unwrap();
    assert!(serde_json::from_value::<ModelGraphV1>(base.clone()).is_ok());
    let cases: [(&str, Patch); 7] = [
        ("root", |v| v["comment"] = json!("x")),
        ("input", |v| v["inputs"][0]["layout"] = json!("nchw")),
        ("parameter", |v| v["parameters"][0]["lr"] = json!(0.1)),
        ("node", |v| v["nodes"][0]["fused"] = json!(true)),
        ("loss", |v| {
            v["loss"]["mean_squared_error"]["weight"] = json!(2.0)
        }),
        ("init", |v| {
            v["parameters"][0]["init"] = json!({"normal": {"std": 0.1, "mean": 1.0}})
        }),
        ("rotary table", |v| {
            v["parameters"][1]["init"]["rotary_cos"]["scaling"] = json!("yarn")
        }),
    ];
    for (place, patch) in cases {
        let mut v = base.clone();
        patch(&mut v);
        let err = serde_json::from_value::<ModelGraphV1>(v)
            .expect_err(&format!("unknown field accepted ({place})"));
        assert!(err.to_string().contains("unknown field"), "{place}: {err}");
    }
}

#[test]
fn an_explicit_table_follows_its_formula() {
    let (t_len, d) = (256usize, 16usize);
    let freqs: Vec<f64> = (0..d / 2).map(|i| 0.9f64.powi(3 * i as i32)).collect();
    let amplitude = 1.3465735902799727;
    for (init, trig) in [
        (
            Init::RotaryCos(Rotary::explicit(freqs.clone(), amplitude)),
            f64::cos as fn(f64) -> f64,
        ),
        (
            Init::RotarySin(Rotary::explicit(freqs.clone(), amplitude)),
            f64::sin,
        ),
    ] {
        let table = position_table(&init, &[1, 1, t_len as u64, d as u64]);
        let mut worst = 0f64;
        for t in 0..t_len {
            for j in 0..d {
                let w = freqs[j % (d / 2)] as f32 as f64;
                let exact = amplitude * trig(t as f64 * w);
                worst = worst.max((table[t * d + j] as f64 - exact).abs());
            }
        }
        // The angle t·ω is rounded to f32: the gap grows with the position, ~ t · 2^-24.
        assert!(worst <= 3e-5, "{init:?}: gap {worst:e}");
    }
}

#[test]
fn the_explicit_form_of_the_original_law_gives_the_same_table() {
    for (base, d) in [(10000.0, 64usize), (500000.0, 128), (1_000_000.0, 8)] {
        let explicit: Vec<f64> = position_frequencies(base, d)
            .into_iter()
            .map(f64::from)
            .collect();
        for (a, b) in [
            (
                Init::RotaryCos(Rotary::from_base(base)),
                Init::RotaryCos(Rotary::explicit(explicit.clone(), 1.0)),
            ),
            (
                Init::RotarySin(Rotary::from_base(base)),
                Init::RotarySin(Rotary::explicit(explicit.clone(), 1.0)),
            ),
        ] {
            let shape = [1, 1, 512, d as u64];
            assert_eq!(position_table(&a, &shape), position_table(&b, &shape));
        }
    }
}

fn frequencies_of(rotary: &Rotary, d: usize) -> (Vec<f32>, f32) {
    rotary.frequencies_f32(d)
}

/// Position interpolation slows the rotation by an exact factor: at position `s·u`, the scaled
/// table equals the original table at position `u`. For a power-of-two factor, dividing then
/// multiplying is exact in binary: the rows are byte-identical.
#[test]
fn linear_interpolates_the_positions() {
    let (d, t_len) = (32u64, 64u64);
    let original = position_table(
        &Init::RotaryCos(Rotary::from_base(10000.0)),
        &[1, 1, t_len, d],
    );
    for factor in [2.0, 4.0, 3.0] {
        let scaled = rope::rotary(10000.0, d, t_len, &RopeScaling::Linear { factor }).unwrap();
        let table = position_table(&Init::RotaryCos(scaled), &[1, 1, t_len, d]);
        let s = factor as usize;
        let mut worst = 0f32;
        for u in 0..t_len as usize / s {
            let a = &table[s * u * d as usize..(s * u + 1) * d as usize];
            let b = &original[u * d as usize..(u + 1) * d as usize];
            worst = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(worst, f32::max);
        }
        if factor.log2().fract() == 0.0 {
            assert_eq!(worst, 0.0, "factor {factor}: rows not identical");
        } else {
            assert!(worst <= 1e-5, "factor {factor}: gap {worst:e}");
        }
    }
}

/// Llama 3.1: a wavelength shorter than `original / high_freq_factor` keeps its frequency, one
/// longer than `original / low_freq_factor` is divided by `factor`, and in between the frequency
/// stays bounded by these two values.
#[test]
fn llama3_keeps_high_frequencies_and_divides_low_ones() {
    let d = 128usize;
    let scaling = RopeScaling::Llama3 {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_positions: 8192,
    };
    let (scaled, amplitude) =
        frequencies_of(&rope::rotary(500000.0, 128, 64, &scaling).unwrap(), d);
    let original = position_frequencies(500000.0, d);
    assert_eq!(amplitude, 1.0);
    let (mut high, mut low, mut medium) = (0, 0, 0);
    for (w, o) in scaled.iter().zip(&original) {
        let wavelen = 2.0 * std::f64::consts::PI / *o as f64;
        if wavelen < 8192.0 / 4.0 {
            assert_eq!(w, o, "high frequency modified");
            high += 1;
        } else if wavelen > 8192.0 {
            assert_eq!(*w, o / 8.0, "low frequency not divided");
            low += 1;
        } else {
            assert!(*w <= *o && *w >= o / 8.0, "{w} outside [{}, {o}]", o / 8.0);
            medium += 1;
        }
    }
    assert!(high > 0 && low > 0 && medium > 0, "{high} {low} {medium}");
    assert!(
        scaled.windows(2).all(|p| p[1] < p[0]),
        "frequencies not decreasing"
    );
}

/// YaRN: amplitude `0.1·ln(factor) + 1`, the highest frequency intact, the lowest interpolated by
/// an exact factor, and all of them bounded.
#[test]
fn yarn_amplifies_and_bounds() {
    let d = 128usize;
    let scaling = |attention_factor| RopeScaling::Yarn {
        factor: 4.0,
        original_max_positions: 32768,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor,
        mscale: None,
        mscale_all_dim: None,
        truncate: true,
    };
    let (scaled, amplitude) = frequencies_of(
        &rope::rotary(1_000_000.0, 128, 64, &scaling(None)).unwrap(),
        d,
    );
    let original = position_frequencies(1_000_000.0, d);
    assert_eq!(amplitude, (0.1 * 4f64.ln() + 1.0) as f32);
    assert_eq!(scaled[0], original[0]);
    let last = d / 2 - 1;
    assert!((scaled[last] / (original[last] / 4.0) - 1.0).abs() <= 1e-6);
    for (w, o) in scaled.iter().zip(&original) {
        assert!(*w <= *o * (1.0 + 1e-6) && *w >= o / 4.0 * (1.0 - 1e-6));
    }
    // An imposed amplitude takes precedence over the derived one.
    let (_, forced) = frequencies_of(
        &rope::rotary(1_000_000.0, 128, 64, &scaling(Some(1.25))).unwrap(),
        d,
    );
    assert_eq!(forced, 1.25);
}

#[test]
fn longrope_picks_its_factors_by_sequence_length() {
    let d = 16usize;
    let short: Vec<f64> = (0..8).map(|i| 1.0 + 0.01 * i as f64).collect();
    let long: Vec<f64> = (0..8).map(|i| 1.0 + 0.5 * i as f64).collect();
    let scaling = RopeScaling::LongRope {
        short_factor: short.clone(),
        long_factor: long.clone(),
        factor: 32.0,
        original_max_positions: 4096,
        attention_factor: None,
    };
    let original = position_frequencies(10000.0, d);
    for (seq, ext) in [(4096u64, &short), (4097, &long)] {
        let (w, a) = frequencies_of(&rope::rotary(10000.0, 16, seq, &scaling).unwrap(), d);
        let expected_a = (1.0 + 32f64.ln() / 4096f64.ln()).sqrt();
        assert_eq!(a, expected_a as f32);
        for i in 0..d / 2 {
            let expected = original[i] as f64 / ext[i];
            assert!(
                (w[i] as f64 / expected - 1.0).abs() <= 3e-7,
                "sequence {seq}, frequency {i}: {} vs {expected}",
                w[i]
            );
        }
    }
    let wrong = RopeScaling::LongRope {
        short_factor: vec![1.0; 3],
        long_factor: long,
        factor: 32.0,
        original_max_positions: 4096,
        attention_factor: None,
    };
    assert!(
        rope::rotary(10000.0, 16, 64, &wrong)
            .unwrap_err()
            .contains("3 factors")
    );
}

#[test]
fn dynamic_ntk_acts_only_beyond_the_context() {
    let scaling = RopeScaling::Dynamic {
        factor: 2.0,
        max_positions: 2048,
    };
    assert_eq!(
        rope::rotary(10000.0, 64, 2048, &scaling).unwrap(),
        Rotary::from_base(10000.0)
    );
    let grown = rope::rotary(10000.0, 64, 4096, &scaling).unwrap();
    let base = grown.base.expect("NTK changes the base, not the law");
    // (2·4096/2048 − 1)^(64/62) = 3^(32/31).
    let expected = 10000.0 * 3f64.powf(64.0 / 62.0);
    assert!(
        (base / expected - 1.0).abs() <= 1e-6,
        "{base} vs {expected}"
    );
}

/// The property that defines RoPE holds for any frequency law and any amplitude: the dot product
/// after rotation depends only on the distance between positions (up to the factor A²).
#[test]
fn every_scaling_keeps_the_relative_property() {
    let (t_len, d) = (32u64, 16u64);
    let yarn = RopeScaling::Yarn {
        factor: 8.0,
        original_max_positions: 16,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: None,
        mscale: None,
        mscale_all_dim: None,
        truncate: false,
    };
    let llama3 = RopeScaling::Llama3 {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_positions: 16,
    };
    for scaling in [yarn, llama3] {
        let r = rope::rotary(10000.0, d, t_len, &scaling).unwrap();
        let (_, amplitude) = r.frequencies_f32(d as usize);
        let table = [1, 1, t_len, d];
        let mut nodes = vec![Node {
            op: "Mul".into(),
            inputs: vec!["x".into(), "w".into()],
            output: "xs".into(),
            attrs: BTreeMap::new(),
        }];
        rotary(&mut nodes, "xs", "rot", d);
        let graph = ModelGraphV1 {
            format: FORMAT.into(),
            name: "rope".into(),
            inputs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![0, 1, t_len, d],
                dtype: "f32".into(),
            }],
            parameters: vec![
                table_param("rope.cos", table.to_vec(), Init::RotaryCos(r.clone())),
                table_param("rope.sin", table.to_vec(), Init::RotarySin(r.clone())),
                table_param("rope.neg", vec![1], Init::Constant { value: -1.0 }),
                ParamSpec {
                    name: "w".into(),
                    shape: vec![1],
                    init: Init::Ones,
                    trainable: true,
                },
            ],
            nodes,
            functions: Vec::new(),
            loss: Loss::MeanSquaredError {
                prediction: "rot".into(),
                target: TensorSpec {
                    name: "y".into(),
                    shape: vec![0, 1, t_len, d],
                    dtype: "f32".into(),
                },
            },
        };
        let prepared = Prepared::new(&graph).expect("valid composition");
        let params = exec::init_parameters(&graph, 0);
        let mut rng = Drbg::new("rope-scaling/vectors", 7);
        let u: Vec<f32> = (0..d).map(|_| rng.next_sym_f32()).collect();
        let v: Vec<f32> = (0..d).map(|_| rng.next_sym_f32()).collect();
        let norm: f64 = u.iter().map(|a| *a as f64 * *a as f64).sum();
        let rotated_dot = |m: usize, n: usize| -> (f64, f64) {
            let du = d as usize;
            let mut x = vec![0f32; (t_len * d) as usize];
            x[m * du..(m + 1) * du].copy_from_slice(&u);
            x[n * du..(n + 1) * du].copy_from_slice(&v);
            let out = exec::forward_cpu(
                &prepared,
                &params,
                &BTreeMap::from([(
                    "x".to_string(),
                    TensorF32::new(vec![1, 1, t_len, d], x).unwrap(),
                )]),
                0,
                "rot",
            )
            .unwrap();
            let a = &out.data[m * du..(m + 1) * du];
            let b = &out.data[n * du..(n + 1) * du];
            let dot = a.iter().zip(b).map(|(p, q)| *p as f64 * *q as f64).sum();
            let own = a.iter().map(|p| *p as f64 * *p as f64).sum();
            (dot, own)
        };
        for (m, n, shift) in [(2usize, 5usize, 20usize), (0, 3, 28), (9, 1, 17)] {
            let (here, own) = rotated_dot(m, n);
            let (there, _) = rotated_dot(m + shift, n + shift);
            assert!(
                (here - there).abs() <= 2e-5,
                "{scaling:?} ({m}, {n}) → +{shift}: {here} vs {there}"
            );
            // A rotation does not change the norm: |u|² remains, multiplied by A².
            let a2 = (amplitude as f64).powi(2);
            assert!(
                (own - norm * a2).abs() <= 1e-5,
                "{scaling:?}: |rot(u)|² = {own} vs |u|²·A² = {}",
                norm * a2
            );
        }
    }
}

#[test]
fn hugging_face_configurations_are_read() {
    let read = |v: serde_json::Value| RopeScaling::from_hugging_face(&v, 131072);
    let (s, theta) = read(json!({
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    }))
    .unwrap();
    assert!(matches!(s, RopeScaling::Llama3 { factor, .. } if factor == 8.0));
    assert_eq!(theta, None);
    // transformers 5: `rope_parameters` also carries the base.
    let (_, theta) = read(json!({"rope_type": "default", "rope_theta": 500000.0})).unwrap();
    assert_eq!(theta, Some(500000.0));
    // transformers 4 before `rope_type`: the key was called `type`.
    assert!(matches!(
        read(json!({"type": "linear", "factor": 2.0})).unwrap().0,
        RopeScaling::Linear { factor } if factor == 2.0
    ));
    // YaRN without a factor: derived from the context ratio, as in the reference.
    assert!(matches!(
        read(json!({"rope_type": "yarn", "original_max_position_embeddings": 4096})).unwrap().0,
        RopeScaling::Yarn { factor, beta_fast, beta_slow, truncate: true, .. }
            if factor == 32.0 && beta_fast == 32.0 && beta_slow == 1.0
    ));

    let refusals = [
        (
            json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096,
                   "mrope_section": [16, 24, 24]}),
            "unknown key 'mrope_section'",
        ),
        (
            json!({"rope_type": "ntk-by-parts", "factor": 2.0}),
            "variant 'ntk-by-parts'",
        ),
        (json!({"rope_type": "linear"}), "requires 'factor'"),
        (
            json!({"rope_type": "linear", "factor": -2.0}),
            "strictly positive",
        ),
        (
            json!({"rope_type": "linear", "factor": "8"}),
            "finite number",
        ),
        (
            json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 4.0,
                   "high_freq_factor": 1.0, "original_max_position_embeddings": 8192}),
            "must exceed",
        ),
        (
            json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                   "high_freq_factor": 4.0, "original_max_position_embeddings": 8192.5}),
            "an integer",
        ),
        (
            json!({"rope_type": "default", "partial_rotary_factor": 0.5}),
            "partial rotation",
        ),
        (
            json!({"rope_type": "longrope", "short_factor": [1.0, 0.0],
                   "long_factor": [1.0, 1.0], "original_max_position_embeddings": 4096}),
            "finite numbers > 0",
        ),
        (json!([1, 2]), "JSON object"),
    ];
    for (v, needle) in refusals {
        let err = read(v.clone()).unwrap_err();
        assert!(
            err.contains(needle),
            "{v}: \"{err}\" does not say \"{needle}\""
        );
    }
}

/// The scaled template keeps its structure and counts; only the tables change, and the
/// description survives a JSON round trip without losing a bit of its frequencies.
#[test]
fn the_template_carries_its_scaling() {
    let dims = |positions| TransformerDims {
        layers: 2,
        dim: 64,
        heads: 4,
        seq: 32,
        vocab: 50,
        ffn: 128,
        causal: true,
        positions,
    };
    let plain = transformer(dims(Positions::rotary(500000.0))).unwrap();
    let scaled_dims = dims(Positions::Rotary {
        base: 500000.0,
        scaling: RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_positions: 16,
        },
    });
    let scaled = transformer(scaled_dims.clone()).unwrap();
    let (a, b) = (
        graph::validate(&plain).unwrap(),
        graph::validate(&scaled).unwrap(),
    );
    assert_eq!(a.order.len(), b.order.len());
    assert_eq!(a.parameters, b.parameters);
    assert_eq!(b.parameters, scaled_dims.parameters());
    assert_eq!(a.trainable_parameters, b.trainable_parameters);
    let cos = |g: &ModelGraphV1| {
        g.parameters
            .iter()
            .find(|p| p.name == "rope.cos")
            .unwrap()
            .init
            .clone()
    };
    assert!(matches!(
        cos(&plain),
        Init::RotaryCos(Rotary { base: Some(_), .. })
    ));
    assert!(matches!(
        cos(&scaled),
        Init::RotaryCos(Rotary {
            frequencies: Some(_),
            ..
        })
    ));
    let text = serde_json::to_string(&scaled).unwrap();
    let back: ModelGraphV1 = serde_json::from_str(&text).unwrap();
    assert_eq!(back, scaled);
    let shape = [1, 1, 32, 16];
    assert_eq!(
        position_table(&cos(&back), &shape),
        position_table(&cos(&scaled), &shape)
    );
}
