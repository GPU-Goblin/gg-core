//! Native CUDA engine (ADR-0033): gradients against the CPU reference, under the rule a validator
//! applies to model tasks (by norm, `catalog::graph_tolerance`), and against PyTorch's where a
//! fixture carries them. Runs only with `--features cuda`, with `NVIDIA_TF32_OVERRIDE=0`; fails (does
//! not skip) if CUDA is unavailable.
//!
//! ```text
//! NVIDIA_TF32_OVERRIDE=0 LIBTORCH_USE_PYTORCH=1 cargo test -p gg-compute --features cuda --test cuda
//! ```

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use gg_compute::graph::{self, exec};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::{BundleF32, TensorF32, TensorLimits};

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(path: &std::path::Path) -> (graph::ModelGraphV1, exec::Prepared) {
    let g: graph::ModelGraphV1 =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let prepared = exec::Prepared::new(&g).unwrap();
    (g, prepared)
}

#[test]
fn the_engine_refuses_to_run_outside_strict_float32() {
    // This process is started with the override (see the module header); the check itself says
    // what it requires when the variable is absent.
    assert_eq!(
        std::env::var("NVIDIA_TF32_OVERRIDE").as_deref(),
        Ok("0"),
        "run these tests with NVIDIA_TF32_OVERRIDE=0"
    );
    let info = gg_compute::engine::cuda::check(0).expect("CUDA device 0");
    assert_eq!(info.numeric_profile, "gpu-f32-libtorch-cuda-v1");
}

/// Every described fixture: the native CUDA gradient passes the validator's rule against the CPU.
#[test]
fn gradients_pass_the_model_task_rule_against_the_cpu() {
    let dir = fixtures();
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir.join("gpu"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    for f in [
        "llama-hf-mini.json",
        "mixtral-mini.json",
        "hifigan-mini.json",
        "wav2vec2-mini.json",
        "convnext-mini.json",
    ] {
        paths.push(dir.join(f));
    }
    paths.sort();
    let tol = gg_compute::catalog::graph_tolerance();
    for path in paths {
        let (g, prepared) = load(&path);
        let mut params = exec::init_parameters(&g, 3);
        let mut d = Drbg::new("test/cuda", 1);
        for spec in g.parameters.iter().filter(|s| s.trainable) {
            let t = params.get_mut(&spec.name).unwrap();
            if t.data.iter().all(|v| *v == 0.0) {
                for v in &mut t.data {
                    *v = 0.1 * d.next_sym_f32();
                }
            }
        }
        let target = g.loss.target().name.clone();
        let batch: BTreeMap<String, TensorF32> = g
            .inputs
            .iter()
            .chain(std::iter::once(g.loss.target()))
            .map(|s| {
                let shape: Vec<u64> = s
                    .shape
                    .iter()
                    .map(|x| if *x == 0 { 4 } else { *x })
                    .collect();
                let count: u64 = shape.iter().product();
                let data = (0..count)
                    .map(|_| {
                        if s.dtype == "i64" || s.name == target {
                            d.next_index(10) as f32
                        } else {
                            d.next_sym_f32()
                        }
                    })
                    .collect();
                (s.name.clone(), TensorF32::new(shape, data).unwrap())
            })
            .collect();
        let cpu = exec::gradient_cpu(&prepared, &params, &batch, 0).unwrap();
        let gpu = exec::torch::gradient_cuda(0, &prepared, params, batch, 0, Default::default())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            ((cpu.loss - gpu.loss) as f64).abs() <= 1e-4 * (cpu.loss as f64).abs().max(1.0),
            "{file}: loss cpu {} cuda {}",
            cpu.loss,
            gpu.loss
        );
        let mut worst = 0f64;
        for (name, c) in &cpu.gradients {
            let cmp = gg_compute::compare::compare_f32_norm(name, &gpu.gradients[name], c, tol);
            let rel = cmp
                .checks
                .iter()
                .find(|k| k.name.ends_with("norm_rel_diff"))
                .and_then(|k| k.measured)
                .unwrap_or(0.0);
            worst = worst.max(rel);
            assert!(cmp.passed, "{file}: {name}: {:?}", cmp.checks);
        }
        eprintln!(
            "{file}: loss {} (cpu {}), worst relative norm {worst:.2e}",
            gpu.loss, cpu.loss
        );
    }
}

/// Fixtures captured from PyTorch carry PyTorch's own loss and gradients: the native CUDA engine
/// finds them.
#[test]
fn gradients_match_pytorch_on_captured_fixtures() {
    let dir = fixtures();
    let bundle = |file: String| {
        BundleF32::decode(
            &std::fs::read(dir.join(file)).unwrap(),
            &TensorLimits::default(),
        )
        .unwrap()
        .entries
    };
    let tol = gg_compute::catalog::graph_tolerance();
    for name in [
        "hifigan-mini",
        "wav2vec2-mini",
        "convnext-mini",
        "mixtral-mini",
        "llama-tied-mini",
    ] {
        let (g, prepared) = load(&dir.join(format!("{name}.json")));
        let torch = bundle(format!("{name}.torch.ggb"));
        let torch_grad = bundle(format!("{name}.grad.ggb"));
        let params: BTreeMap<String, TensorF32> = g
            .parameters
            .iter()
            .map(|p| (p.name.clone(), torch[&p.name].clone()))
            .collect();
        let inputs = BTreeMap::from([
            (g.inputs[0].name.clone(), torch["__x"].clone()),
            (g.loss.target().name.clone(), torch["__y"].clone()),
        ]);
        let step = exec::torch::gradient_cuda(0, &prepared, params, inputs, 0, Default::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected = torch["__loss"].data[0];
        assert!(
            (step.loss - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "{name}: loss {} versus {expected} in PyTorch",
            step.loss
        );
        for (p, gr) in &step.gradients {
            let cmp = gg_compute::compare::compare_f32_norm(p, gr, &torch_grad[p], tol);
            assert!(cmp.passed, "{name}: {p}: {:?}", cmp.checks);
        }
    }
}

/// Which engine runs a chain of stages.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Chain {
    /// The CPU reference.
    Cpu,
    /// LibTorch with Burn's generic backward formulas (the stage path before ADR-0033's autograd).
    BurnOnTorch,
    /// LibTorch with PyTorch's autograd.
    Native,
}

/// The grid's staged path in one process: forward passes in order, backward passes in reverse.
/// Returns the loss, the summed parameter gradients and the time of the backward passes.
fn chain(
    engine: Chain,
    prepared: &exec::Prepared,
    stages: &[graph::stages::StageSpec],
    params: &BTreeMap<String, TensorF32>,
    data: &BTreeMap<String, TensorF32>,
) -> (f32, BTreeMap<String, TensorF32>, std::time::Duration) {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    let cuda = LibTorchDevice::Cuda(0);
    let mut produced = data.clone();
    let mut kept = Vec::new();
    for s in stages {
        let only: BTreeMap<String, TensorF32> = s
            .params
            .iter()
            .map(|n| (n.clone(), params[n].clone()))
            .collect();
        let ins: BTreeMap<String, TensorF32> = s
            .inputs
            .iter()
            .map(|n| (n.clone(), produced[n].clone()))
            .collect();
        let (out, _) = match engine {
            Chain::Cpu => exec::stage_forward::<burn::backend::Flex>(
                &burn::backend::flex::FlexDevice,
                prepared,
                s,
                &only,
                &ins,
                0,
            ),
            _ => exec::stage_forward::<LibTorch<f32>>(&cuda, prepared, s, &only, &ins, 0),
        }
        .unwrap();
        produced.extend(out);
        kept.push((only, ins));
    }
    let mut received: BTreeMap<String, Vec<TensorF32>> = BTreeMap::new();
    let mut total: BTreeMap<String, TensorF32> = BTreeMap::new();
    let mut loss = f32::NAN;
    let mut backward = std::time::Duration::ZERO;
    for (s, (only, ins)) in stages.iter().zip(&kept).rev() {
        let t = std::time::Instant::now();
        let r = match engine {
            Chain::Cpu => exec::stage_backward::<burn::backend::Autodiff<burn::backend::Flex>>(
                &burn::backend::flex::FlexDevice,
                prepared,
                s,
                only,
                ins,
                &received,
                0,
            ),
            Chain::BurnOnTorch => exec::stage_backward::<burn::backend::Autodiff<LibTorch<f32>>>(
                &cuda, prepared, s, only, ins, &received, 0,
            ),
            Chain::Native => {
                exec::torch::stage_backward_cuda(0, prepared, s, only, ins, &received, 0)
            }
        }
        .unwrap_or_else(|e| panic!("{engine:?}: {e}"));
        backward += t.elapsed();
        if let Some(l) = r.loss {
            loss = l;
        }
        for (name, g) in r.input_gradients {
            received.entry(name).or_default().push(g);
        }
        for (name, g) in r.gradients {
            match total.get_mut(&name) {
                Some(acc) => acc.data.iter_mut().zip(&g.data).for_each(|(a, b)| *a += *b),
                None => {
                    total.insert(name, g);
                }
            }
        }
    }
    (loss, total, backward)
}

/// Stage backward passes with PyTorch's autograd: chained across stages, they give the gradient the
/// CPU chain gives, under the validators' rule, for transformers and convolutional networks cut in
/// two and three stages. The time of the backward passes is reported against Burn's formulas on
/// the same card.
#[test]
fn native_stage_backward_passes_the_model_task_rule_against_the_cpu() {
    use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
    let dir = fixtures();
    let tol = gg_compute::catalog::graph_tolerance();
    let mut models: Vec<(String, graph::ModelGraphV1)> = vec![(
        "transformer (RoPE)".into(),
        transformer(TransformerDims {
            layers: 3,
            dim: 64,
            heads: 4,
            seq: 16,
            vocab: 50,
            ffn: 128,
            causal: true,
            positions: Positions::rotary(10000.0),
        })
        .unwrap(),
    )];
    for f in [
        "llama-hf-mini.json",
        "convnext-mini.json",
        "gpu/resnet18-32px.json",
    ] {
        models.push((f.into(), load(&dir.join(f)).0));
    }
    for (label, g) in models {
        let prepared = exec::Prepared::new(&g).unwrap();
        let report = graph::validate(&g).unwrap();
        let mut params = exec::init_parameters(&g, 3);
        let mut d = Drbg::new("test/cuda/stages", 1);
        for spec in g.parameters.iter().filter(|s| s.trainable) {
            let t = params.get_mut(&spec.name).unwrap();
            if t.data.iter().all(|v| *v == 0.0) {
                for v in &mut t.data {
                    *v = 0.1 * d.next_sym_f32()
                        + if spec.name.ends_with("norm.weight") {
                            1.0
                        } else {
                            0.0
                        };
                }
            }
        }
        let target = g.loss.target().name.clone();
        let data: BTreeMap<String, TensorF32> = g
            .inputs
            .iter()
            .chain(std::iter::once(g.loss.target()))
            .map(|s| {
                let shape: Vec<u64> = s
                    .shape
                    .iter()
                    .map(|x| if *x == 0 { 4 } else { *x })
                    .collect();
                let count: u64 = shape.iter().product();
                let data = (0..count)
                    .map(|_| {
                        if s.dtype == "i64" || s.name == target {
                            d.next_index(10) as f32
                        } else {
                            d.next_sym_f32()
                        }
                    })
                    .collect();
                (s.name.clone(), TensorF32::new(shape, data).unwrap())
            })
            .collect();
        for k in [2usize, 3] {
            let bounds = graph::stages::even_bounds(report.order.len(), k);
            let st = graph::stages::stages(&g, &report, &bounds).unwrap();
            let (cpu_loss, cpu, _) = chain(Chain::Cpu, &prepared, &st, &params, &data);
            // Once to warm up the card and its kernels, then measured.
            chain(Chain::Native, &prepared, &st, &params, &data);
            let (loss, native, t_native) = chain(Chain::Native, &prepared, &st, &params, &data);
            chain(Chain::BurnOnTorch, &prepared, &st, &params, &data);
            let (_, _, t_burn) = chain(Chain::BurnOnTorch, &prepared, &st, &params, &data);
            assert!(
                ((cpu_loss - loss) as f64).abs() <= 1e-4 * (cpu_loss as f64).abs().max(1.0),
                "{label}, {k} stages: loss cpu {cpu_loss} cuda {loss}"
            );
            let mut worst = 0f64;
            for (name, c) in &cpu {
                let cmp = gg_compute::compare::compare_f32_norm(name, &native[name], c, tol);
                worst = worst.max(
                    cmp.checks
                        .iter()
                        .find(|k| k.name.ends_with("norm_rel_diff"))
                        .and_then(|k| k.measured)
                        .unwrap_or(0.0),
                );
                assert!(cmp.passed, "{label}, {k} stages: {name}: {:?}", cmp.checks);
            }
            eprintln!(
                "{label}, {k} stages: worst relative norm {worst:.2e}; backward passes {:.1} ms \
native against {:.1} ms with Burn's formulas",
                t_native.as_secs_f64() * 1e3,
                t_burn.as_secs_f64() * 1e3
            );
        }
    }
}

/// Measurement at real size, run explicitly: the backward passes of a model cut in two stages,
/// native against Burn's formulas on the same card.
///
/// ```text
/// GG_STAGE_BENCH=smollm.json GG_STAGE_BATCH=8 NVIDIA_TF32_OVERRIDE=0 cargo test --release \
///   -p gg-compute --features cuda --test cuda stage_backward_time -- --ignored --nocapture
/// ```
#[test]
#[ignore = "measurement: GG_STAGE_BENCH=<description.json>"]
fn stage_backward_time() {
    let path = std::env::var("GG_STAGE_BENCH").expect("GG_STAGE_BENCH");
    let n: u64 = std::env::var("GG_STAGE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let (g, prepared) = load(std::path::Path::new(&path));
    let report = graph::validate(&g).unwrap();
    let mut params = exec::init_parameters(&g, 3);
    let mut d = Drbg::new("test/cuda/bench", 1);
    for spec in g.parameters.iter().filter(|s| s.trainable) {
        let t = params.get_mut(&spec.name).unwrap();
        if t.data.iter().all(|v| *v == 0.0) {
            for v in &mut t.data {
                *v = 0.02 * d.next_sym_f32()
                    + if spec.name.ends_with("norm.weight") {
                        1.0
                    } else {
                        0.0
                    };
            }
        }
    }
    let target = g.loss.target().name.clone();
    let data: BTreeMap<String, TensorF32> = g
        .inputs
        .iter()
        .chain(std::iter::once(g.loss.target()))
        .map(|s| {
            let shape: Vec<u64> = s
                .shape
                .iter()
                .map(|x| if *x == 0 { n } else { *x })
                .collect();
            let count: u64 = shape.iter().product();
            let data = (0..count)
                .map(|_| {
                    if s.dtype == "i64" || s.name == target {
                        d.next_index(10) as f32
                    } else {
                        d.next_sym_f32()
                    }
                })
                .collect();
            (s.name.clone(), TensorF32::new(shape, data).unwrap())
        })
        .collect();
    let bounds = graph::stages::even_bounds(report.order.len(), 2);
    let st = graph::stages::stages(&g, &report, &bounds).unwrap();
    for engine in [Chain::Native, Chain::BurnOnTorch] {
        chain(engine, &prepared, &st, &params, &data);
        let runs: Vec<f64> = (0..3)
            .map(|_| {
                chain(engine, &prepared, &st, &params, &data)
                    .2
                    .as_secs_f64()
                    * 1e3
            })
            .collect();
        eprintln!("{path}, {n} examples, 2 stages, {engine:?}: backward passes {runs:.0?} ms");
    }
}

/// Measurement, run explicitly: ways of moving a checkpoint's worth of tensors (SmolLM2's shapes)
/// to the card and back, repeated in one process as an executor session repeats tasks.
#[test]
#[ignore = "measurement: host-device transfers"]
fn transfer_variants() {
    use tch::{Device, Kind, Tensor};
    let (g, _) = load(std::path::Path::new(
        &std::env::var("GG_STAGE_BENCH").expect("GG_STAGE_BENCH"),
    ));
    let host: Vec<Vec<f32>> = g
        .parameters
        .iter()
        .map(|p| vec![0.5f32; p.shape.iter().product::<u64>() as usize])
        .collect();
    let total: usize = host.iter().map(Vec::len).sum();
    let cuda = Device::Cuda(0);
    let sync = || tch::Cuda::synchronize(0);
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
    for round in 0..5 {
        // Up, one tensor at a time, pageable.
        let t = std::time::Instant::now();
        let per: Vec<Tensor> = host
            .iter()
            .map(|h| Tensor::from_slice(h).to_device(cuda))
            .collect();
        sync();
        let up_per = ms(t);
        // Up, one block, pageable.
        let t = std::time::Instant::now();
        let mut flat = Vec::with_capacity(total);
        host.iter().for_each(|h| flat.extend_from_slice(h));
        let one = Tensor::from_slice(&flat).to_device(cuda);
        sync();
        let up_flat = ms(t);
        // Up, one block, pinned.
        let t = std::time::Instant::now();
        let mut flat = Vec::with_capacity(total);
        host.iter().for_each(|h| flat.extend_from_slice(h));
        let pinned = Tensor::from_slice(&flat).pin_memory(cuda);
        let t_pin = ms(t);
        let one_p = pinned.to_device_(cuda, Kind::Float, true, false);
        sync();
        let up_pinned = ms(t);
        drop((pinned, flat));
        // Down, one tensor at a time.
        let t = std::time::Instant::now();
        let back: Vec<Vec<f32>> = per
            .iter()
            .map(|d| {
                let c = d.to_device(Device::Cpu);
                let n = c.numel();
                let mut v = vec![0f32; n];
                c.copy_data(&mut v, n);
                v
            })
            .collect();
        let down_per = ms(t);
        // Down, one block into a pinned buffer, then split.
        let t = std::time::Instant::now();
        let flat_dev = Tensor::cat(&per.iter().map(|d| d.reshape([-1])).collect::<Vec<_>>(), 0);
        let mut buf = Tensor::empty([total as i64], (Kind::Float, Device::Cpu)).pin_memory(cuda);
        buf.copy_(&flat_dev);
        let mut v = vec![0f32; total];
        buf.copy_data(&mut v, total);
        let down_pinned = ms(t);
        eprintln!(
            "round {round}: up per-tensor {up_per:.0} ms, one block {up_flat:.0}, one pinned block \
{up_pinned:.0} (of which pinning {t_pin:.0}); down per-tensor {down_per:.0}, one pinned block \
{down_pinned:.0}"
        );
        assert_eq!(back.len(), host.len());
        drop((one, one_p));
    }
}

/// Memory levers on the native engine (ADR-0038): segmented recomputation and parameter
/// offloading give the gradient the CPU gives with the same levers, under the validators' rule,
/// and hold less than the whole model at once.
#[test]
fn memory_levers_run_natively() {
    let dir = fixtures();
    let tol = gg_compute::catalog::graph_tolerance();
    for f in ["llama-hf-mini.json", "convnext-mini.json"] {
        let (g, prepared) = load(&dir.join(f));
        let mut params = exec::init_parameters(&g, 3);
        let mut d = Drbg::new("test/cuda/levers", 1);
        for spec in g.parameters.iter().filter(|s| s.trainable) {
            let t = params.get_mut(&spec.name).unwrap();
            if t.data.iter().all(|v| *v == 0.0) {
                for v in &mut t.data {
                    *v = 0.1 * d.next_sym_f32()
                        + if spec.name.ends_with("norm.weight") {
                            1.0
                        } else {
                            0.0
                        };
                }
            }
        }
        let target = g.loss.target().name.clone();
        let batch: BTreeMap<String, TensorF32> = g
            .inputs
            .iter()
            .chain(std::iter::once(g.loss.target()))
            .map(|s| {
                let shape: Vec<u64> = s
                    .shape
                    .iter()
                    .map(|x| if *x == 0 { 4 } else { *x })
                    .collect();
                let count: u64 = shape.iter().product();
                let data = (0..count)
                    .map(|_| {
                        if s.dtype == "i64" || s.name == target {
                            d.next_index(10) as f32
                        } else {
                            d.next_sym_f32()
                        }
                    })
                    .collect();
                (s.name.clone(), TensorF32::new(shape, data).unwrap())
            })
            .collect();
        let whole = exec::torch::gradient_cuda(
            0,
            &prepared,
            params.clone(),
            batch.clone(),
            0,
            Default::default(),
        )
        .unwrap();
        for (segments, offload) in [(3usize, false), (4, true), (1, true)] {
            let options = exec::ExecOptions {
                segments,
                offload_parameters: offload,
            };
            let cpu = exec::gradient_cpu_with(&prepared, &params, &batch, 0, options).unwrap();
            let gpu =
                exec::torch::gradient_cuda(0, &prepared, params.clone(), batch.clone(), 0, options)
                    .unwrap_or_else(|e| panic!("{f}, {segments} segments: {e}"));
            assert!(
                ((cpu.loss - gpu.loss) as f64).abs() <= 1e-4 * (cpu.loss as f64).abs().max(1.0),
                "{f}, {segments} segments: loss cpu {} cuda {}",
                cpu.loss,
                gpu.loss
            );
            assert_eq!(gpu.gradients.len(), cpu.gradients.len(), "{f}");
            for (name, c) in &cpu.gradients {
                let cmp = gg_compute::compare::compare_f32_norm(name, &gpu.gradients[name], c, tol);
                assert!(
                    cmp.passed,
                    "{f}, {segments} segments: {name}: {:?}",
                    cmp.checks
                );
            }
            if segments > 1 {
                assert!(
                    gpu.peak_parameter_bytes > 0
                        && gpu.peak_parameter_bytes
                            < g.parameters
                                .iter()
                                .map(|p| p.shape.iter().product::<u64>() * 4)
                                .sum::<u64>(),
                    "{f}: a segment holds only its own parameters"
                );
            }
            eprintln!(
                "{f}, {segments} segments, offload {offload}: peak activations {} (whole {}), \
parameters {}",
                gpu.peak_activation_bytes, whole.peak_activation_bytes, gpu.peak_parameter_bytes
            );
        }
    }
}
