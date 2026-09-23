//! GPU measurement: WGPU/Vulkan matmul against the CPU reference on explicitly selected hardware.
//! Runs only with `--features gpu`; fails (does not skip) if the device is unavailable.

#![cfg(feature = "gpu")]

use std::sync::OnceLock;

use gg_compute::compare::compare_f32;
use gg_compute::engine::EngineInfo;
use gg_compute::engine::gpu::{init_vulkan, list_vulkan_adapters};

/// One runtime initialization per process: cubecl refuses to initialize the same device twice.
fn device() -> &'static (burn::backend::wgpu::WgpuDevice, EngineInfo) {
    static DEVICE: OnceLock<(burn::backend::wgpu::WgpuDevice, EngineInfo)> = OnceLock::new();
    DEVICE.get_or_init(|| init_vulkan(0).expect("GPU requested but unavailable"))
}
use gg_compute::matmul::{matmul_naive_f64, matmul_on, matmul_reference};
use gg_protocol::drbg::Drbg;
use gg_protocol::messages::ToleranceV1;
use gg_protocol::tensor_file::TensorF32;

#[test]
fn vulkan_matmul_within_documented_tolerance() {
    let adapters = list_vulkan_adapters();
    eprintln!("vulkan adapters: {adapters:?}");
    let (device, info) = device().clone();
    eprintln!("using {}", info.device_name);
    assert!(!info.device_name.to_lowercase().contains("llvmpipe"));
    let tol = ToleranceV1 {
        abs: 1e-4,
        rel: 1e-4,
    };
    for (i, (m, k, n)) in [
        (64u32, 64u32, 64u32),
        (256, 512, 128),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    .iter()
    .enumerate()
    {
        let mut d = Drbg::new("gpu", i as u64);
        let a =
            TensorF32::new(vec![*m as u64, *k as u64], d.sym_f32_vec((m * k) as usize)).unwrap();
        let b =
            TensorF32::new(vec![*k as u64, *n as u64], d.sym_f32_vec((k * n) as usize)).unwrap();
        let cpu = matmul_reference(&a, &b).unwrap();
        let t0 = std::time::Instant::now();
        let gpu = matmul_on::<burn::backend::Vulkan>(&device, &a, &b).unwrap();
        let cold = t0.elapsed();
        let t1 = std::time::Instant::now();
        let gpu2 = matmul_on::<burn::backend::Vulkan>(&device, &a, &b).unwrap();
        let warm = t1.elapsed();
        assert_eq!(gpu.data, gpu2.data, "GPU result must be stable across runs");
        let cmp = compare_f32("c", &gpu, &cpu, tol);
        let max_abs = cmp
            .checks
            .iter()
            .find(|c| c.name == "c.max_abs_diff")
            .and_then(|c| c.measured)
            .unwrap();
        let max_rel = cmp
            .checks
            .iter()
            .find(|c| c.name == "c.max_rel_diff")
            .and_then(|c| c.measured)
            .unwrap();
        eprintln!(
            "[{m}x{k}x{n}] gpu-vs-cpu max_abs={max_abs:.3e} max_rel={max_rel:.3e} cold={cold:?} warm={warm:?}"
        );
        assert!(cmp.passed, "{:?}", cmp.checks);
        if *k <= 512 {
            let naive = matmul_naive_f64(&a, &b);
            let cmp2 = compare_f32("c", &gpu, &naive, tol);
            assert!(cmp2.passed, "gpu vs f64 oracle: {:?}", cmp2.checks);
        }
    }
}

/// Reproduces the demo inputs (seeds 1000..1004, 1024³) on the GPU, builds a decision from the
/// comparison and checks that its signature survives the JSON transport used by the validator.
#[test]
fn demo_decisions_survive_json_transport() {
    use gg_core::*;
    use gg_protocol::domain::Domain;
    use gg_protocol::keys::SecretKey;
    use gg_protocol::messages::*;
    use gg_protocol::signed::Signed;
    let (device, _info) = device().clone();
    let key = SecretKey::generate().unwrap();
    let net = NetworkId::new("gg-demo").unwrap();
    for seed in 1000u64..1004 {
        let mut d = Drbg::new("gg-cli/gen-matmul", seed);
        let a = TensorF32::new(vec![1024, 1024], d.sym_f32_vec(1024 * 1024)).unwrap();
        let b = TensorF32::new(vec![1024, 1024], d.sym_f32_vec(1024 * 1024)).unwrap();
        let cpu = matmul_reference(&a, &b).unwrap();
        let gpu = matmul_on::<burn::backend::Vulkan>(&device, &a, &b).unwrap();
        let cmp = compare_f32(
            "outputs.c",
            &gpu,
            &cpu,
            ToleranceV1 {
                abs: 1e-4,
                rel: 1e-4,
            },
        );
        eprintln!(
            "seed {seed}: {:?}",
            cmp.checks
                .iter()
                .map(|c| (c.name.clone(), c.measured))
                .collect::<Vec<_>>()
        );
        let decision = VerificationDecisionV1 {
            protocol_version: gg_protocol::PROTOCOL_VERSION,
            network: net.clone(),
            decision_id: DecisionId::new(),
            submission_hash: Sha256Digest::zero(),
            task_id: TaskId::new(),
            attempt_id: AttemptId::new(),
            validator_id: ValidatorId::new(),
            method: "recompute-reference".into(),
            method_version: 1,
            reference_profile: "cpu-f32-flex-v1".into(),
            checks: cmp.checks.clone(),
            artifacts_checked: vec![],
            outcome: if cmp.passed {
                VerificationOutcome::Accepted
            } else {
                VerificationOutcome::Rejected
            },
            decided_at: UnixMillis::now(),
        };
        let signed = Signed::sign(&key, &net, Domain::Verification, decision).unwrap();
        let text = serde_json::to_string(&serde_json::to_value(&signed).unwrap()).unwrap();
        let back: Signed<VerificationDecisionV1> = serde_json::from_str(&text).unwrap();
        if let Err(e) = back.verify(&net, Domain::Verification) {
            panic!(
                "seed {seed}: signature lost in JSON transport ({e}); checks = {:?}",
                cmp.checks
            );
        }
    }
}

/// Measures how far Vulkan training drifts from the CPU reference for the demo configuration
/// (used to decide `profiles::MLP_GPU_ADMITTED` and the default tolerance of `mlp.train.v1`).
#[test]
fn mlp_training_gpu_vs_cpu() {
    use gg_compute::mlp::{self, MlpTask, Optim};
    use gg_compute::synthetic::SyntheticMlpConfig;
    let (device, _info) = device().clone();
    let ds = SyntheticMlpConfig {
        d_input: 16,
        d_output: 4,
        d_teacher: 32,
        n_train: 4096,
        n_eval: 1024,
        seed: 7,
        noise: 0.05,
    }
    .generate()
    .unwrap();
    let data = mlp::Data {
        train_x: ds.train_x,
        train_y: ds.train_y,
        eval_x: ds.eval_x,
        eval_y: ds.eval_y,
    };
    let cpu_dev = burn::backend::flex::FlexDevice;
    let mut worst_w = 0.0f64;
    let mut worst_rel = 0.0f64;
    let mut worst_loss = 0.0f64;
    let mut all_within = true;
    // The demo grid (hidden × lr × seed) plus one SGD configuration.
    let mut configs: Vec<(Optim, f64, usize, u64)> = Vec::new();
    for d_h in [32usize, 64] {
        for lr in [0.02f64, 0.05] {
            for seed in 1u64..=3 {
                configs.push((Optim::Adam, lr, d_h, seed));
            }
        }
    }
    configs.push((Optim::Sgd, 0.05, 64, 3));
    for (opt, lr, d_h, seed) in configs {
        let task = MlpTask {
            d_in: 16,
            d_h,
            d_out: 4,
            optimizer: opt,
            lr,
            batch: 64,
            steps: 300,
            seed,
            n_train: 4096,
            n_eval: 1024,
        };
        let (wc, mc) = mlp::train::<burn::backend::Autodiff<burn::backend::Flex>>(
            &cpu_dev, &task, &data, None, None, "cpu",
        )
        .unwrap();
        let (wg, mg) = mlp::train::<burn::backend::Autodiff<burn::backend::Vulkan>>(
            &device, &task, &data, None, None, "gpu",
        )
        .unwrap();
        let tol = ToleranceV1 {
            abs: 1e-4,
            rel: 1e-3,
        };
        for ((name, c), (_, g)) in wc.named().iter().zip(wg.named().iter()) {
            let cmp = compare_f32(name, g, c, tol);
            let abs = cmp
                .checks
                .iter()
                .find(|k| k.name.ends_with("max_abs_diff"))
                .and_then(|k| k.measured)
                .unwrap_or(0.0);
            let rel = cmp
                .checks
                .iter()
                .find(|k| k.name.ends_with("max_rel_diff"))
                .and_then(|k| k.measured)
                .unwrap_or(0.0);
            worst_w = worst_w.max(abs);
            worst_rel = worst_rel.max(rel);
            all_within &= cmp.passed;
        }
        let dl = (mc.eval_loss as f64 - mg.eval_loss as f64).abs();
        worst_loss = worst_loss.max(dl);
        let mut cfg_abs = 0.0f64;
        let mut cfg_ok = true;
        for ((name, c), (_, g)) in wc.named().iter().zip(wg.named().iter()) {
            let cmp = compare_f32(name, g, c, tol);
            cfg_ok &= cmp.passed;
            cfg_abs = cfg_abs.max(
                cmp.checks
                    .iter()
                    .find(|k| k.name.ends_with("max_abs_diff"))
                    .and_then(|k| k.measured)
                    .unwrap_or(0.0),
            );
        }
        eprintln!(
            "[{opt:?} lr={lr} d_h={d_h} seed={seed}] eval cpu={} gpu={} |Δ|={dl:.3e} max_abs_weight_diff={cfg_abs:.3e} within_tol={cfg_ok}",
            mc.eval_loss, mg.eval_loss
        );
    }
    eprintln!(
        "mlp gpu-vs-cpu after 300 steps: max_abs_weight_diff={worst_w:.3e} max_rel_weight_diff={worst_rel:.3e} max_eval_loss_diff={worst_loss:.3e} all_within_default_tolerance={all_within}"
    );
    // Measurement only: admission is a documented decision, not an assertion here.
}

/// Gradient of one micro-batch on Vulkan vs the CPU reference (decides `GRAD_GPU_ADMITTED`).
#[test]
fn grad_gpu_vs_cpu() {
    use gg_compute::grad::{gradient, round_indices};
    use gg_compute::mlp::{MlpTask, Optim, initial_weights};
    use gg_compute::synthetic::SyntheticMlpConfig;
    let (device, _info) = device().clone();
    let ds = SyntheticMlpConfig {
        d_input: 16,
        d_output: 4,
        d_teacher: 32,
        n_train: 4096,
        n_eval: 8,
        seed: 7,
        noise: 0.05,
    }
    .generate()
    .unwrap();
    let cpu_dev = burn::backend::flex::FlexDevice;
    let tol = ToleranceV1 {
        abs: 1e-5,
        rel: 1e-4,
    };
    let mut worst = 0.0f64;
    let mut all_ok = true;
    for (seed, d_h) in [(1u64, 32usize), (2, 64), (3, 64), (4, 128)] {
        let task = MlpTask {
            d_in: 16,
            d_h,
            d_out: 4,
            optimizer: Optim::Sgd,
            lr: 0.0,
            batch: 1,
            steps: 1,
            seed,
            n_train: 4096,
            n_eval: 8,
        };
        let w = initial_weights(&task);
        let idx = round_indices(seed, 4096, 0, 256);
        let width = 16;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        for &i in &idx {
            xd.extend_from_slice(&ds.train_x.data[i * width..(i + 1) * width]);
            yd.extend_from_slice(&ds.train_y.data[i * 4..(i + 1) * 4]);
        }
        let x = TensorF32::new(vec![256, 16], xd).unwrap();
        let y = TensorF32::new(vec![256, 4], yd).unwrap();
        let (gc, lc) =
            gradient::<burn::backend::Autodiff<burn::backend::Flex>>(&cpu_dev, &w, &x, &y).unwrap();
        let (gg, lg) =
            gradient::<burn::backend::Autodiff<burn::backend::Vulkan>>(&device, &w, &x, &y)
                .unwrap();
        for ((name, c), g) in ["w1", "b1", "w2", "b2"]
            .iter()
            .zip([&gc.w1, &gc.b1, &gc.w2, &gc.b2])
            .zip([&gg.w1, &gg.b1, &gg.w2, &gg.b2])
        {
            let cmp = compare_f32(name, g, c, tol);
            let abs = cmp
                .checks
                .iter()
                .find(|k| k.name.ends_with("max_abs_diff"))
                .and_then(|k| k.measured)
                .unwrap_or(0.0);
            worst = worst.max(abs);
            all_ok &= cmp.passed;
        }
        eprintln!(
            "[grad seed={seed} d_h={d_h}] loss cpu={lc} gpu={lg} |Δ|={:.3e}",
            (lc as f64 - lg as f64).abs()
        );
    }
    eprintln!("grad gpu-vs-cpu: max_abs_diff={worst:.3e} all_within_default_tolerance={all_ok}");
}

// ---------------------------------------------------------------------------------------------
// Graphs (ADR-0016, increment 3): the executor on Vulkan against the Flex reference, and the
// throughput it reaches.
// ---------------------------------------------------------------------------------------------

mod graph_on_gpu {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex, Vulkan};
    use gg_compute::graph::exec::{self, Prepared};
    use gg_compute::graph::templates::{TransformerDims, transformer};
    use gg_compute::graph::{self, ModelGraphV1};
    use gg_protocol::drbg::Drbg;
    use gg_protocol::tensor_file::TensorF32;

    fn token_ids(n: u64, seq: u64, vocab: u64, seed: u64) -> TensorF32 {
        let mut d = Drbg::new("gpu/graph/ids", seed);
        TensorF32::new(
            vec![n, seq],
            (0..n * seq)
                .map(|_| d.next_index(vocab as u32) as f32)
                .collect(),
        )
        .unwrap()
    }

    fn max_abs(a: &TensorF32, b: &TensorF32) -> f32 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The verdict a CPU validator would give on this GPU gradient, at the published tolerance of
    /// `graph.grad.v1`: this is what decides whether the GPU profile is admitted for this type.
    fn report_tolerance(label: &str, cpu: &exec::StepResult, gpu: &exec::StepResult) {
        let tol =
            gg_compute::catalog::default_policy(&gg_protocol::messages::TaskParamsV1::GraphGrad(
                gg_protocol::messages::GraphGradParamsV1 {
                    graph_hash: gg_core::Sha256Digest::of_bytes(b"").to_hex(),
                    graph_bytes: 1,
                    parameters: 1,
                    checkpoint_bytes: 8,
                    gradient_bytes: 8,
                    segments: 1,
                    offload_parameters: false,
                    training_flops_per_example: 1,
                    activation_bytes_per_example: 1,
                    datasets: vec![],
                    round_seed: 0,
                    example_start: 0,
                    example_count: 1,
                    total_examples: 1,
                    shard_bytes: 0,
                    batch_inputs: false,
                    gradient_bf16: false,
                },
            ))
            .unwrap()
            .tolerance;
        let mut failed = Vec::new();
        let mut max_abs_all = 0.0f64;
        for (name, r) in &cpu.gradients {
            // The rule the validator applies to model tasks (`recompute-reference` v2, ADR-0031).
            let cmp = gg_compute::compare::compare_f32_norm(name, &gpu.gradients[name], r, tol);
            let get = |suffix: &str| {
                cmp.checks
                    .iter()
                    .find(|c| c.name.ends_with(suffix))
                    .and_then(|c| c.measured)
                    .unwrap_or(f64::NAN)
            };
            max_abs_all = max_abs_all.max(get("max_abs"));
            if !cmp.passed {
                let why: Vec<String> = cmp
                    .checks
                    .iter()
                    .filter(|c| !c.passed)
                    .map(|c| {
                        format!(
                            "{}: {} (measured {:?}, tolerance {:?})",
                            c.name, c.detail, c.measured, c.tolerance
                        )
                    })
                    .collect();
                failed.push(format!(
                    "{name} (|ref| max {:.3e}) {why:?}",
                    r.data.iter().fold(0.0f32, |m, v| m.max(v.abs()))
                ));
            }
        }
        eprintln!(
            "[{label}] maximum absolute GPU/CPU deviation over all gradients: {max_abs_all:.3e}"
        );
        eprintln!(
            "[{label}] CPU verdict on the GPU gradient at ‖Δ‖ ≤ {}·√n + {}·‖ref‖: {} tensors, {} rejected {:?}",
            tol.abs,
            tol.rel,
            cpu.gradients.len(),
            failed.len(),
            failed
        );
        assert!(
            failed.is_empty(),
            "[{label}] a CPU validator would not accept the GPU profile: {failed:?}"
        );
    }

    fn cpu_and_gpu(
        graph: &ModelGraphV1,
        params: &BTreeMap<String, TensorF32>,
        inputs: &BTreeMap<String, TensorF32>,
    ) -> (exec::StepResult, exec::StepResult, f64) {
        let prepared = Prepared::new(graph).expect("validated");
        let cpu = exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, params, inputs, 1)
            .expect("cpu");
        let (device, info) = super::device();
        eprintln!("gpu: {}", info.device_name);
        // Warm-up: kernel compilation, outside the timed section.
        let _ = exec::gradient::<Autodiff<Vulkan>>(device, &prepared, params, inputs, 1)
            .expect("gpu (warm-up)");
        let start = Instant::now();
        let gpu =
            exec::gradient::<Autodiff<Vulkan>>(device, &prepared, params, inputs, 1).expect("gpu");
        let secs = start.elapsed().as_secs_f64();
        // Is the GPU reproducible with itself? (backward passes that use atomic scatter are not
        // always)
        let again = exec::gradient::<Autodiff<Vulkan>>(device, &prepared, params, inputs, 1)
            .expect("gpu (repeat)");
        let unstable: Vec<&String> = gpu
            .gradients
            .iter()
            .filter(|(k, g)| g.encode() != again.gradients[*k].encode())
            .map(|(k, _)| k)
            .collect();
        eprintln!(
            "gpu twice: {} tensors not byte-identical {unstable:?}",
            unstable.len()
        );
        (cpu, gpu, secs)
    }

    #[test]
    fn transformer_block_gpu_vs_cpu() {
        let dims = TransformerDims {
            layers: 2,
            dim: 64,
            heads: 4,
            seq: 16,
            vocab: 100,
            ffn: 128,
            causal: true,
            positions: gg_compute::graph::templates::Positions::None,
        };
        let graph = transformer(dims).unwrap();
        let params = exec::init_parameters(&graph, 5);
        let inputs = BTreeMap::from([
            ("ids".to_string(), token_ids(8, 16, 100, 1)),
            ("y".to_string(), token_ids(8, 16, 100, 2)),
        ]);
        let (cpu, gpu, _) = cpu_and_gpu(&graph, &params, &inputs);
        let loss_gap = (cpu.loss - gpu.loss).abs();
        let mut worst = 0.0f32;
        let mut worst_name = String::new();
        for (name, g) in &cpu.gradients {
            let gap = max_abs(g, &gpu.gradients[name]);
            let scale = g.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let rel = if scale > 0.0 { gap / scale } else { gap };
            if rel > worst {
                worst = rel;
                worst_name = name.clone();
            }
        }
        eprintln!(
            "[transformer 2x64] loss cpu {} gpu {} (gap {loss_gap:.3e}); worst relative gradient deviation {worst:.3e} on {worst_name}",
            cpu.loss, gpu.loss
        );
        report_tolerance("transformer 2x64", &cpu, &gpu);
        assert!(loss_gap <= 1e-4, "loss: gap {loss_gap}");
        assert!(
            worst <= 1e-3,
            "gradient of {worst_name}: relative deviation {worst} above 1e-3"
        );
    }

    #[test]
    fn cnn_gpu_vs_cpu() {
        let graph: ModelGraphV1 = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/models/cnn-cifar.json"
            ))
            .expect("bundled example"),
        )
        .unwrap();
        let params = exec::init_parameters(&graph, 3);
        let mut d = Drbg::new("gpu/graph/cnn", 0);
        let n = 8u64;
        let x = TensorF32::new(
            vec![n, 3, 32, 32],
            (0..n * 3 * 32 * 32).map(|_| d.next_sym_f32()).collect(),
        )
        .unwrap();
        let y = TensorF32::new(vec![n], (0..n).map(|_| d.next_index(10) as f32).collect()).unwrap();
        let inputs = BTreeMap::from([("x".to_string(), x), ("y".to_string(), y)]);
        let (cpu, gpu, _) = cpu_and_gpu(&graph, &params, &inputs);
        let loss_gap = (cpu.loss - gpu.loss).abs();
        let mut worst = 0.0f32;
        for (name, g) in &cpu.gradients {
            let gap = max_abs(g, &gpu.gradients[name]);
            let scale = g.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            worst = worst.max(if scale > 0.0 { gap / scale } else { gap });
        }
        eprintln!(
            "[cnn-cifar] loss cpu {} gpu {} (gap {loss_gap:.3e}); worst relative deviation {worst:.3e}",
            cpu.loss, gpu.loss
        );
        report_tolerance("cnn-cifar", &cpu, &gpu);
        assert!(loss_gap <= 1e-4);
        assert!(worst <= 1e-3, "relative deviation {worst}");
    }

    /// Throughput actually reached by the graph executor on a stage of realistic size.
    /// This is the number that decides kernel priorities (ADR-0017).
    #[test]
    fn executor_throughput_on_a_realistic_stage() {
        let dims = TransformerDims {
            layers: 4,
            dim: 768,
            heads: 12,
            seq: 256,
            vocab: 8000,
            ffn: 3072,
            causal: true,
            positions: gg_compute::graph::templates::Positions::None,
        };
        let graph = transformer(dims).unwrap();
        let report = graph::validate(&graph).unwrap();
        let params = exec::init_parameters(&graph, 5);
        let batch = 8u64;
        let inputs = BTreeMap::from([
            ("ids".to_string(), token_ids(batch, 256, 8000, 1)),
            ("y".to_string(), token_ids(batch, 256, 8000, 2)),
        ]);
        let prepared = Prepared::new(&graph).unwrap();
        let flops = report.training_flops_per_example as f64 * batch as f64;

        let start = Instant::now();
        let _ =
            exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 1).unwrap();
        let cpu_s = start.elapsed().as_secs_f64();

        let (device, info) = super::device();
        let _ = exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 1).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = Instant::now();
            let _ =
                exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 1).unwrap();
            best = best.min(start.elapsed().as_secs_f64());
        }
        eprintln!(
            "[stage 4x768, batch {batch}] {:.1} GFLOP per micro-batch; cpu {cpu_s:.2} s ({:.1} GFLOP/s); gpu {} {best:.3} s ({:.1} GFLOP/s)",
            flops / 1e9,
            flops / cpu_s / 1e9,
            info.device_name,
            flops / best / 1e9
        );
        assert!(best.is_finite() && best > 0.0);
    }
}

// ---------------------------------------------------------------------------------------------
// GPU/CPU deviation operator by operator (ADR-0016: each operator carries its measured tolerance).
// ---------------------------------------------------------------------------------------------

mod graph_ops_on_gpu {
    use std::collections::BTreeMap;

    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex, Vulkan};
    use gg_compute::compare::compare_f32;
    use gg_compute::graph::ModelGraphV1;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_protocol::drbg::Drbg;
    use gg_protocol::messages::ToleranceV1;
    use gg_protocol::tensor_file::TensorF32;

    fn random(shape: &[u64], seed: u64) -> TensorF32 {
        let mut d = Drbg::new("gpu/ops", seed);
        let n: usize = shape.iter().product::<u64>() as usize;
        TensorF32::new(shape.to_vec(), (0..n).map(|_| d.next_sym_f32()).collect()).unwrap()
    }

    fn ids(n: u64, seq: u64, vocab: u64, seed: u64) -> TensorF32 {
        let mut d = Drbg::new("gpu/ops/ids", seed);
        TensorF32::new(
            vec![n, seq],
            (0..n * seq)
                .map(|_| d.next_index(vocab as u32) as f32)
                .collect(),
        )
        .unwrap()
    }

    /// One minimal graph per operator: the backward pass goes through the operator to reach at
    /// least one parameter upstream.
    fn cases() -> Vec<(&'static str, String, BTreeMap<String, TensorF32>)> {
        let dense_inputs = || {
            BTreeMap::from([
                ("x".to_string(), random(&[4, 8], 1)),
                ("y".to_string(), random(&[4, 4], 2)),
            ])
        };
        let unary = |op: &str, attrs: &str| {
            format!(
                r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[0,8]}}],
              "parameters":[{{"name":"w1","shape":[8,8],"init":"xavier_uniform"}},{{"name":"w2","shape":[8,4],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"a"}},{{"op":"{op}","inputs":["a"],"output":"b"{attrs}}},{{"op":"MatMul","inputs":["b","w2"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,4]}}}}}}}}"#
            )
        };
        let binary = |op: &str| {
            format!(
                r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[0,8]}}],
              "parameters":[{{"name":"w1","shape":[8,8],"init":"xavier_uniform"}},{{"name":"w2","shape":[8,8],"init":"xavier_uniform"}},{{"name":"one","shape":[1,8],"init":"ones"}},{{"name":"w3","shape":[8,4],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"a"}},{{"op":"MatMul","inputs":["x","w2"],"output":"b0"}},{{"op":"Sigmoid","inputs":["b0"],"output":"b1"}},{{"op":"Add","inputs":["b1","one"],"output":"b"}},{{"op":"{op}","inputs":["a","b"],"output":"c"}},{{"op":"MatMul","inputs":["c","w3"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,4]}}}}}}}}"#
            )
        };
        let norm = |op: &str, params: &str, inputs: &str| {
            format!(
                r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[0,8]}}],
              "parameters":[{{"name":"w1","shape":[8,8],"init":"xavier_uniform"}},{params},{{"name":"w2","shape":[8,4],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"a"}},{{"op":"{op}","inputs":[{inputs}],"output":"b"}},{{"op":"MatMul","inputs":["b","w2"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,4]}}}}}}}}"#
            )
        };
        let image_inputs = || {
            BTreeMap::from([
                ("x".to_string(), random(&[4, 2, 6, 6], 3)),
                ("y".to_string(), random(&[4, 4], 4)),
            ])
        };
        let image = |mid: &str, flat_in: &str, width: u64| {
            format!(
                r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[0,2,6,6]}}],
              "parameters":[{{"name":"k","shape":[3,2,3,3],"init":"xavier_uniform"}},{{"name":"kb","shape":[3],"init":"zeros"}},{{"name":"g","shape":[3],"init":"ones"}},{{"name":"bt","shape":[3],"init":"zeros"}},{{"name":"w","shape":[{width},4],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"Conv2d","inputs":["x","k","kb"],"output":"c","attrs":{{"padding":[1,1]}}}},{mid}{{"op":"Flatten","inputs":["{flat_in}"],"output":"f","attrs":{{"axis":1}}}},{{"op":"MatMul","inputs":["f","w"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,4]}}}}}}}}"#
            )
        };
        let gather = r#"{"format":"gg-graph/v1","name":"op","inputs":[{"name":"ids","shape":[0,4],"dtype":"i64"}],
              "parameters":[{"name":"emb","shape":[6,8],"init":"xavier_uniform"},{"name":"w","shape":[8,3],"init":"xavier_uniform"}],
              "nodes":[{"op":"Gather","inputs":["emb","ids"],"output":"h"},{"op":"MatMul","inputs":["h","w"],"output":"s"},{"op":"Reshape","inputs":["s"],"output":"pred","attrs":{"shape":[-1,3]}}],
              "loss":{"cross_entropy":{"prediction":"pred","target":{"name":"y","shape":[0,4],"dtype":"i64"}}}}"#.to_string();
        // gg-ops/7 (ADR-0027). A dense tensor [N, 8] reshaped to [N, 2, 4] for the reductions.
        let reduce = |op: &str, attrs: &str, width: u64| {
            format!(
                r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[0,8]}}],
              "parameters":[{{"name":"w1","shape":[8,8],"init":"xavier_uniform"}},{{"name":"w2","shape":[{width},4],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"a"}},{{"op":"Reshape","inputs":["a"],"output":"a3","attrs":{{"shape":[-1,2,4]}}}},
                       {{"op":"{op}","inputs":["a3"],"output":"r","attrs":{attrs}}},{{"op":"Flatten","inputs":["r"],"output":"f","attrs":{{"axis":1}}}},
                       {{"op":"MatMul","inputs":["f","w2"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,4]}}}}}}}}"#
            )
        };
        let dense_to = |nodes: &str, width: u64| {
            format!(
                r#"{{"format":"gg-graph/v1","name":"op","inputs":[{{"name":"x","shape":[0,8]}}],
              "parameters":[{{"name":"w1","shape":[8,8],"init":"xavier_uniform"}},{{"name":"w2","shape":[{width},4],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"a"}},{nodes},{{"op":"MatMul","inputs":["b","w2"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,4]}}}}}}}}"#
            )
        };
        let conv_t = r#"{"format":"gg-graph/v1","name":"op","inputs":[{"name":"x","shape":[0,2,6,6]}],
              "parameters":[{"name":"k","shape":[2,3,3,3],"init":"xavier_uniform"},{"name":"kb","shape":[3],"init":{"normal":{"std":0.1}}},{"name":"w","shape":[432,4],"init":"xavier_uniform"}],
              "nodes":[{"op":"ConvTranspose2d","inputs":["x","k","kb"],"output":"c","attrs":{"stride":[2,2],"padding":[1,1],"output_padding":[1,1]}},
                       {"op":"Flatten","inputs":["c"],"output":"f","attrs":{"axis":1}},{"op":"MatMul","inputs":["f","w"],"output":"pred"}],
              "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,4]}}}}"#.to_string();
        vec![
            ("MatMul", unary("Relu", ""), dense_inputs()),
            ("Relu", unary("Relu", ""), dense_inputs()),
            ("Gelu", unary("Gelu", ""), dense_inputs()),
            ("Gelu tanh", unary("Gelu", r#","attrs":{"approximate":"tanh"}"#), dense_inputs()),
            ("Sigmoid", unary("Sigmoid", ""), dense_inputs()),
            ("Tanh", unary("Tanh", ""), dense_inputs()),
            ("Softmax", unary("Softmax", r#","attrs":{"axis":1}"#), dense_inputs()),
            ("Dropout(0)", unary("Dropout", r#","attrs":{"rate":0.0}"#), dense_inputs()),
            ("Reshape", unary("Reshape", r#","attrs":{"shape":[-1,2,4]}"#).replace(r#"{"op":"MatMul","inputs":["b","w2"]"#, r#"{"op":"Flatten","inputs":["b"],"output":"b2","attrs":{"axis":1}},{"op":"MatMul","inputs":["b2","w2"]"#), dense_inputs()),
            ("Transpose", unary("Transpose", r#","attrs":{"perm":[1,0]}"#).replace(r#"{"op":"MatMul","inputs":["b","w2"]"#, r#"{"op":"Transpose","inputs":["b"],"output":"b2","attrs":{"perm":[1,0]}},{"op":"MatMul","inputs":["b2","w2"]"#), dense_inputs()),
            ("Flatten", unary("Flatten", r#","attrs":{"axis":1}"#), dense_inputs()),
            ("Concat", unary("Concat", r#","attrs":{"axis":1}"#).replace(r#""inputs":["a"],"output":"b""#, r#""inputs":["a","a"],"output":"b""#).replace(r#"{"name":"w2","shape":[8,4]"#, r#"{"name":"w2","shape":[16,4]"#), dense_inputs()),
            ("Add", binary("Add"), dense_inputs()),
            ("Sub", binary("Sub"), dense_inputs()),
            ("Mul", binary("Mul"), dense_inputs()),
            ("Div", binary("Div"), dense_inputs()),
            ("LayerNorm", norm("LayerNorm", r#"{"name":"g","shape":[8],"init":"ones"},{"name":"bt","shape":[8],"init":"zeros"}"#, r#""a","g","bt""#), dense_inputs()),
            ("RMSNorm", norm("RMSNorm", r#"{"name":"g","shape":[8],"init":"ones"}"#, r#""a","g""#), dense_inputs()),
            ("Conv2d", image("", "c", 108), image_inputs()),
            ("MaxPool2d", image(r#"{"op":"MaxPool2d","inputs":["c"],"output":"p","attrs":{"kernel":[2,2]}},"#, "p", 27), image_inputs()),
            ("AvgPool2d", image(r#"{"op":"AvgPool2d","inputs":["c"],"output":"p","attrs":{"kernel":[2,2]}},"#, "p", 27), image_inputs()),
            ("BatchNorm2d", image(r#"{"op":"BatchNorm2d","inputs":["c","g","bt"],"output":"p"},"#, "p", 108), image_inputs()),
            (
                "Slice",
                r#"{"format":"gg-graph/v1","name":"op","inputs":[{"name":"x","shape":[0,8]}],
              "parameters":[{"name":"w1","shape":[8,8],"init":"xavier_uniform"},{"name":"w2","shape":[3,4],"init":"xavier_uniform"}],
              "nodes":[{"op":"MatMul","inputs":["x","w1"],"output":"a"},{"op":"Slice","inputs":["a"],"output":"b","attrs":{"axis":1,"start":2,"end":5}},
                       {"op":"MatMul","inputs":["b","w2"],"output":"pred"}],
              "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,4]}}}}"#
                    .to_string(),
                dense_inputs(),
            ),
            (
                "CausalMask",
                r#"{"format":"gg-graph/v1","name":"op","inputs":[{"name":"x","shape":[0,4,4]}],
              "parameters":[{"name":"w1","shape":[4,4],"init":"xavier_uniform"},{"name":"w2","shape":[4,3],"init":"xavier_uniform"}],
              "nodes":[{"op":"MatMul","inputs":["x","w1"],"output":"s"},{"op":"CausalMask","inputs":["s"],"output":"m"},
                       {"op":"Softmax","inputs":["m"],"output":"a","attrs":{"axis":2}},{"op":"MatMul","inputs":["a","w2"],"output":"pred"}],
              "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,4,3]}}}}"#
                    .to_string(),
                BTreeMap::from([
                    ("x".to_string(), random(&[4, 4, 4], 21)),
                    ("y".to_string(), random(&[4, 4, 3], 22)),
                ]),
            ),
            ("Gather", gather, BTreeMap::from([("ids".to_string(), ids(4, 4, 6, 7)), ("y".to_string(), ids(4, 4, 3, 8))])),
            (
                "ReduceMean",
                r#"{"format":"gg-graph/v1","name":"op","inputs":[{"name":"x","shape":[0,8]}],
              "parameters":[{"name":"w1","shape":[8,8],"init":"xavier_uniform"},{"name":"w2","shape":[8,4],"init":"xavier_uniform"}],
              "nodes":[{"op":"MatMul","inputs":["x","w1"],"output":"a"},{"op":"ReduceMean","inputs":["a"],"output":"m","attrs":{"axis":1}},
                       {"op":"Sub","inputs":["a","m"],"output":"b"},{"op":"MatMul","inputs":["b","w2"],"output":"pred"}],
              "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,4]}}}}"#
                    .to_string(),
                dense_inputs(),
            ),
            ("ConvTranspose2d", conv_t, image_inputs()),
            ("Pad reflect", image(r#"{"op":"Pad","inputs":["c"],"output":"p","attrs":{"pads":[0,0,1,2,0,0,2,1],"mode":"reflect"}},"#, "p", 243), image_inputs()),
            ("Pad constant", dense_to(r#"{"op":"Pad","inputs":["a"],"output":"b","attrs":{"pads":[0,2,0,3],"value":0.5}}"#, 13), dense_inputs()),
            ("Pad edge", dense_to(r#"{"op":"Pad","inputs":["a"],"output":"b","attrs":{"pads":[0,3,0,1],"mode":"edge"}}"#, 12), dense_inputs()),
            ("Resize nearest", image(r#"{"op":"Resize","inputs":["c"],"output":"p","attrs":{"size":[9,12]}},"#, "p", 324), image_inputs()),
            ("Resize linear", image(r#"{"op":"Resize","inputs":["c"],"output":"p","attrs":{"size":[10,7],"mode":"linear"}},"#, "p", 210), image_inputs()),
            ("Resize linear aligned", image(r#"{"op":"Resize","inputs":["c"],"output":"p","attrs":{"size":[11,4],"mode":"linear","align_corners":1}},"#, "p", 132), image_inputs()),
            ("Flip", dense_to(r#"{"op":"Flip","inputs":["a"],"output":"b","attrs":{"axes":[1]}}"#, 8), dense_inputs()),
            ("ReduceSum", reduce("ReduceSum", r#"{"axis":2,"keepdims":0}"#, 2), dense_inputs()),
            ("ReduceMax", reduce("ReduceMax", r#"{"axis":2}"#, 2), dense_inputs()),
            ("ReduceMin", reduce("ReduceMin", r#"{"axis":1,"keepdims":0}"#, 4), dense_inputs()),
            ("ReduceProd", reduce("ReduceProd", r#"{"axis":2,"keepdims":0}"#, 2), dense_inputs()),
            ("CumSum", dense_to(r#"{"op":"CumSum","inputs":["a"],"output":"b","attrs":{"axis":1}}"#, 8), dense_inputs()),
            ("IndexSelect", dense_to(r#"{"op":"IndexSelect","inputs":["a"],"output":"b","attrs":{"axis":1,"index":[7,0,3,3,5,1,2,6,6]}}"#, 9), dense_inputs()),
            ("IndexAdd", dense_to(r#"{"op":"IndexSelect","inputs":["a"],"output":"m","attrs":{"axis":1,"index":[0,2,4,6,1]}},{"op":"IndexAdd","inputs":["a","m"],"output":"b","attrs":{"axis":1,"index":[1,1,2,7,0]}}"#, 8), dense_inputs()),
            // gg-ops/8 (ADR-0029): integer tensors computed in the graph.
            ("ArgTopK+GatherElements", dense_to(r#"{"op":"ArgTopK","inputs":["a"],"output":"i","attrs":{"axis":1,"k":3}},{"op":"GatherElements","inputs":["a","i"],"output":"b","attrs":{"axis":1}}"#, 3), dense_inputs()),
            ("ArgSort+GatherElements", dense_to(r#"{"op":"ArgSort","inputs":["a"],"output":"i","attrs":{"axis":1,"descending":1}},{"op":"GatherElements","inputs":["a","i"],"output":"b","attrs":{"axis":1}}"#, 8), dense_inputs()),
            ("ArgMax+GatherElements", dense_to(r#"{"op":"ArgMax","inputs":["a"],"output":"i","attrs":{"axis":1}},{"op":"GatherElements","inputs":["a","i"],"output":"m","attrs":{"axis":1}},{"op":"Mul","inputs":["a","m"],"output":"b"}"#, 8), dense_inputs()),
            ("ArgMin+Cast", dense_to(r#"{"op":"ArgMin","inputs":["a"],"output":"i","attrs":{"axis":1}},{"op":"Cast","inputs":["i"],"output":"c","attrs":{"to":"f32"}},{"op":"Add","inputs":["a","c"],"output":"b"}"#, 8), dense_inputs()),
            ("ScatterElements add", dense_to(r#"{"op":"ArgTopK","inputs":["a"],"output":"i","attrs":{"axis":1,"k":2}},{"op":"GatherElements","inputs":["a","i"],"output":"t","attrs":{"axis":1}},{"op":"ScatterElements","inputs":["a","i","t"],"output":"b","attrs":{"axis":1}}"#, 8), dense_inputs()),
            ("ScatterElements none", dense_to(r#"{"op":"ArgTopK","inputs":["a"],"output":"i","attrs":{"axis":1,"k":2}},{"op":"GatherElements","inputs":["a","i"],"output":"t","attrs":{"axis":1}},{"op":"Softmax","inputs":["t"],"output":"w","attrs":{"axis":1}},{"op":"ScatterElements","inputs":["a","i","w"],"output":"b","attrs":{"axis":1,"reduction":"none"}}"#, 8), dense_inputs()),
            (
                "Gather rank-3 routed",
                r#"{"format":"gg-graph/v1","name":"op","inputs":[{"name":"x","shape":[0,8]}],
              "parameters":[{"name":"r","shape":[8,3],"init":"xavier_uniform"},{"name":"we","shape":[3,4,8],"init":"xavier_uniform"}],
              "nodes":[{"op":"MatMul","inputs":["x","r"],"output":"logits"},{"op":"ArgTopK","inputs":["logits"],"output":"i","attrs":{"axis":1,"k":1}},
                       {"op":"Reshape","inputs":["i"],"output":"e","attrs":{"shape":[-1]}},{"op":"Gather","inputs":["we","e"],"output":"w"},
                       {"op":"Reshape","inputs":["x"],"output":"x3","attrs":{"shape":[-1,8,1]}},{"op":"MatMul","inputs":["w","x3"],"output":"o"},
                       {"op":"Reshape","inputs":["o"],"output":"pred","attrs":{"shape":[-1,4]}}],
              "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,4]}}}}"#
                    .to_string(),
                dense_inputs(),
            ),
            (
                "Rsqrt",
                r#"{"format":"gg-graph/v1","name":"op","inputs":[{"name":"x","shape":[0,8]}],
              "parameters":[{"name":"w1","shape":[8,8],"init":"xavier_uniform"},{"name":"one","shape":[1],"init":"ones"},{"name":"w2","shape":[8,4],"init":"xavier_uniform"}],
              "nodes":[{"op":"MatMul","inputs":["x","w1"],"output":"a"},{"op":"Mul","inputs":["a","a"],"output":"a2"},{"op":"Add","inputs":["a2","one"],"output":"p"},
                       {"op":"Rsqrt","inputs":["p"],"output":"b"},{"op":"MatMul","inputs":["b","w2"],"output":"pred"}],
              "loss":{"mean_squared_error":{"prediction":"pred","target":{"name":"y","shape":[0,4]}}}}"#
                    .to_string(),
                dense_inputs(),
            ),
        ]
    }

    #[test]
    fn deviation_per_operator() {
        let (device, info) = super::device();
        eprintln!("gpu: {}", info.device_name);
        let tol = ToleranceV1 {
            abs: 1e-5,
            rel: 1e-3,
        };
        let mut lines = Vec::new();
        let mut rejected = Vec::new();
        for (label, json, inputs) in cases() {
            let graph: ModelGraphV1 =
                serde_json::from_str(&json).unwrap_or_else(|e| panic!("{label}: {e}\n{json}"));
            let prepared = Prepared::new(&graph).unwrap_or_else(|e| panic!("{label}: {e}"));
            let params = exec::init_parameters(&graph, 11);
            let cpu = exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 1)
                .unwrap();
            let gpu =
                exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 1).unwrap();
            let mut max_abs = 0.0f64;
            let mut max_rel = 0.0f64;
            let mut ok = true;
            for (name, r) in &cpu.gradients {
                let g = &gpu.gradients[name];
                for (a, b) in g.data.iter().zip(&r.data) {
                    let d = (*a as f64 - *b as f64).abs();
                    max_abs = max_abs.max(d);
                    if b.abs() > 1e-6 {
                        max_rel = max_rel.max(d / (*b as f64).abs());
                    }
                }
                ok &= compare_f32(name, g, r, tol).passed;
            }
            if !ok {
                rejected.push(label);
            }
            lines.push(format!(
                "| {label:<11} | {max_abs:.2e} | {max_rel:.2e} | {} |",
                if ok { "yes" } else { "**no**" }
            ));
        }
        eprintln!(
            "| Operator | max abs deviation | max rel deviation | admitted at abs 1e-5 + rel 1e-3 |"
        );
        eprintln!("|---|---|---|---|");
        for l in &lines {
            eprintln!("{l}");
        }
        eprintln!("rejected: {rejected:?}");
        assert!(
            rejected.is_empty(),
            "operators outside tolerance on GPU: {rejected:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Causal mask on the GPU (ADR-0020): no information flows back from the future, bit for bit, on GPU.
// ---------------------------------------------------------------------------------------------

mod causal_on_gpu {
    use std::collections::BTreeMap;

    use burn::backend::Vulkan;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_compute::graph::templates::{TransformerDims, transformer};
    use gg_protocol::drbg::Drbg;
    use gg_protocol::tensor_file::TensorF32;

    #[test]
    fn no_future_leak_on_gpu() {
        let (seq, vocab) = (8usize, 13usize);
        let graph = transformer(TransformerDims {
            layers: 2,
            dim: 32,
            heads: 4,
            seq: seq as u64,
            vocab: vocab as u64,
            ffn: 64,
            causal: true,
            positions: gg_compute::graph::templates::Positions::None,
        })
        .unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        let params = exec::init_parameters(&graph, 5);
        let mut d = Drbg::new("gpu/causal/ids", 1);
        let ids = TensorF32::new(
            vec![2, seq as u64],
            (0..2 * seq)
                .map(|_| d.next_index(vocab as u32) as f32)
                .collect(),
        )
        .unwrap();
        let (device, info) = super::device();
        eprintln!("gpu: {}", info.device_name);
        let run = |ids: &TensorF32| {
            exec::forward::<Vulkan>(
                device,
                &prepared,
                &params,
                &BTreeMap::from([("ids".to_string(), ids.clone())]),
                0,
                "seqlogits",
            )
            .unwrap()
        };
        let base = run(&ids);
        for j in 1..seq {
            let mut changed = ids.clone();
            changed.data[j] = ((changed.data[j] as usize + 1) % vocab) as f32;
            let after = run(&changed);
            let prefix = |t: &TensorF32| {
                t.data[..j * vocab]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                prefix(&base),
                prefix(&after),
                "GPU: token {j} changes an earlier position"
            );
        }
        // And the GPU computes the same logits as the CPU reference, within the type's tolerance.
        let cpu = exec::forward_cpu(
            &prepared,
            &params,
            &BTreeMap::from([("ids".to_string(), ids.clone())]),
            0,
            "seqlogits",
        )
        .unwrap();
        let max = cpu
            .data
            .iter()
            .zip(&base.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[causal transformer] maximum GPU/CPU logit deviation: {max:.3e}");
        assert!(max <= 1e-4, "GPU/CPU logits: deviation {max}");
    }
}

mod rotary_on_gpu {
    use std::collections::BTreeMap;

    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex, Vulkan};
    use gg_compute::compare::compare_f32;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_compute::graph::rope::RopeScaling;
    use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
    use gg_protocol::drbg::Drbg;
    use gg_protocol::messages::ToleranceV1;
    use gg_protocol::tensor_file::TensorF32;

    #[test]
    fn rotary_transformer_gpu_vs_cpu() {
        // YaRN: explicit frequencies and an amplitude other than 1 (ADR-0022).
        let yarn = Positions::Rotary {
            base: 10000.0,
            scaling: RopeScaling::Yarn {
                factor: 8.0,
                original_max_positions: 4,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None,
                mscale: None,
                mscale_all_dim: None,
                truncate: true,
            },
        };
        for positions in [Positions::rotary(10000.0), Positions::Learned, yarn] {
            let label = format!("{positions:?}");
            let graph = transformer(TransformerDims {
                layers: 2,
                dim: 64,
                heads: 4,
                seq: 16,
                vocab: 100,
                ffn: 128,
                causal: true,
                positions,
            })
            .unwrap();
            let prepared = Prepared::new(&graph).unwrap();
            let params = exec::init_parameters(&graph, 5);
            let mut d = Drbg::new("gpu/rotary", 1);
            let ids = |d: &mut Drbg| {
                TensorF32::new(
                    vec![8, 16],
                    (0..128).map(|_| d.next_index(100) as f32).collect(),
                )
                .unwrap()
            };
            let inputs = BTreeMap::from([
                ("ids".to_string(), ids(&mut d)),
                ("y".to_string(), ids(&mut d)),
            ]);
            let cpu = exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 1)
                .unwrap();
            let (device, _) = super::device();
            let gpu =
                exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 1).unwrap();
            let tol = ToleranceV1 {
                abs: 1e-4,
                rel: 1e-3,
            };
            let mut worst = 0.0f32;
            let mut refused = Vec::new();
            for (name, r) in &cpu.gradients {
                let g = &gpu.gradients[name];
                worst = worst.max(
                    g.data
                        .iter()
                        .zip(&r.data)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max),
                );
                if !compare_f32(name, g, r, tol).passed {
                    refused.push(name.clone());
                }
            }
            eprintln!(
                "[transformer {label}] loss cpu {} gpu {}; maximum absolute gradient deviation {worst:.3e}; rejected {refused:?}",
                cpu.loss, gpu.loss
            );
            assert!((cpu.loss - gpu.loss).abs() <= 1e-4);
            assert!(
                refused.is_empty(),
                "a CPU validator would reject: {refused:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// A real Llama from `transformers`, captured by the driver (GQA, SwiGLU, hand-written RMSNorm,
// Llama 3.1 RoPE), trains on the GPU with gradients that a CPU validator accepts.
// ---------------------------------------------------------------------------------------------
mod hf_llama_on_gpu {
    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex, Vulkan};
    use gg_compute::compare::compare_f32;
    use gg_compute::graph::ModelGraphV1;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_protocol::drbg::Drbg;
    use gg_protocol::messages::ToleranceV1;
    use gg_protocol::tensor_file::TensorF32;
    use std::collections::BTreeMap;

    #[test]
    fn hf_llama_gpu_vs_cpu() {
        let text = include_str!("fixtures/llama-hf-mini.json");
        let graph: ModelGraphV1 = serde_json::from_str(text).unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        // The driver declares the module's weights as zeros (they travel in the checkpoint); here,
        // random weights for the trained parameters, and the frozen tables and constants as their
        // generator produces them.
        let mut params = exec::init_parameters(&graph, 3);
        let mut d = Drbg::new("gpu/hf-llama", 1);
        for p in graph.parameters.iter().filter(|p| p.trainable) {
            let t = params.get_mut(&p.name).unwrap();
            for v in &mut t.data {
                *v = 0.1 * d.next_sym_f32()
                    + if p.name.ends_with("norm.weight") {
                        1.0
                    } else {
                        0.0
                    };
            }
        }
        let ids = |d: &mut Drbg| {
            TensorF32::new(
                vec![4, 8],
                (0..32).map(|_| d.next_index(64) as f32).collect(),
            )
            .unwrap()
        };
        let input = graph.inputs[0].name.clone();
        let inputs = BTreeMap::from([(input, ids(&mut d)), ("y".to_string(), ids(&mut d))]);
        let cpu =
            exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 1).unwrap();
        let (device, _) = super::device();
        let gpu =
            exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 1).unwrap();
        let tol = ToleranceV1 {
            abs: 1e-4,
            rel: 1e-3,
        };
        let mut worst = 0f32;
        let mut refused = Vec::new();
        for (name, r) in &cpu.gradients {
            let g = &gpu.gradients[name];
            worst = g
                .data
                .iter()
                .zip(&r.data)
                .map(|(a, b)| (a - b).abs())
                .fold(worst, f32::max);
            if !compare_f32(name, g, r, tol).passed {
                refused.push(name.clone());
            }
        }
        eprintln!(
            "[HF Llama] {} nodes, loss cpu {} gpu {}; maximum absolute gradient deviation {worst:.3e}; rejected {refused:?}",
            graph.nodes.len(),
            cpu.loss,
            gpu.loss
        );
        assert!((cpu.loss - gpu.loss).abs() <= 1e-4);
        assert!(
            refused.is_empty(),
            "a CPU validator would reject: {refused:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Stages on the GPU (D2): the stage chain of a captured Llama, computed on GPU, yields gradients
// that a CPU validator accepts, stage by stage.
// ---------------------------------------------------------------------------------------------
mod stages_on_gpu {
    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex, Vulkan};
    use gg_compute::compare::compare_f32;
    use gg_compute::graph::ModelGraphV1;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_compute::graph::stages;
    use gg_protocol::drbg::Drbg;
    use gg_protocol::messages::ToleranceV1;
    use gg_protocol::tensor_file::TensorF32;
    use std::collections::BTreeMap;

    #[test]
    fn stage_chain_gpu_vs_cpu() {
        let graph: ModelGraphV1 =
            serde_json::from_str(include_str!("fixtures/llama-hf-mini.json")).unwrap();
        let report = gg_compute::graph::validate(&graph).unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        let mut params = exec::init_parameters(&graph, 3);
        let mut d = Drbg::new("gpu/stages", 1);
        for p in graph.parameters.iter().filter(|p| p.trainable) {
            for v in &mut params.get_mut(&p.name).unwrap().data {
                *v = 0.1 * d.next_sym_f32()
                    + if p.name.ends_with("norm.weight") {
                        1.0
                    } else {
                        0.0
                    };
            }
        }
        let ids = |d: &mut Drbg| {
            TensorF32::new(
                vec![4, 8],
                (0..32).map(|_| d.next_index(64) as f32).collect(),
            )
            .unwrap()
        };
        let data = BTreeMap::from([
            (graph.inputs[0].name.clone(), ids(&mut d)),
            ("y".to_string(), ids(&mut d)),
        ]);
        let st =
            stages::stages(&graph, &report, &stages::even_bounds(report.order.len(), 3)).unwrap();
        let (device, _) = super::device();
        let tol = ToleranceV1 {
            abs: 1e-4,
            rel: 1e-3,
        };
        // Each stage is computed on both engines from the SAME inputs (the CPU's), just as a CPU
        // validator recomputes a stage produced on GPU.
        let mut produced = data.clone();
        let mut worst = 0f32;
        let mut refused = Vec::new();
        for s in &st {
            let ins: BTreeMap<_, _> = s
                .inputs
                .iter()
                .map(|n| (n.clone(), produced[n].clone()))
                .collect();
            let (c, _) =
                exec::stage_forward::<Flex>(&FlexDevice, &prepared, s, &params, &ins, 0).unwrap();
            let (g, _) =
                exec::stage_forward::<Vulkan>(device, &prepared, s, &params, &ins, 0).unwrap();
            for (n, t) in &c {
                if !compare_f32(n, &g[n], t, tol).passed {
                    refused.push(format!("forward {} {n}", s.index));
                }
                worst = worst.max(
                    g[n].data
                        .iter()
                        .zip(&t.data)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max),
                );
            }
            produced.extend(c);
        }
        let mut received: BTreeMap<String, Vec<TensorF32>> = BTreeMap::new();
        for s in st.iter().rev() {
            let ins: BTreeMap<_, _> = s
                .inputs
                .iter()
                .map(|n| (n.clone(), produced[n].clone()))
                .collect();
            let c = exec::stage_backward::<Autodiff<Flex>>(
                &FlexDevice,
                &prepared,
                s,
                &params,
                &ins,
                &received,
                0,
            )
            .unwrap();
            let g = exec::stage_backward::<Autodiff<Vulkan>>(
                device, &prepared, s, &params, &ins, &received, 0,
            )
            .unwrap();
            for (n, t) in c.gradients.iter().chain(c.input_gradients.iter()) {
                let other = g.gradients.get(n).or(g.input_gradients.get(n)).unwrap();
                if !compare_f32(n, other, t, tol).passed {
                    refused.push(format!("backward {} {n}", s.index));
                }
                worst = worst.max(
                    other
                        .data
                        .iter()
                        .zip(&t.data)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max),
                );
            }
            for (n, t) in c.input_gradients {
                received.entry(n).or_default().push(t);
            }
        }
        eprintln!(
            "[HF Llama stages] {} stages, maximum GPU/CPU deviation {worst:.3e}, rejected {refused:?}",
            st.len()
        );
        assert!(
            refused.is_empty(),
            "a CPU validator would reject: {refused:?}"
        );
    }
}

/// `Formula` (ADR-0025) on the GPU: every function of the grammar, in value and in gradient,
/// against the CPU, at the tolerance a CPU validator applies.
mod formula_on_gpu {
    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex};
    use gg_compute::compare::compare_f32;
    use gg_compute::graph::ModelGraphV1;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_protocol::drbg::Drbg;
    use gg_protocol::messages::ToleranceV1;
    use gg_protocol::tensor_file::TensorF32;
    use std::collections::BTreeMap;

    type Vulkan = burn::backend::Vulkan;

    #[test]
    fn every_formula_function_matches_the_cpu() {
        // Two layers; between them, a formula that uses the whole grammar. `log`, `sqrt` and the
        // fractional power receive a positive argument (softplus + 0.1).
        let expr = "tanh(a) * sigmoid(b) + softplus(a - b) * 0.5 - erf(a / 2) \
                    + sin(b) * cos(a) + log(softplus(a) + 0.1) - sqrt(softplus(b) + 0.1) \
                    + rsqrt(softplus(a) + 1) + log1p(abs(b)) + max(a, b) - min(a, 0.25) \
                    + relu(b)^3 / 8 + (softplus(a) + 0.1)^0.5 + exp(-abs(a)) - b^-2 * 0 + a";
        let json = format!(
            r#"{{"format":"gg-graph/v1","name":"f","inputs":[{{"name":"x","shape":[0,32]}}],
              "parameters":[{{"name":"w1","shape":[32,64],"init":"xavier_uniform"}},
                            {{"name":"w2","shape":[32,64],"init":"xavier_uniform"}},
                            {{"name":"w3","shape":[64,8],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"h1"}},
                       {{"op":"MatMul","inputs":["x","w2"],"output":"h2"}},
                       {{"op":"Formula","inputs":["h1","h2"],"output":"f","attrs":{{"expr":{}}}}},
                       {{"op":"MatMul","inputs":["f","w3"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,8]}}}}}}}}"#,
            serde_json::to_string(expr).unwrap()
        );
        let graph: ModelGraphV1 = serde_json::from_str(&json).unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        let params = exec::init_parameters(&graph, 11);
        let mut d = Drbg::new("test/gpu/formula", 1);
        let mut t = |shape: Vec<u64>| {
            let n = shape.iter().product::<u64>() as usize;
            TensorF32::new(shape, (0..n).map(|_| d.next_sym_f32() * 2.0).collect()).unwrap()
        };
        let inputs = BTreeMap::from([
            ("x".to_string(), t(vec![16, 32])),
            ("y".to_string(), t(vec![16, 8])),
        ]);
        let (device, _) = super::device();
        let cpu =
            exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 0).unwrap();
        let gpu =
            exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 0).unwrap();
        let tol = ToleranceV1 {
            abs: 1e-4,
            rel: 1e-3,
        };
        let mut worst = 0f32;
        for (n, c) in &cpu.gradients {
            let g = &gpu.gradients[n];
            assert!(g.data.iter().all(|v| v.is_finite()), "{n}: not finite");
            assert!(
                compare_f32(n, g, c, tol).passed,
                "{n}: a CPU validator would reject it"
            );
            worst = worst.max(
                g.data
                    .iter()
                    .zip(&c.data)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
            );
        }
        eprintln!(
            "[Formula] loss CPU {} GPU {}, maximum gradient deviation {worst:.3e}",
            cpu.loss, gpu.loss
        );
        assert!((cpu.loss - gpu.loss).abs() <= 1e-4 * cpu.loss.abs().max(1.0));
    }

    /// The functions `gg-ops/7` adds to the grammar (ADR-0027), each with an argument inside its
    /// domain, and the piecewise-constant ones, whose value must be identical on the GPU (the
    /// rounding of ties included: SPIR-V leaves it to the implementation, the executor does not).
    #[test]
    fn functions_of_gg_ops_7_match_the_cpu() {
        let expr = "tan(a * 0.5) * b + asin(tanh(a) * 0.9) - acos(tanh(b) * 0.9) + atan(a * b) \
                    + sinh(b) * 0.1 + cosh(a * 0.5) + asinh(a * 2) + acosh(b * b + 1.5) \
                    + atanh(tanh(a) * 0.8) + atan2(a, b) + atan2(1, b) * a \
                    + (floor(a * 3) + ceil(b * 3) + round(a * 4) + trunc(b * 5) + sign(a)) * b";
        let json = format!(
            r#"{{"format":"gg-graph/v1","name":"f","inputs":[{{"name":"x","shape":[0,32]}}],
              "parameters":[{{"name":"w1","shape":[32,64],"init":"xavier_uniform"}},
                            {{"name":"w2","shape":[32,64],"init":"xavier_uniform"}},
                            {{"name":"w3","shape":[64,8],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w1"],"output":"h1"}},
                       {{"op":"MatMul","inputs":["x","w2"],"output":"h2"}},
                       {{"op":"Formula","inputs":["h1","h2"],"output":"f","attrs":{{"expr":{}}}}},
                       {{"op":"MatMul","inputs":["f","w3"],"output":"pred"}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,8]}}}}}}}}"#,
            serde_json::to_string(expr).unwrap()
        );
        let graph: ModelGraphV1 = serde_json::from_str(&json).unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        let params = exec::init_parameters(&graph, 11);
        let mut d = Drbg::new("test/gpu/formula7", 1);
        let mut t = |shape: Vec<u64>| {
            let n = shape.iter().product::<u64>() as usize;
            TensorF32::new(shape, (0..n).map(|_| d.next_sym_f32() * 2.0).collect()).unwrap()
        };
        let inputs = BTreeMap::from([
            ("x".to_string(), t(vec![16, 32])),
            ("y".to_string(), t(vec![16, 8])),
        ]);
        let (device, _) = super::device();
        let cpu =
            exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 0).unwrap();
        let gpu =
            exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 0).unwrap();
        let tol = ToleranceV1 {
            abs: 1e-4,
            rel: 1e-3,
        };
        let mut worst = 0f32;
        for (n, c) in &cpu.gradients {
            let g = &gpu.gradients[n];
            assert!(g.data.iter().all(|v| v.is_finite()), "{n}: not finite");
            assert!(
                compare_f32(n, g, c, tol).passed,
                "{n}: a CPU validator would reject it"
            );
            worst = worst.max(
                g.data
                    .iter()
                    .zip(&c.data)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
            );
        }
        eprintln!(
            "[Formula gg-ops/7] loss CPU {} GPU {}, maximum gradient deviation {worst:.3e}",
            cpu.loss, gpu.loss
        );
        assert!((cpu.loss - gpu.loss).abs() <= 1e-4 * cpu.loss.abs().max(1.0));

        // Ties of `round` and exact values of the piecewise-constant functions: the same bytes.
        let values: Vec<f32> = vec![
            -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -3.7, 0.0, 2.2, -0.2, 7.0,
        ];
        let n = values.len() as u64;
        for expr in [
            "round(a + b)",
            "floor(a + b)",
            "ceil(a + b)",
            "trunc(a + b)",
            "sign(a + b)",
        ] {
            let json = format!(
                r#"{{"format":"gg-graph/v1","name":"r","inputs":[{{"name":"x","shape":[0,{n}]}}],
                  "parameters":[{{"name":"w","shape":[{n}],"init":"zeros"}}],
                  "nodes":[{{"op":"Formula","inputs":["x","w"],"output":"pred","attrs":{{"expr":"{expr}"}}}}],
                  "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,{n}]}}}}}}}}"#
            );
            let graph: ModelGraphV1 = serde_json::from_str(&json).unwrap();
            let prepared = Prepared::new(&graph).unwrap();
            let params = exec::init_parameters(&graph, 0);
            let inputs = BTreeMap::from([(
                "x".to_string(),
                TensorF32::new(vec![1, n], values.clone()).unwrap(),
            )]);
            let c =
                exec::forward::<Flex>(&FlexDevice, &prepared, &params, &inputs, 0, "pred").unwrap();
            let g =
                exec::forward::<Vulkan>(device, &prepared, &params, &inputs, 0, "pred").unwrap();
            assert_eq!(c.data, g.data, "{expr}: CPU and GPU differ");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Captured speech models on the GPU (ADR-0027): a HiFi-GAN vocoder and a Wav2Vec2 encoder, with
// PyTorch's weights; the GPU gradient is one a CPU validator accepts.
// ---------------------------------------------------------------------------------------------
mod speech_on_gpu {
    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex, Vulkan};
    use gg_compute::compare::compare_f32;
    use gg_compute::graph::ModelGraphV1;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_protocol::messages::ToleranceV1;
    use gg_protocol::tensor_file::{BundleF32, TensorF32, TensorLimits};
    use std::collections::BTreeMap;

    #[test]
    fn speech_models_gpu_vs_cpu() {
        let dir = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let (device, _) = super::device();
        for name in ["hifigan-mini", "wav2vec2-mini"] {
            let graph: ModelGraphV1 = serde_json::from_str(
                &std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap(),
            )
            .unwrap();
            let torch = BundleF32::decode(
                &std::fs::read(format!("{dir}/{name}.torch.ggb")).unwrap(),
                &TensorLimits::default(),
            )
            .unwrap()
            .entries;
            let prepared = Prepared::new(&graph).unwrap();
            let params: BTreeMap<String, TensorF32> = graph
                .parameters
                .iter()
                .map(|p| (p.name.clone(), torch[&p.name].clone()))
                .collect();
            let inputs = BTreeMap::from([
                (graph.inputs[0].name.clone(), torch["__x"].clone()),
                ("y".to_string(), torch["__y"].clone()),
            ]);
            let cpu = exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 0)
                .unwrap();
            let gpu =
                exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 0).unwrap();
            let tol = ToleranceV1 {
                abs: 1e-4,
                rel: 1e-3,
            };
            let mut worst = 0f32;
            let mut refused = Vec::new();
            for (p, c) in &cpu.gradients {
                let g = &gpu.gradients[p];
                worst = g
                    .data
                    .iter()
                    .zip(&c.data)
                    .map(|(a, b)| (a - b).abs())
                    .fold(worst, f32::max);
                if !compare_f32(p, g, c, tol).passed {
                    refused.push(p.clone());
                }
            }
            eprintln!(
                "[{name}] loss CPU {} GPU {}, maximum absolute gradient deviation {worst:.3e}, rejected {refused:?}",
                cpu.loss, gpu.loss
            );
            assert!((cpu.loss - gpu.loss).abs() <= 1e-4 * cpu.loss.abs().max(1.0));
            assert!(
                refused.is_empty(),
                "{name}: a CPU validator would reject {refused:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Integer tensors on the GPU (ADR-0029): given the same values, the same indices, byte for byte,
// ties and NaNs included.
// ---------------------------------------------------------------------------------------------
mod ints_on_gpu {
    use burn::backend::flex::FlexDevice;
    use burn::backend::{Flex, Vulkan};
    use gg_compute::graph::ModelGraphV1;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_protocol::drbg::Drbg;
    use gg_protocol::tensor_file::TensorF32;
    use std::collections::BTreeMap;

    #[test]
    fn indices_are_identical_on_cpu_and_gpu() {
        let (device, _) = super::device();
        // Many repeated values (ties everywhere) and NaNs (sqrt of negatives).
        let mut d = Drbg::new("gpu/ints", 1);
        let (rows, n) = (64usize, 16usize);
        let values: Vec<f32> = (0..rows * n)
            .map(|_| (d.next_index(9) as f32) - 3.0)
            .collect();
        let x = TensorF32::new(vec![rows as u64, n as u64], values).unwrap();
        for (node, width) in [
            (
                r#"{"op":"ArgMax","inputs":["h"],"output":"i","attrs":{"axis":1}}"#,
                1,
            ),
            (
                r#"{"op":"ArgMin","inputs":["h"],"output":"i","attrs":{"axis":1}}"#,
                1,
            ),
            (
                r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":1,"k":5}}"#,
                5,
            ),
            (
                r#"{"op":"ArgTopK","inputs":["h"],"output":"i","attrs":{"axis":1,"k":4,"largest":0}}"#,
                4,
            ),
            (
                r#"{"op":"ArgSort","inputs":["h"],"output":"i","attrs":{"axis":1}}"#,
                n,
            ),
            (
                r#"{"op":"ArgSort","inputs":["h"],"output":"i","attrs":{"axis":1,"descending":1}}"#,
                n,
            ),
        ] {
            for expr in ["a + b", "sqrt(a + b)"] {
                let json = format!(
                    r#"{{"format":"gg-graph/v1","name":"i","inputs":[{{"name":"x","shape":[0,{n}]}}],
                      "parameters":[{{"name":"p","shape":[{n}],"init":"zeros"}}],
                      "nodes":[{{"op":"Formula","inputs":["x","p"],"output":"h","attrs":{{"expr":"{expr}"}}}},{node},
                               {{"op":"Cast","inputs":["i"],"output":"pred","attrs":{{"to":"f32"}}}}],
                      "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,{width}]}}}}}}}}"#
                );
                let graph: ModelGraphV1 = serde_json::from_str(&json).unwrap();
                let prepared = Prepared::new(&graph).unwrap();
                let params = exec::init_parameters(&graph, 0);
                let inputs = BTreeMap::from([("x".to_string(), x.clone())]);
                let c = exec::forward::<Flex>(&FlexDevice, &prepared, &params, &inputs, 0, "pred")
                    .unwrap();
                let g = exec::forward::<Vulkan>(device, &prepared, &params, &inputs, 0, "pred")
                    .unwrap();
                assert_eq!(
                    c.data, g.data,
                    "{node} on {expr}: CPU and GPU choose differently"
                );
            }
        }
    }

    /// Scores computed by a matrix product differ by rounding between CPU and GPU; a top-k can then
    /// be decided differently on a near-tie (ADR-0029, section 4). Measured on 8192 tokens routed to
    /// 2 of 16 experts: every differing choice must be a near-tie of the CPU scores (a gap below
    /// 1e-5 relative), never a real disagreement.
    #[test]
    fn routing_flips_only_on_near_ties() {
        let (device, _) = super::device();
        let (tokens, dim, experts) = (8192u64, 64u64, 16u64);
        let json = format!(
            r#"{{"format":"gg-graph/v1","name":"r","inputs":[{{"name":"x","shape":[0,{dim}]}}],
              "parameters":[{{"name":"w","shape":[{dim},{experts}],"init":"xavier_uniform"}}],
              "nodes":[{{"op":"MatMul","inputs":["x","w"],"output":"s"}},
                       {{"op":"ArgTopK","inputs":["s"],"output":"i","attrs":{{"axis":1,"k":2}}}},
                       {{"op":"GatherElements","inputs":["s","i"],"output":"pred","attrs":{{"axis":1}}}}],
              "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,2]}}}}}}}}"#
        );
        let graph: ModelGraphV1 = serde_json::from_str(&json).unwrap();
        let prepared = Prepared::new(&graph).unwrap();
        let params = exec::init_parameters(&graph, 5);
        let mut d = Drbg::new("gpu/routing", 2);
        let x = TensorF32::new(
            vec![tokens, dim],
            (0..tokens * dim).map(|_| d.next_sym_f32()).collect(),
        )
        .unwrap();
        let inputs = BTreeMap::from([("x".to_string(), x)]);
        let run_cpu = |o: &str| {
            exec::forward::<Flex>(&FlexDevice, &prepared, &params, &inputs, 0, o).unwrap()
        };
        let (ci, cs) = (run_cpu("i"), run_cpu("s"));
        let gi = exec::forward::<Vulkan>(device, &prepared, &params, &inputs, 0, "i").unwrap();
        let mut flipped = 0;
        for t in 0..tokens as usize {
            let (c, g) = (&ci.data[2 * t..2 * t + 2], &gi.data[2 * t..2 * t + 2]);
            if c == g {
                continue;
            }
            flipped += 1;
            // The scores involved in the difference are within rounding of each other.
            let row = &cs.data[t * experts as usize..(t + 1) * experts as usize];
            let involved: Vec<f32> = c.iter().chain(g).map(|e| row[*e as usize]).collect();
            let (lo, hi) = involved
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            let differing: Vec<f32> = c
                .iter()
                .zip(g)
                .filter(|(a, b)| a != b)
                .flat_map(|(a, b)| [row[*a as usize], row[*b as usize]])
                .collect();
            let gap = differing
                .iter()
                .fold(0f32, |m, v| m.max((v - differing[0]).abs()));
            assert!(
                gap <= 1e-5 * hi.abs().max(lo.abs()).max(1.0),
                "token {t}: CPU {c:?}, GPU {g:?}, scores {involved:?}: not a near-tie"
            );
        }
        eprintln!(
            "[routing] {flipped} of {tokens} tokens routed differently by the GPU (near-ties only)"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Mixtures of experts on the GPU (ADR-0029): with PyTorch's weights, the router chooses the same
// experts on CPU and GPU (measured, token by token), and the gradient is one a CPU validator
// accepts.
// ---------------------------------------------------------------------------------------------
mod moe_on_gpu {
    use burn::backend::flex::FlexDevice;
    use burn::backend::{Autodiff, Flex, Vulkan};
    use gg_compute::compare::compare_f32;
    use gg_compute::graph::ModelGraphV1;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_protocol::messages::ToleranceV1;
    use gg_protocol::tensor_file::{BundleF32, TensorF32, TensorLimits};
    use std::collections::BTreeMap;

    #[test]
    fn moe_gpu_vs_cpu() {
        let dir = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let (device, _) = super::device();
        for name in ["mixtral-mini", "moe-top2-mini"] {
            let graph: ModelGraphV1 = serde_json::from_str(
                &std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap(),
            )
            .unwrap();
            let torch = BundleF32::decode(
                &std::fs::read(format!("{dir}/{name}.torch.ggb")).unwrap(),
                &TensorLimits::default(),
            )
            .unwrap()
            .entries;
            let prepared = Prepared::new(&graph).unwrap();
            let params: BTreeMap<String, TensorF32> = graph
                .parameters
                .iter()
                .map(|p| (p.name.clone(), torch[&p.name].clone()))
                .collect();
            let inputs = BTreeMap::from([
                (graph.inputs[0].name.clone(), torch["__x"].clone()),
                ("y".to_string(), torch["__y"].clone()),
            ]);
            // The routing: every ArgTopK output, CPU against GPU.
            let (mut choices, mut flipped) = (0usize, 0usize);
            for n in graph.nodes.iter().filter(|n| n.op == "ArgTopK") {
                let c =
                    exec::forward::<Flex>(&FlexDevice, &prepared, &params, &inputs, 0, &n.output)
                        .unwrap();
                let g = exec::forward::<Vulkan>(device, &prepared, &params, &inputs, 0, &n.output)
                    .unwrap();
                choices += c.data.len();
                flipped += c.data.iter().zip(&g.data).filter(|(a, b)| a != b).count();
            }
            let cpu = exec::gradient::<Autodiff<Flex>>(&FlexDevice, &prepared, &params, &inputs, 0)
                .unwrap();
            let gpu =
                exec::gradient::<Autodiff<Vulkan>>(device, &prepared, &params, &inputs, 0).unwrap();
            let tol = ToleranceV1 {
                abs: 1e-4,
                rel: 1e-3,
            };
            let mut worst = 0f32;
            let mut refused = Vec::new();
            for (p, c) in &cpu.gradients {
                let g = &gpu.gradients[p];
                worst = g
                    .data
                    .iter()
                    .zip(&c.data)
                    .map(|(a, b)| (a - b).abs())
                    .fold(worst, f32::max);
                if !compare_f32(p, g, c, tol).passed {
                    refused.push(p.clone());
                }
            }
            eprintln!(
                "[{name}] routing: {flipped} of {choices} expert choices differ between CPU and GPU; \
loss CPU {} GPU {}, maximum absolute gradient deviation {worst:.3e}, rejected {refused:?}",
                cpu.loss, gpu.loss
            );
            assert_eq!(flipped, 0, "{name}: the GPU routes differently");
            assert!((cpu.loss - gpu.loss).abs() <= 1e-4 * cpu.loss.abs().max(1.0));
            assert!(
                refused.is_empty(),
                "{name}: a CPU validator would reject {refused:?}"
            );
        }
    }
}

/// Gradients of described models, GPU against CPU, per trained parameter:
/// `‖g_gpu − g_cpu‖ / ‖g_cpu‖`. A ResNet18 trained on a GPU worker was rejected by the validator:
/// on WGPU/Vulkan, Burn computes a wrong gradient for a convolution whose kernel is shorter than
/// its stride (the 1×1 stride-2 shortcut), and a wrong transposed convolution in the same case,
/// forward included; single-layer tests had not seen it, the convolution being first, with no
/// input gradient. The executor rewrites those convolutions (`exec/extra.rs`); these fixtures are
/// the cases measured at 100 % error before the rewrite.
#[test]
fn graph_gradients_gpu_vs_cpu() {
    use gg_compute::graph::{self, exec};
    use std::collections::BTreeMap;
    let (device, _) = device().clone();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut cases: Vec<(std::path::PathBuf, f64)> = std::fs::read_dir(dir.join("gpu"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .map(|p| {
            // A deep network accumulates rounding differences; a single layer does not.
            let tol = if p.to_string_lossy().contains("resnet") {
                1e-3
            } else {
                1e-5
            };
            (p, tol)
        })
        .collect();
    for f in [
        "llama-hf-mini.json",
        "mixtral-mini.json",
        "hifigan-mini.json",
        "wav2vec2-mini.json",
    ] {
        cases.push((dir.join(f), 1e-4));
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    for (path, tol) in cases {
        let g: graph::ModelGraphV1 =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let prepared = exec::Prepared::new(&g).unwrap();
        let mut params = exec::init_parameters(&g, 3);
        let mut d = Drbg::new("test/gpu/graph", 1);
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
        let cpu = exec::gradient::<burn::backend::Autodiff<burn::backend::Flex>>(
            &burn::backend::flex::FlexDevice,
            &prepared,
            &params,
            &batch,
            0,
        )
        .unwrap();
        let gpu = exec::gradient::<burn::backend::Autodiff<burn::backend::Vulkan>>(
            &device, &prepared, &params, &batch, 0,
        )
        .unwrap();
        let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
        // A gradient that is zero in exact arithmetic (a bias before a normalization) is only
        // rounding noise on both sides: measured against the largest gradient of the model.
        let floor = cpu
            .gradients
            .values()
            .map(|t| norm(&mut t.data.iter().map(|v| *v as f64)))
            .fold(0.0, f64::max)
            * 1e-3;
        let mut worst = (0.0f64, String::new());
        for (name, c) in &cpu.gradients {
            let gp = &gpu.gradients[name];
            let diff = norm(&mut c.data.iter().zip(&gp.data).map(|(a, b)| (a - b) as f64));
            let err = diff / norm(&mut c.data.iter().map(|v| *v as f64)).max(floor);
            if err > worst.0 {
                worst = (err, name.clone());
            }
        }
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        eprintln!(
            "{file}: loss cpu {} gpu {}, worst relative error {:.2e} ({})",
            cpu.loss, gpu.loss, worst.0, worst.1
        );
        assert!(
            ((cpu.loss - gpu.loss) as f64).abs() <= 1e-4 * (cpu.loss as f64).abs().max(1.0),
            "{file}: loss cpu {} gpu {}",
            cpu.loss,
            gpu.loss
        );
        assert!(
            worst.0 <= tol,
            "{file}: {:.2e} on {} > {tol:e}",
            worst.0,
            worst.1
        );
        // And the rule a validator applies (`recompute-reference` v2, ADR-0031) accepts every
        // honest GPU gradient, a gradient that is zero in exact arithmetic included.
        let tolerance = gg_compute::catalog::graph_tolerance();
        for (name, c) in &cpu.gradients {
            let cmp =
                gg_compute::compare::compare_f32_norm(name, &gpu.gradients[name], c, tolerance);
            assert!(cmp.passed, "{file}: {name}: {:?}", cmp.checks);
        }
    }
}
