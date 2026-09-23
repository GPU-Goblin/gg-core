//! The overhead of tiling at a realistic size: is it the same order of magnitude as for small matrices?
use std::collections::BTreeMap;
use std::time::Instant;

use gg_compute::graph::ModelGraphV1;
use gg_compute::graph::exec::{self, ExecOptions, Prepared};
use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

fn random(shape: &[u64], seed: u64) -> TensorF32 {
    let mut d = Drbg::new("scale", seed);
    let n: usize = shape.iter().product::<u64>() as usize;
    TensorF32::new(
        shape.to_vec(),
        (0..n).map(|_| d.next_sym_f32() * 0.05).collect(),
    )
    .unwrap()
}

fn build(d_in: u64, d_out: u64, tiles: u64) -> ModelGraphV1 {
    let width = d_out / tiles;
    let params: Vec<String> = (0..tiles)
        .map(|i| format!(r#"{{"name":"w{i}","shape":[{d_in},{width}],"init":"zeros"}}"#))
        .collect();
    let mut nodes: Vec<String> = (0..tiles)
        .map(|i| format!(r#"{{"op":"MatMul","inputs":["x","w{i}"],"output":"h{i}"}}"#))
        .collect();
    let mut parts: Vec<String> = (0..tiles).map(|i| format!("h{i}")).collect();
    let mut round = 0;
    while parts.len() > 1 {
        let mut next = Vec::new();
        for (g, chunk) in parts.chunks(8).enumerate() {
            if chunk.len() == 1 {
                next.push(chunk[0].clone());
                continue;
            }
            let out = if parts.len() <= 8 {
                "h".to_string()
            } else {
                format!("c{round}_{g}")
            };
            nodes.push(format!(
                r#"{{"op":"Concat","inputs":[{}],"output":"{out}","attrs":{{"axis":1}}}}"#,
                chunk
                    .iter()
                    .map(|n| format!("\"{n}\""))
                    .collect::<Vec<_>>()
                    .join(",")
            ));
            next.push(out);
        }
        parts = next;
        round += 1;
    }
    if tiles == 1 {
        nodes = vec![r#"{"op":"MatMul","inputs":["x","w0"],"output":"h"}"#.to_string()];
    }
    nodes.push(r#"{"op":"Relu","inputs":["h"],"output":"pred"}"#.to_string());
    serde_json::from_str(&format!(
        r#"{{"format":"gg-graph/v1","name":"t","inputs":[{{"name":"x","shape":[0,{d_in}]}}],
        "parameters":[{}],"nodes":[{}],
        "loss":{{"mean_squared_error":{{"prediction":"pred","target":{{"name":"y","shape":[0,{d_out}]}}}}}}}}"#,
        params.join(","), nodes.join(",")
    )).unwrap()
}

fn split(w: &TensorF32, d_in: u64, d_out: u64, tiles: u64) -> BTreeMap<String, TensorF32> {
    let width = (d_out / tiles) as usize;
    let mut out = BTreeMap::new();
    for i in 0..tiles as usize {
        let mut data = Vec::with_capacity(d_in as usize * width);
        for row in 0..d_in as usize {
            let s = row * d_out as usize + i * width;
            data.extend_from_slice(&w.data[s..s + width]);
        }
        out.insert(
            format!("w{i}"),
            TensorF32::new(vec![d_in, width as u64], data).unwrap(),
        );
    }
    out
}

#[cfg(feature = "gpu")]
fn gpu_device() -> &'static burn::backend::wgpu::WgpuDevice {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<burn::backend::wgpu::WgpuDevice> = OnceLock::new();
    DEVICE.get_or_init(|| {
        let (d, info) = gg_compute::engine::gpu::init_vulkan(0).expect("GPU Vulkan");
        eprintln!("engine: {} ({})", info.device_name, info.engine);
        d
    })
}

#[cfg(feature = "gpu")]
fn measure_gpu(
    prepared: &Prepared,
    p: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    opts: ExecOptions,
    reps: usize,
) -> (f64, exec::StepResult) {
    use burn::backend::{Autodiff, Vulkan};
    let device = gpu_device();
    let run =
        || exec::gradient_with::<Autodiff<Vulkan>>(device, prepared, p, inputs, 7, opts).unwrap();
    let first = run();
    let start = Instant::now();
    for _ in 0..reps {
        run();
    }
    (start.elapsed().as_secs_f64() / reps as f64, first)
}

fn main() {
    #[cfg(not(feature = "gpu"))]
    eprintln!("engine: CPU (burn flex)");
    for (d_in, d_out, n) in [(64u64, 256u64, 8u64), (512, 2048, 32), (2048, 8192, 64)] {
        let w = random(&[d_in, d_out], 3);
        let inputs = BTreeMap::from([
            ("x".to_string(), random(&[n, d_in], 1)),
            ("y".to_string(), random(&[n, d_out], 2)),
        ]);
        let flops = 2.0 * (n * d_in * d_out) as f64 * 3.0;
        println!(
            "\n== weights {d_in}x{d_out} ({:.1} MiB), batch {n}, {:.2} MFLOP per step",
            (d_in * d_out * 4) as f64 / 1048576.0,
            flops / 1e6
        );
        let mut base = 0.0;
        for tiles in [1u64, 4, 16, 64] {
            if d_out % tiles != 0 {
                continue;
            }
            let g = build(d_in, d_out, tiles);
            let p = split(&w, d_in, d_out, tiles);
            let prepared = Prepared::new(&g).expect("validated");
            let opts = if tiles == 1 {
                ExecOptions::default()
            } else {
                ExecOptions {
                    segments: usize::MAX,
                    offload_parameters: true,
                }
            };
            let reps = if d_in > 1000 { 3 } else { 20 };
            #[cfg(feature = "gpu")]
            let (t, first) = measure_gpu(&prepared, &p, &inputs, opts, reps.max(5));
            #[cfg(not(feature = "gpu"))]
            let (t, first) = {
                let run = || exec::gradient_cpu_with(&prepared, &p, &inputs, 7, opts).unwrap();
                let first = run();
                let start = Instant::now();
                for _ in 0..reps {
                    run();
                }
                (start.elapsed().as_secs_f64() / reps as f64, first)
            };
            if tiles == 1 {
                base = t;
            }
            println!(
                "  {tiles:>2} tile(s): {:>8.2} ms ({:.2}x), peak parameters {:>10} B, loss {:.6}",
                t * 1e3,
                t / base,
                first.peak_parameter_bytes,
                first.loss
            );
        }
    }
}
