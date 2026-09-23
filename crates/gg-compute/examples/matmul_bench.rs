//! Throughput of the bare matrix product on the GPU, at transformer shapes (D5).
//! `cargo run -p gg-compute --release --features gpu --example matmul_bench`
#[cfg(feature = "gpu")]
fn main() {
    use burn::backend::Vulkan;
    use burn::tensor::{Distribution, Tensor};
    let (device, info) = gg_compute::engine::gpu::init_vulkan(0).unwrap();
    println!("{}", info.device_name);
    for (m, k, n) in [
        (2048usize, 768usize, 768usize),
        (2048, 768, 3072),
        (2048, 3072, 768),
        (4096, 4096, 4096),
        (2048, 768, 8000),
    ] {
        let a: Tensor<Vulkan, 2> = Tensor::random([m, k], Distribution::Default, &device);
        let b: Tensor<Vulkan, 2> = Tensor::random([k, n], Distribution::Default, &device);
        let _ = a.clone().matmul(b.clone()).into_data();
        let reps = 20;
        let start = std::time::Instant::now();
        let mut acc = a.clone().matmul(b.clone());
        for _ in 1..reps {
            acc = acc + a.clone().matmul(b.clone());
        }
        let _ = acc.into_data();
        let s = start.elapsed().as_secs_f64();
        println!(
            "{m}×{k} @ {k}×{n} : {:.0} GFLOP/s",
            2.0 * (m * k * n) as f64 * reps as f64 / s / 1e9
        );
    }
}
#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("build with --features gpu");
}
