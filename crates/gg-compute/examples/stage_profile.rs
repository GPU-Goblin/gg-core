//! Where the time goes in a forward pass of a realistic stage on the GPU (D5).
//! `GG_PROFILE=1 cargo run -p gg-compute --release --features gpu --example stage_profile`
#[cfg(feature = "gpu")]
fn main() {
    use burn::backend::Vulkan;
    use gg_compute::graph::exec::{self, Prepared};
    use gg_compute::graph::templates::{Positions, TransformerDims, transformer};
    use gg_protocol::tensor_file::TensorF32;
    use std::collections::BTreeMap;
    let graph = transformer(TransformerDims {
        layers: 4,
        dim: 768,
        heads: 12,
        seq: 256,
        vocab: 8000,
        ffn: 3072,
        causal: true,
        positions: Positions::None,
    })
    .unwrap();
    let prepared = Prepared::new(&graph).unwrap();
    let params = exec::init_parameters(&graph, 5);
    let ids = TensorF32::new(vec![8, 256], (0..2048).map(|i| (i % 8000) as f32).collect()).unwrap();
    let inputs = BTreeMap::from([("ids".to_string(), ids.clone()), ("y".to_string(), ids)]);
    let t = std::time::Instant::now();
    let (device, _) = gg_compute::engine::gpu::init_vulkan(0).unwrap();
    println!(
        "Vulkan initialization: {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    {
        use burn::tensor::{Tensor, TensorData};
        let t = std::time::Instant::now();
        let mut n = 0usize;
        let mut keep = Vec::new();
        for p in params.values() {
            n += p.data.len();
            let shape: Vec<usize> = p.shape.iter().map(|d| *d as usize).collect();
            let data = TensorData::new(p.data.clone(), shape.clone());
            keep.push(Tensor::<Vulkan, 1>::from_data(
                TensorData::new(p.data.clone(), vec![p.data.len()]),
                &device,
            ));
            let _ = data;
        }
        <Vulkan as burn::tensor::backend::Backend>::sync(&device).unwrap();
        println!(
            "upload of {n} parameters ({:.0} MB): {:.1} ms",
            n as f64 * 4.0 / 1e6,
            t.elapsed().as_secs_f64() * 1e3
        );
        // A single upload: all parameters back to back, then split on the GPU.
        let t = std::time::Instant::now();
        let mut flat = Vec::with_capacity(n);
        for p in params.values() {
            flat.extend_from_slice(&p.data);
        }
        let t_pack = t.elapsed();
        let big = Tensor::<Vulkan, 1>::from_data(TensorData::new(flat, vec![n]), &device);
        let mut off = 0;
        let mut views = Vec::new();
        for p in params.values() {
            views.push(big.clone().narrow(0, off, p.data.len()));
            off += p.data.len();
        }
        <Vulkan as burn::tensor::backend::Backend>::sync(&device).unwrap();
        println!(
            "single upload: {:.1} ms (of which {:.1} ms host copy)",
            t.elapsed().as_secs_f64() * 1e3,
            t_pack.as_secs_f64() * 1e3
        );
        let _ = views;
        let t = std::time::Instant::now();
        let mut back = 0usize;
        for k in keep {
            back += k.into_data().as_slice::<f32>().unwrap().len();
        }
        println!(
            "readback of {back} values: {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    for pass in 0..2 {
        let t = std::time::Instant::now();
        let _ = exec::forward::<Vulkan>(&device, &prepared, &params, &inputs, 1, "logits").unwrap();
        if pass == 0 {
            println!(
                "first forward pass (including kernel compilation): {:.1} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let total = t.elapsed();
        let prof = exec::take_profile();
        if pass == 1 {
            let mut v: Vec<_> = prof.into_iter().collect();
            v.sort_by_key(|e| std::cmp::Reverse(e.1.1));
            println!("forward pass: {:.1} ms", total.as_secs_f64() * 1e3);
            for (op, (n, d)) in v {
                println!("  {op:<12} {n:>4} nodes {:>8.2} ms", d.as_secs_f64() * 1e3);
            }
        }
    }
}
#[cfg(not(feature = "gpu"))]
fn main() {}
