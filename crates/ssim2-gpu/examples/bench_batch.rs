//! Per-image throughput: `Ssim2::compute_with_reference` (sequential
//! cached path) vs `Ssim2Batch::compute_batch` (kernel-batched).
//!
//! Same inputs in both paths — the JPEG quality corpus repeated
//! `repeats` times to reach a stable timing window. Reports per-image
//! ms and the speedup ratio.
//!
//! ## Caveats
//!
//! - This is a wall-clock test on RTX 5070 + CUDA 13.2. CubeCL handles
//!   stream submission internally; we don't insert explicit syncs
//!   between calls in the sequential path, so the measured time
//!   covers submission + execution end-to-end (the `read_one` inside
//!   each call does an implicit sync, so per-call latency is
//!   measured).
//! - Cold-launch effects mean the first iteration is always the
//!   slowest. We report median of `repeats` outer trials.

use std::time::Instant;

use cubecl::Runtime;
#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
use ssim2_gpu::{Ssim2, Ssim2Batch};

const CORPUS_DIR: &str = "../dssim-cuda/test_data";

fn load_rgb8(path: &str) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let rgb8 = img.to_rgb8();
    let (w, h) = (rgb8.width(), rgb8.height());
    (rgb8.into_raw(), w, h)
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn run(qs_per_call: usize) {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(manifest).join(CORPUS_DIR);
    let (src_bytes, w, h) = load_rgb8(dir.join("source.png").to_str().unwrap());

    let qs = [1u32, 5, 20, 45, 70, 90];
    let dis_pool: Vec<Vec<u8>> = qs
        .iter()
        .map(|q| load_rgb8(dir.join(format!("q{q}.jpg")).to_str().unwrap()).0)
        .collect();
    let mut dis: Vec<Vec<u8>> = Vec::with_capacity(qs_per_call);
    for i in 0..qs_per_call {
        dis.push(dis_pool[i % dis_pool.len()].clone());
    }

    // Sequential (cached-reference) path.
    let client = Backend::client(&Default::default());
    let mut sequential = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    sequential.set_reference(&src_bytes).expect("set_reference");
    // Warmup.
    for d in &dis {
        let _ = sequential.compute_with_reference(d).expect("compute");
    }
    let trials = 5;
    let mut seq_ms = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        for d in &dis {
            let _ = sequential.compute_with_reference(d).expect("compute");
        }
        seq_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let seq_total = median(&mut seq_ms);
    let seq_per = seq_total / qs_per_call as f64;

    // Batched path.
    let client = Backend::client(&Default::default());
    let mut batch =
        Ssim2Batch::<Backend>::new(client, w, h, qs_per_call as u32).expect("Ssim2Batch::new");
    batch.set_reference(&src_bytes).expect("set_reference");
    // Warmup.
    let _ = batch.compute_batch(&dis).expect("compute_batch");
    let mut batch_ms = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let _ = batch.compute_batch(&dis).expect("compute_batch");
        batch_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let batch_total = median(&mut batch_ms);
    let batch_per = batch_total / qs_per_call as f64;

    let speedup = seq_per / batch_per;
    println!(
        "{:>3}×  seq {:>7.2} ms total ({:>5.2} ms/img)  |  batch {:>7.2} ms total ({:>5.2} ms/img)  |  {:>4.2}× per-image",
        qs_per_call, seq_total, seq_per, batch_total, batch_per, speedup
    );
}

fn main() {
    println!("256×256 source, JPEG corpus repeated to fill batch slots");
    println!();
    for n in [1usize, 2, 4, 8, 16] {
        run(n);
    }
}
