//! Smoke test for `Ssim2Batch`: cache one reference, score N
//! distorted images via `compute_batch`, verify each result matches
//! `Ssim2::compute(ref, dis)` to within 1e-4 absolute.

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

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(manifest).join(CORPUS_DIR);

    let (src_bytes, w, h) = load_rgb8(dir.join("source.png").to_str().unwrap());

    let qs = [1u32, 5, 20, 45, 70, 90];
    let dis: Vec<Vec<u8>> = qs
        .iter()
        .map(|q| load_rgb8(dir.join(format!("q{q}.jpg")).to_str().unwrap()).0)
        .collect();

    // Reference path: one Ssim2 instance, run compute() per pair.
    let client = Backend::client(&Default::default());
    let mut single = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let single_results: Vec<f64> = dis
        .iter()
        .map(|d| single.compute(&src_bytes, d).expect("compute").score)
        .collect();

    // Batch path: one Ssim2Batch handles all 6 in a single compute_batch.
    let client = Backend::client(&Default::default());
    let mut batch =
        Ssim2Batch::<Backend>::new(client, w, h, dis.len() as u32).expect("Ssim2Batch::new");
    batch.set_reference(&src_bytes).expect("set_reference");
    let batch_results = batch.compute_batch(&dis).expect("compute_batch");

    assert_eq!(single_results.len(), batch_results.len());
    let mut max_d: f64 = 0.0;
    println!("{:>4}  {:>10}  {:>10}  {:>9}", "q", "single", "batch", "Δ");
    for (i, (s, b)) in single_results.iter().zip(batch_results.iter()).enumerate() {
        let d = (*s - b.score).abs();
        if d > max_d {
            max_d = d;
        }
        println!("{:>4}  {:>10.4}  {:>10.4}  {:>9.6}", qs[i], s, b.score, d);
    }
    println!("Ssim2Batch ↔ Ssim2 max drift: {max_d:.3e}");
    assert!(max_d < 1e-4, "batch and single paths disagree by {max_d}");
    println!("Ssim2Batch smoke test passed for {} images.", dis.len());
}
