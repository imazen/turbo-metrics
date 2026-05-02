//! Smoke test for `Ssim2Batch`: cache one reference, score N
//! distorted images, verify each result matches `compute(ref, dis)`.

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
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

    let dis_paths = [1u32, 5, 20, 45, 70, 90]
        .iter()
        .map(|q| dir.join(format!("q{q}.jpg")))
        .collect::<Vec<_>>();
    let dis: Vec<Vec<u8>> = dis_paths
        .iter()
        .map(|p| load_rgb8(p.to_str().unwrap()).0)
        .collect();

    // Reference path: one Ssim2 instance, run compute() per pair.
    let client = CudaRuntime::client(&Default::default());
    let mut single = Ssim2::<CudaRuntime>::new(client, w, h).expect("Ssim2::new");
    let single_results: Vec<f64> = dis
        .iter()
        .map(|d| single.compute(&src_bytes, d).expect("compute").score)
        .collect();

    // Batch path.
    let client = CudaRuntime::client(&Default::default());
    let mut batch = Ssim2Batch::<CudaRuntime>::new(client, w, h).expect("batch");
    batch.set_reference(&src_bytes).expect("set_reference");
    let batch_results = batch.compute_many(&dis).expect("batch compute_many");

    assert_eq!(single_results.len(), batch_results.len());
    let mut max_d: f64 = 0.0;
    for (s, b) in single_results.iter().zip(batch_results.iter()) {
        let d = (*s - b.score).abs();
        if d > max_d {
            max_d = d;
        }
    }
    println!("Ssim2Batch ↔ Ssim2 max drift: {max_d:.3e}");
    assert!(max_d < 1e-4, "batch and single paths disagree by {max_d}");
    println!("Ssim2Batch smoke test passed for {} images.", dis.len());
}
