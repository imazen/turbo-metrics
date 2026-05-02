//! Cached-reference parity: scoring the same image pair via
//! `compute(ref, dis)` vs `set_reference(ref) + compute_with_reference(dis)`
//! must give identical scores. Runs the JPEG quality corpus through
//! both paths and reports the delta.

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::Ssim2;

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
    println!("source: {}×{}", w, h);

    let client_a = CudaRuntime::client(&Default::default());
    let client_b = CudaRuntime::client(&Default::default());

    // Direct path.
    let mut s_direct = Ssim2::<CudaRuntime>::new(client_a, w, h).expect("direct");
    // Cached path.
    let mut s_cached = Ssim2::<CudaRuntime>::new(client_b, w, h).expect("cached");
    s_cached.set_reference(&src_bytes).expect("set_reference");

    println!("{:>4}  {:>10}  {:>10}  {:>9}", "q", "direct", "cached", "Δ");
    let mut max_d: f64 = 0.0;
    for q in [1u32, 5, 20, 45, 70, 90] {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis_bytes, _, _) = load_rgb8(path.to_str().unwrap());

        let direct = s_direct.compute(&src_bytes, &dis_bytes).expect("direct").score;
        let cached = s_cached
            .compute_with_reference(&dis_bytes)
            .expect("cached")
            .score;
        let d = (direct - cached).abs();
        if d > max_d {
            max_d = d;
        }
        println!("{:>4}  {:>10.4}  {:>10.4}  {:>9.6}", q, direct, cached, d);
    }
    if max_d > 1e-4 {
        println!("FAIL: max drift {max_d:.6} > 1e-4");
        std::process::exit(1);
    }
    println!("Cached-reference path matches direct path (max Δ {:.3e}).", max_d);
}
