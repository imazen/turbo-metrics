//! Measure the per-instance setup cost (allocation + first-launch
//! kernel codegen) for various dimensions, plus the steady-state cost
//! of running on a different dimension after the kernels are warm.
//!
//! This answers: "what's the overhead of changing image dimensions
//! between `compute` calls?"

use std::time::Instant;

use butteraugli_gpu::Butteraugli;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn make_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w * h * 3) as usize;
    let mut r = vec![128_u8; n];
    let mut d = r.clone();
    for (i, v) in d.iter_mut().enumerate() {
        *v = ((*v as i32 + (i as i32 % 7) - 3) as u8).clamp(0, 255);
    }
    for i in 0..n {
        r[i] = ((i.wrapping_mul(7)) & 0xff) as u8;
    }
    (r, d)
}

fn time_ms<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

fn main() {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    println!("Per-instance setup cost (allocation + first kernel launch):");
    println!(
        "  size           |  Butteraugli::new  | first compute() | second compute() | steady min over 10"
    );

    let sizes: &[(u32, u32)] = &[
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];

    // Warm cubecl kernel cache once with a throwaway 64×64 instance.
    {
        let mut warmup = Butteraugli::<Backend>::new(client.clone(), 64, 64);
        let (r, d) = make_pair(64, 64);
        let _ = warmup.compute(&r, &d);
    }

    for &(w, h) in sizes {
        let (r, d) = make_pair(w, h);

        let (mut bu, t_alloc) = time_ms(|| Butteraugli::<Backend>::new(client.clone(), w, h));
        let (_, t_first) = time_ms(|| bu.compute(&r, &d));
        let (_, t_second) = time_ms(|| bu.compute(&r, &d));
        let mut steady = f64::INFINITY;
        for _ in 0..10 {
            let (_, t) = time_ms(|| bu.compute(&r, &d));
            if t < steady {
                steady = t;
            }
        }
        println!(
            "  {:>4}×{:<4}     |  {:>8.2} ms        |  {:>7.2} ms     |  {:>7.2} ms      | {:>7.2} ms",
            w, h, t_alloc, t_first, t_second, steady,
        );
    }

    println!("\nDim-switching scenario (alternate sizes, kernels already warm):");
    println!("  pre-allocate one Butteraugli per size, then alternate compute() calls");

    let mut instances: Vec<(u32, Butteraugli<Backend>)> = sizes
        .iter()
        .map(|&(w, h)| (w, Butteraugli::<Backend>::new(client.clone(), w, h)))
        .collect();
    let pairs: Vec<(u32, Vec<u8>, Vec<u8>)> = sizes
        .iter()
        .map(|&(w, h)| {
            let (r, d) = make_pair(w, h);
            (w, r, d)
        })
        .collect();

    // Warm each one
    for (i, (_w, bu)) in instances.iter_mut().enumerate() {
        let (_, r, d) = &pairs[i];
        let _ = bu.compute(r, d);
    }

    // Round-robin alternation
    let n_iters = 30;
    let t = Instant::now();
    for it in 0..n_iters {
        let i = it % instances.len();
        let (_, r, d) = &pairs[i];
        let _ = instances[i].1.compute(r, d);
    }
    let dt = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  {n_iters} round-robin compute() calls across {} sizes: {:.2} ms total ({:.2} ms/call avg)",
        sizes.len(),
        dt,
        dt / n_iters as f64,
    );

    // Same iteration count using ONLY the largest instance for comparison
    let big_idx = instances.len() - 1;
    let (_, r, d) = &pairs[big_idx];
    let t = Instant::now();
    for _ in 0..n_iters {
        let _ = instances[big_idx].1.compute(r, d);
    }
    let dt_one = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  same n_iters using only {}×{} (no switching): {:.2} ms total ({:.2} ms/call avg)",
        sizes[big_idx].0,
        sizes[big_idx].1,
        dt_one,
        dt_one / n_iters as f64,
    );

    std::process::exit(0);
}
