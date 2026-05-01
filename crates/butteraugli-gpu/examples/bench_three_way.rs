//! Wall-clock comparison: CPU butteraugli vs butteraugli-gpu (CubeCL).
//!
//! Reports min and median ms/call after warm-up over N iterations on the
//! same image pair. Per-call cost on GPU includes the host→device upload
//! of the sRGB buffers and the device→host readback of the (max, sums)
//! reduction; that's the realistic cost for a one-shot encoder lookup.
//!
//! `butteraugli-cuda` (the older PTX implementation) is the natural third
//! comparison but its `cudarse-npp` dependency uses NPP API symbols that
//! were removed in CUDA 13.x, so we currently can't link against it on
//! the same host that has CUDA 13.2 installed for cubecl. Once
//! `cudarse-npp` is updated to the `_Ctx` variants this example should
//! grow a third row.
//!
//! Usage:
//!   cargo run --release -p butteraugli-gpu --example bench_three_way
//!   cargo run --release -p butteraugli-gpu --example bench_three_way -- <path.png>

use std::env;
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli};
use butteraugli_gpu::Butteraugli as GpuButteraugli;
use image::ImageReader;
use imgref::ImgVec;
use rgb::{RGB, RGB8};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

const WARM: usize = 3;
const ITERS: usize = 15;

fn load_or_synthetic(path: Option<&str>) -> (Vec<u8>, Vec<u8>, u32, u32) {
    if let Some(p) = path {
        let img = ImageReader::open(p)
            .expect("open png")
            .decode()
            .expect("decode png")
            .to_rgb8();
        let (w, h) = (img.width(), img.height());
        let raw = img.into_raw();
        let mut dist = raw.clone();
        for y in 0..h as usize {
            for x in 0..w as usize {
                let bx = x / 8;
                let by = y / 8;
                let stripe = (((bx + by) % 3) as i32 - 1).signum();
                for c in 0..3 {
                    let v = raw[(y * w as usize + x) * 3 + c] as i32 + stripe * 4;
                    dist[(y * w as usize + x) * 3 + c] = v.clamp(0, 255) as u8;
                }
            }
        }
        (raw, dist, w, h)
    } else {
        // Synthetic — exercise multiple sizes
        let w = 1024_u32;
        let h = 1024_u32;
        let mut r = vec![0_u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                r[i] = ((x.wrapping_mul(7)) & 0xff) as u8;
                r[i + 1] = ((y.wrapping_mul(11)) & 0xff) as u8;
                r[i + 2] = (((x ^ y).wrapping_mul(13)) & 0xff) as u8;
            }
        }
        let mut d = r.clone();
        for v in d.iter_mut() {
            *v = v.saturating_add(4);
        }
        (r, d, w, h)
    }
}

fn rgb_to_imgvec(buf: &[u8], w: u32, h: u32) -> ImgVec<RGB8> {
    let pixels: Vec<RGB8> = buf
        .chunks_exact(3)
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    ImgVec::new(pixels, w as usize, h as usize)
}

fn bench<F: FnMut() -> f64>(mut f: F) -> (f64, f64, f64) {
    for _ in 0..WARM {
        let _ = f();
    }
    let mut times = Vec::with_capacity(ITERS);
    let mut last = 0.0;
    for _ in 0..ITERS {
        let t = Instant::now();
        last = f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[0], times[ITERS / 2], last)
}

fn run_size(label: &str, w: u32, h: u32, ref_rgb: &[u8], dist_rgb: &[u8]) {
    let n = (w as usize) * (h as usize);
    let mp = n as f64 / 1_000_000.0;

    let ref_iv = rgb_to_imgvec(ref_rgb, w, h);
    let dist_iv = rgb_to_imgvec(dist_rgb, w, h);
    let params_full = ButteraugliParams::default();
    let params_single = ButteraugliParams::default().with_single_resolution(true);

    let (cpu_full_min, cpu_full_med, cpu_full_score) = bench(|| {
        butteraugli(ref_iv.as_ref(), dist_iv.as_ref(), &params_full)
            .unwrap()
            .score
    });
    let (cpu_single_min, cpu_single_med, cpu_single_score) = bench(|| {
        butteraugli(ref_iv.as_ref(), dist_iv.as_ref(), &params_single)
            .unwrap()
            .score
    });

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);
    let mut gpu_cubecl = GpuButteraugli::<Backend>::new(client, w, h);
    let (cubecl_min, cubecl_med, cubecl_score) =
        bench(|| gpu_cubecl.compute(ref_rgb, dist_rgb).score as f64);

    println!("\n=== {label} ({}×{} = {:.2} MP) ===", w, h, mp);
    println!("                      median ms │  min ms │   MP/s │     score");
    println!(
        " CPU multi-res         {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cpu_full_med,
        cpu_full_min,
        mp / (cpu_full_min / 1000.0),
        cpu_full_score
    );
    println!(
        " CPU single-res        {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cpu_single_med,
        cpu_single_min,
        mp / (cpu_single_min / 1000.0),
        cpu_single_score
    );
    println!(
        " GPU CubeCL single-res {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cubecl_med,
        cubecl_min,
        mp / (cubecl_min / 1000.0),
        cubecl_score
    );
    println!(
        " speedup vs CPU multi-res:   {:>5.1}× | vs CPU single-res:   {:>5.1}×",
        cpu_full_min / cubecl_min,
        cpu_single_min / cubecl_min
    );
}

fn main() {
    let path_arg: Option<String> = env::args().nth(1);
    let (ref_rgb, dist_rgb, w, h) = load_or_synthetic(path_arg.as_deref());
    println!("warm-up={WARM}, iters={ITERS}, hw=RTX 5070 + CUDA 13.2 (CubeCL backend)");
    run_size("primary", w, h, &ref_rgb, &dist_rgb);

    // Also exercise some smaller sizes from the same data, to show the
    // per-call fixed-overhead vs per-pixel slope.
    let scaled_sizes: &[u32] = &[256, 512, 1024];
    for &s in scaled_sizes {
        if s >= w || s >= h {
            continue;
        }
        // Simple decimate to s×s — rough but consistent across runs
        let mut r = vec![0_u8; (s * s * 3) as usize];
        let mut d = vec![0_u8; (s * s * 3) as usize];
        for y in 0..s as usize {
            for x in 0..s as usize {
                let sx = x * (w as usize) / s as usize;
                let sy = y * (h as usize) / s as usize;
                for c in 0..3 {
                    r[(y * s as usize + x) * 3 + c] = ref_rgb[(sy * w as usize + sx) * 3 + c];
                    d[(y * s as usize + x) * 3 + c] = dist_rgb[(sy * w as usize + sx) * 3 + c];
                }
            }
        }
        run_size(&format!("downsampled to {s}×{s}"), s, s, &r, &d);
    }
}
