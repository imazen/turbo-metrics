//! Wall-clock comparison: CPU butteraugli vs butteraugli-cuda (PTX) vs
//! butteraugli-gpu (CubeCL). Reports min and median ms/call after warm-up
//! over N iterations on the same image pair. Per-call cost on both GPUs
//! includes the host→device upload of sRGB inputs and the device→host
//! readback of the (max, sums) reduction.
//!
//! Usage:
//!   cargo run --release -p butteraugli-gpu --example bench_three_way
//!   cargo run --release -p butteraugli-gpu --example bench_three_way -- <path.png>

use std::env;
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli};
use butteraugli_gpu::Butteraugli as GpuButteraugli;
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{C, Image, Img, ImgMut};
use cudarse_npp::set_stream;
use image::ImageReader;
use imgref::ImgVec;
use rgb::{RGB, RGB8};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

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

    // ── CPU butteraugli ──
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

    // ── butteraugli-gpu (CubeCL) ──
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);
    let mut gpu_single = GpuButteraugli::<Backend>::new(client.clone(), w, h);
    let mut gpu_multi = GpuButteraugli::<Backend>::new_multires(client.clone(), w, h);
    let mut gpu_cached = GpuButteraugli::<Backend>::new_multires(client, w, h);
    gpu_cached.set_reference(ref_rgb);
    let (cubecl_min, cubecl_med, cubecl_score) =
        bench(|| gpu_single.compute(ref_rgb, dist_rgb).score as f64);
    let (cubecl_multi_min, cubecl_multi_med, cubecl_multi_score) =
        bench(|| gpu_multi.compute(ref_rgb, dist_rgb).score as f64);
    let (cubecl_cached_min, cubecl_cached_med, cubecl_cached_score) =
        bench(|| gpu_cached.compute_with_reference(dist_rgb).score as f64);

    // ── butteraugli-cuda (PTX, multi-resolution) ──
    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();
    let mut gpu_ref = Image::<u8, C<3>>::malloc(w, h).unwrap();
    let mut gpu_dis = gpu_ref.malloc_same_size().unwrap();
    let mut gpu_cuda = butteraugli_cuda::Butteraugli::new(w, h).unwrap();
    let (cuda_min, cuda_med, cuda_score) = bench(|| {
        // Re-upload each iteration for a fair comparison with the GPU
        // CubeCL implementation, which uploads on every `.compute()`.
        gpu_ref.copy_from_cpu(ref_rgb, stream.inner() as _).unwrap();
        gpu_dis
            .copy_from_cpu(dist_rgb, stream.inner() as _)
            .unwrap();
        let s = gpu_cuda
            .compute(gpu_ref.full_view(), gpu_dis.full_view())
            .unwrap();
        s as f64
    });

    println!("\n=== {label} ({}×{} = {:.2} MP) ===", w, h, mp);
    println!("                            median ms │  min ms │   MP/s │     score");
    println!(
        " CPU multi-res               {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cpu_full_med,
        cpu_full_min,
        mp / (cpu_full_min / 1000.0),
        cpu_full_score
    );
    println!(
        " CPU single-res              {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cpu_single_med,
        cpu_single_min,
        mp / (cpu_single_min / 1000.0),
        cpu_single_score
    );
    println!(
        " butteraugli-cuda (PTX)      {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cuda_med,
        cuda_min,
        mp / (cuda_min / 1000.0),
        cuda_score
    );
    println!(
        " CubeCL single-res           {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cubecl_med,
        cubecl_min,
        mp / (cubecl_min / 1000.0),
        cubecl_score
    );
    println!(
        " CubeCL multi-res            {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cubecl_multi_med,
        cubecl_multi_min,
        mp / (cubecl_multi_min / 1000.0),
        cubecl_multi_score
    );
    println!(
        " CubeCL cached-ref multi-res {:>7.2} │ {:>7.2} │ {:>6.1} │  {:>8.4}",
        cubecl_cached_med,
        cubecl_cached_med, // print median twice — cached path is more variance-prone
        mp / (cubecl_cached_min / 1000.0),
        cubecl_cached_score,
    );
    println!(
        " speedup vs CPU multi-res:   PTX={:.1}× | CubeCL multi-res={:.1}× | cached-ref={:.1}×",
        cpu_full_min / cuda_min,
        cpu_full_min / cubecl_multi_min,
        cpu_full_min / cubecl_cached_min,
    );
    println!(
        " cache speedup vs full multi-res: {:.2}×",
        cubecl_multi_min / cubecl_cached_min,
    );
}

fn main() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("cuda init");

    let path_arg: Option<String> = env::args().nth(1);
    let (ref_rgb, dist_rgb, w, h) = load_or_synthetic(path_arg.as_deref());
    println!("warm-up={WARM}, iters={ITERS}, hw=RTX 5070 + CUDA 13.2");
    run_size("primary", w, h, &ref_rgb, &dist_rgb);

    // Also exercise smaller sizes from the same data.
    let scaled_sizes: &[u32] = &[256, 512, 1024];
    for &s in scaled_sizes {
        if s >= w || s >= h {
            continue;
        }
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

    std::process::exit(0); // avoid CUDA cleanup crash
}
