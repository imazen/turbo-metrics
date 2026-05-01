//! Diagnostic: compare GPU vs CPU diffmap pixel-by-pixel to localize the
//! divergence source.

use butteraugli::{ButteraugliParams, butteraugli};
use butteraugli_gpu::Butteraugli;
use imgref::ImgVec;
use rgb::{RGB, RGB8};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn make_image(width: u32, height: u32, salt: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x.wrapping_mul(7).wrapping_add(salt)) & 0xff) as u8;
            let g = ((y.wrapping_mul(11).wrapping_add(salt.wrapping_mul(3))) & 0xff) as u8;
            let b = (((x ^ y).wrapping_mul(13).wrapping_add(salt.wrapping_mul(5))) & 0xff) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

fn perturb(src: &[u8], width: u32, _height: u32, mag: i32) -> Vec<u8> {
    let mut out = src.to_vec();
    let w = width as usize;
    let n = src.len() / 3;
    for i in 0..n {
        let x = i % w;
        let y = i / w;
        let stripe = (((x + y) % 7) as i32 - 3).signum();
        for c in 0..3 {
            let delta = stripe * mag * (1 + c as i32);
            let v = src[i * 3 + c] as i32 + delta;
            out[i * 3 + c] = v.clamp(0, 255) as u8;
        }
    }
    out
}

fn rgb_buf_to_imgvec(buf: &[u8], width: u32, height: u32) -> ImgVec<RGB8> {
    let pixels: Vec<RGB8> = buf
        .chunks_exact(3)
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    ImgVec::new(pixels, width as usize, height as usize)
}

fn main() {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let width = 64u32;
    let height = 64u32;
    let mag = 12_i32;

    // Flat gray image with one perturbed pixel — easiest case to debug.
    // Diffmap should be the convolution kernel shape centered on the
    // perturbed pixel, modulated by the mask.
    let mut ref_rgb = vec![128_u8; (width * height * 3) as usize];
    let mut dist_rgb = ref_rgb.clone();
    let cx = (width / 2) as usize;
    let cy = (height / 2) as usize;
    for c in 0..3 {
        dist_rgb[(cy * width as usize + cx) * 3 + c] = 200;
    }
    let _ = (make_image, perturb); // keep helpers for swap tests
    println!(
        "[flat-128 with single perturbed pixel at ({}, {})] mag (kept for variable use): {}",
        cx, cy, mag
    );

    // Sanity: identical images should give zero diffmap.
    {
        let cpu_zero = butteraugli(
            rgb_buf_to_imgvec(&ref_rgb, width, height).as_ref(),
            rgb_buf_to_imgvec(&ref_rgb, width, height).as_ref(),
            &ButteraugliParams::default().with_single_resolution(true),
        )
        .unwrap();
        let mut g0 = Butteraugli::<Backend>::new(client.clone(), width, height);
        let g0_res = g0.compute(&ref_rgb, &ref_rgb);
        println!(
            "[identical-image sanity] CPU score={:.6}  GPU score={:.6}  GPU pnorm={:.6}",
            cpu_zero.score, g0_res.score, g0_res.pnorm_3,
        );
    }

    // CPU
    let ref_img = rgb_buf_to_imgvec(&ref_rgb, width, height);
    let dist_img = rgb_buf_to_imgvec(&dist_rgb, width, height);
    let params = ButteraugliParams::default()
        .with_single_resolution(true)
        .with_compute_diffmap(true);
    let cpu = butteraugli(ref_img.as_ref(), dist_img.as_ref(), &params).unwrap();
    let cpu_dm = cpu.diffmap.expect("diffmap requested");
    let cpu_w = cpu_dm.width();
    let cpu_h = cpu_dm.height();
    let cpu_buf = cpu_dm.buf();

    // GPU
    let mut gpu = Butteraugli::<Backend>::new(client, width, height);
    let gres = gpu.compute(&ref_rgb, &dist_rgb);
    let gpu_dm = gpu.copy_diffmap();

    // Extract CPU diffmap (ImgVec is contiguous when produced by butteraugli)
    let cpu_flat: Vec<f32> = cpu_buf.to_vec();

    println!(
        "CPU diffmap {}×{}, GPU diffmap {} pixels",
        cpu_w,
        cpu_h,
        gpu_dm.len()
    );
    assert_eq!(cpu_flat.len(), gpu_dm.len());

    // Stats
    let mut max_cpu = 0.0f32;
    let mut max_gpu = 0.0f32;
    let mut sum_cpu = 0.0f64;
    let mut sum_gpu = 0.0f64;
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut argmax_idx = 0usize;
    for i in 0..cpu_flat.len() {
        let c = cpu_flat[i];
        let g = gpu_dm[i];
        max_cpu = max_cpu.max(c);
        max_gpu = max_gpu.max(g);
        sum_cpu += c as f64;
        sum_gpu += g as f64;
        let abs_diff = (c - g).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            argmax_idx = i;
        }
        if c > 1e-3 {
            let rel = abs_diff / c;
            if rel > max_rel_diff {
                max_rel_diff = rel;
            }
        }
    }

    println!(
        "GPU score (max-norm)={}  CPU score={}",
        gres.score, cpu.score,
    );
    println!(
        "max_cpu={:.4} max_gpu={:.4} ratio={:.3}",
        max_cpu,
        max_gpu,
        max_gpu / max_cpu
    );
    println!(
        "mean_cpu={:.4} mean_gpu={:.4} ratio={:.3}",
        sum_cpu / cpu_flat.len() as f64,
        sum_gpu / cpu_flat.len() as f64,
        sum_gpu / sum_cpu
    );
    println!(
        "max abs diff={:.4} at idx={} (cpu={:.4} gpu={:.4})",
        max_abs_diff, argmax_idx, cpu_flat[argmax_idx], gpu_dm[argmax_idx]
    );
    println!("max rel diff (where cpu>1e-3) = {:.3}", max_rel_diff);

    // Inspect intermediate GPU buffers at (32, 4) — the supposed-zero point.
    let probe = (4 * width as usize) + 32; // (32, 4)
    let center = (32 * width as usize) + 32; // (32, 32)
    let print_at = |name: &str, plane: &[f32]| {
        println!(
            "  {name:>22}  @(32,4)={:>+12.6}  @(32,32)={:>+12.6}",
            plane[probe], plane[center]
        );
    };
    println!("\n--- GPU intermediate buffers ---");
    print_at("mask", &gpu.debug_mask());
    print_at("block_diff_ac[X]", &gpu.debug_block_diff_ac(0));
    print_at("block_diff_ac[Y]", &gpu.debug_block_diff_ac(1));
    print_at("block_diff_ac[B]", &gpu.debug_block_diff_ac(2));
    print_at("block_diff_dc[X]", &gpu.debug_block_diff_dc(0));
    print_at("block_diff_dc[Y]", &gpu.debug_block_diff_dc(1));
    print_at("block_diff_dc[B]", &gpu.debug_block_diff_dc(2));
    print_at("LF_a[X]", &gpu.debug_lf(true, 0));
    print_at("LF_b[X]", &gpu.debug_lf(false, 0));
    print_at("LF_a[Y]", &gpu.debug_lf(true, 1));
    print_at("LF_b[Y]", &gpu.debug_lf(false, 1));
    print_at("HF_a[Y]", &gpu.debug_freq(true, 1, 1));
    print_at("HF_b[Y]", &gpu.debug_freq(false, 1, 1));
    print_at("MF_a[Y]", &gpu.debug_freq(true, 2, 1));
    print_at("MF_b[Y]", &gpu.debug_freq(false, 2, 1));
    print_at("UHF_a[Y]", &gpu.debug_freq(true, 0, 1));
    print_at("UHF_b[Y]", &gpu.debug_freq(false, 0, 1));

    // Sample 9 evenly-spaced pixels for ratio inspection
    println!("\n--- 9 sample pixels (cpu / gpu / ratio) ---");
    for sy in [4, height as usize / 2, height as usize - 5] {
        for sx in [4, width as usize / 2, width as usize - 5] {
            let idx = sy * width as usize + sx;
            println!(
                "  ({:>3},{:>3}) cpu={:.4}  gpu={:.4}  ratio={:.3}",
                sx,
                sy,
                cpu_flat[idx],
                gpu_dm[idx],
                if cpu_flat[idx] > 1e-6 {
                    gpu_dm[idx] / cpu_flat[idx]
                } else {
                    0.0
                }
            );
        }
    }
}
