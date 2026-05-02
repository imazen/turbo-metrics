//! Parity check on a real PNG: GPU butteraugli vs CPU butteraugli v0.9.2
//! using `single_resolution=true` so both run the same algorithm shape.

use butteraugli::{ButteraugliParams, butteraugli};
use butteraugli_gpu::Butteraugli;
use image::ImageReader;
use imgref::ImgVec;
use rgb::{RGB, RGB8};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

fn load_rgb8(path: &str) -> (Vec<u8>, u32, u32) {
    let img = ImageReader::open(path)
        .expect("open png")
        .decode()
        .expect("decode png")
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    (img.into_raw(), w, h)
}

fn perturb_jpeg_like(src: &[u8], width: u32, _height: u32, mag: i32) -> Vec<u8> {
    // 8×8-block-aligned alternating noise — looks like JPEG ringing/blocking.
    let mut out = src.to_vec();
    let w = width as usize;
    let n = src.len() / 3;
    for i in 0..n {
        let x = i % w;
        let y = i / w;
        let bx = x / 8;
        let by = y / 8;
        let stripe = (((bx + by) % 3) as i32 - 1).signum();
        for c in 0..3 {
            let v = src[i * 3 + c] as i32 + stripe * mag;
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
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/lilith/work/imageflow-dotnet-server/examples/hosting_bundle.png".to_string()
    });
    let (ref_rgb, width, height) = load_rgb8(&path);
    println!("Loaded {} ({}×{})", path, width, height);

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let mut gpu_single = Butteraugli::<Backend>::new(client.clone(), width, height);
    let mut gpu_multi = Butteraugli::<Backend>::new_multires(client.clone(), width, height);
    let mut gpu_ref = Butteraugli::<Backend>::new_multires(client, width, height);
    gpu_ref.set_reference(&ref_rgb);

    println!("\n--- single-resolution mode (CPU.with_single_resolution(true)) ---");
    for &mag in &[1_i32, 4, 12, 32] {
        let dist_rgb = perturb_jpeg_like(&ref_rgb, width, height, mag);
        let ref_img = rgb_buf_to_imgvec(&ref_rgb, width, height);
        let dist_img = rgb_buf_to_imgvec(&dist_rgb, width, height);
        let params = ButteraugliParams::default().with_single_resolution(true);
        let cpu = butteraugli(ref_img.as_ref(), dist_img.as_ref(), &params).unwrap();
        let g = gpu_single.compute(&ref_rgb, &dist_rgb);
        let rel = (g.score as f64 - cpu.score) / cpu.score * 100.0;
        let rel_p = (g.pnorm_3 as f64 - cpu.pnorm_3) / cpu.pnorm_3 * 100.0;
        println!(
            " mag={:>2} | CPU score={:.4} pnorm3={:.4} | GPU score={:.4} pnorm3={:.4} | Δ score={:>+5.2}% Δ pnorm3={:>+5.2}%",
            mag, cpu.score, cpu.pnorm_3, g.score, g.pnorm_3, rel, rel_p,
        );
    }

    println!("\n--- multi-resolution mode (CPU default) ---");
    for &mag in &[1_i32, 4, 12, 32] {
        let dist_rgb = perturb_jpeg_like(&ref_rgb, width, height, mag);
        let ref_img = rgb_buf_to_imgvec(&ref_rgb, width, height);
        let dist_img = rgb_buf_to_imgvec(&dist_rgb, width, height);
        let params = ButteraugliParams::default();
        let cpu = butteraugli(ref_img.as_ref(), dist_img.as_ref(), &params).unwrap();
        let g_full = gpu_multi.compute(&ref_rgb, &dist_rgb);
        let g_cached = gpu_ref.compute_with_reference(&dist_rgb);
        let rel = (g_full.score as f64 - cpu.score) / cpu.score * 100.0;
        let rel_p = (g_full.pnorm_3 as f64 - cpu.pnorm_3) / cpu.pnorm_3 * 100.0;
        let cache_drift =
            (g_full.score as f64 - g_cached.score as f64).abs() / (g_full.score as f64).max(1e-9);
        println!(
            " mag={:>2} | CPU score={:.4} | GPU full={:.4} cached_ref={:.4} | Δ score={:>+5.2}% Δ pnorm3={:>+5.2}% | cache drift={:.1e}",
            mag, cpu.score, g_full.score, g_cached.score, rel, rel_p, cache_drift,
        );
    }
}
