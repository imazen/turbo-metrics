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

    let mut gpu = Butteraugli::<Backend>::new(client, width, height);

    for &mag in &[1_i32, 4, 12, 32] {
        let dist_rgb = perturb_jpeg_like(&ref_rgb, width, height, mag);
        let ref_img = rgb_buf_to_imgvec(&ref_rgb, width, height);
        let dist_img = rgb_buf_to_imgvec(&dist_rgb, width, height);

        let params = ButteraugliParams::default().with_single_resolution(true);
        let cpu =
            butteraugli(ref_img.as_ref(), dist_img.as_ref(), &params).expect("cpu butteraugli");

        let g = gpu.compute(&ref_rgb, &dist_rgb);

        let cpu_score = cpu.score;
        let cpu_pnorm = cpu.pnorm_3;
        let rel_score = (g.score as f64 - cpu_score) / cpu_score * 100.0;
        let rel_pnorm = (g.pnorm_3 as f64 - cpu_pnorm) / cpu_pnorm * 100.0;
        println!(
            "{w}×{h} mag={:>2} | CPU score={:.4} pnorm3={:.4} | GPU score={:.4} pnorm3={:.4} | Δ score={:>+5.1}% Δ pnorm3={:>+5.1}%",
            mag,
            cpu_score,
            cpu_pnorm,
            g.score,
            g.pnorm_3,
            rel_score,
            rel_pnorm,
            w = width,
            h = height,
        );
    }
}
