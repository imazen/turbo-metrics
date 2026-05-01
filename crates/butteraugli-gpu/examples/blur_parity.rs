//! Validate the separable blur kernels (horizontal + vertical) against
//! a CPU reference matching the existing butteraugli-cuda implementation.

use butteraugli_gpu::kernels::blur::{horizontal_blur_kernel, vertical_blur_kernel};
use cubecl::prelude::*;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

const M: f32 = 2.25;

fn cpu_gauss(d: f32, s: f32) -> f32 {
    let z = d / s;
    (-0.5 * z * z).exp()
}

fn cpu_horizontal_blur(src: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = ((M * sigma) as usize).max(1);
    let mut dst = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let begin = x.saturating_sub(radius);
            let end = (x + radius).min(width - 1);
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for i in begin..=end {
                let dist = (i as i32 - x as i32).unsigned_abs() as f32;
                let w = cpu_gauss(dist, sigma);
                sum += src[y * width + i] * w;
                wsum += w;
            }
            dst[y * width + x] = sum / wsum;
        }
    }
    dst
}

fn cpu_vertical_blur(src: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = ((M * sigma) as usize).max(1);
    let mut dst = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let begin = y.saturating_sub(radius);
            let end = (y + radius).min(height - 1);
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for i in begin..=end {
                let dist = (i as i32 - y as i32).unsigned_abs() as f32;
                let w = cpu_gauss(dist, sigma);
                sum += src[i * width + x] * w;
                wsum += w;
            }
            dst[y * width + x] = sum / wsum;
        }
    }
    dst
}

fn run_pass(label: &str, sigma: f32) {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);
    let width: usize = 64;
    let height: usize = 64;
    let n = width * height;

    let src: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % width) as f32;
            let y = (i / width) as f32;
            (x * 0.13).sin() + (y * 0.07).cos()
        })
        .collect();

    let src_handle = client.create_from_slice(f32::as_bytes(&src));
    let h_handle = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let v_handle = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    const TPB: u32 = 256;
    let cubes = ((n as u32) + TPB - 1) / TPB;

    unsafe {
        horizontal_blur_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src_handle.clone(), n),
            ArrayArg::from_raw_parts(h_handle.clone(), n),
            width as u32,
            height as u32,
            sigma,
        );
        vertical_blur_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(h_handle.clone(), n),
            ArrayArg::from_raw_parts(v_handle.clone(), n),
            width as u32,
            height as u32,
            sigma,
        );
    }

    let h_bytes = client.read_one(h_handle).expect("h");
    let v_bytes = client.read_one(v_handle).expect("v");
    let h_gpu = f32::from_bytes(&h_bytes);
    let v_gpu = f32::from_bytes(&v_bytes);

    let h_cpu = cpu_horizontal_blur(&src, width, height, sigma);
    let v_cpu = cpu_vertical_blur(&h_cpu, width, height, sigma);

    let mut max_h = 0.0f32;
    let mut max_v = 0.0f32;
    for i in 0..n {
        max_h = max_h.max((h_gpu[i] - h_cpu[i]).abs());
        max_v = max_v.max((v_gpu[i] - v_cpu[i]).abs());
    }
    println!("[{label} sigma={sigma}] H abs diff = {max_h:.2e}  HV abs diff = {max_v:.2e}");
}

fn main() {
    // The five sigmas butteraugli actually uses.
    run_pass("init", 1.2);
    run_pass("hf_sep", 1.564_163_3);
    run_pass("mask", 2.7);
    run_pass("mf_sep", 3.224_899);
    run_pass("lf_sep", 7.155_933_4);
}
