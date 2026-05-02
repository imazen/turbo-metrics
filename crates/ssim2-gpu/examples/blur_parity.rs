//! Recursive Gaussian blur parity vs the published `ssimulacra2`
//! crate's `Blur::blur` (the reference path). The CPU does
//! `horizontal_pass(src) → temp; vertical_pass(temp) → dst`. We do
//! `vpass(src) → vt; transpose(vt) → tv; vpass(tv) → out` (output stays
//! in transposed coordinates), which is what the CUDA `process_scale`
//! also does — same algebraic operation, parity should be sub-1e-4.
//!
//! The IIR is non-trivial; this is the load-bearing parity gate.

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
use cubecl::prelude::*;
use ssim2_gpu::kernels::{blur, transpose};

fn cpu_blur(src: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut blur = ssimulacra2::Blur::new(width, height);
    let r = src.to_vec();
    let g = src.to_vec();
    let b = src.to_vec();
    let result = blur.blur(&[r, g, b]);
    result[0].clone()
}

fn run_case(width: u32, height: u32, name: &str, src: &[f32]) -> bool {
    let client = Backend::client(&Default::default());
    let n = (width as usize) * (height as usize);
    assert_eq!(src.len(), n);

    let src_h = client.create_from_slice(f32::as_bytes(src));
    let v_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let t_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let out_h = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    const TPB_1D: u32 = 256;
    let cubes_1d = ((n as u32) + TPB_1D - 1) / TPB_1D;
    let blur_cubes_w = (width + blur::BLOCK_WIDTH - 1) / blur::BLOCK_WIDTH;
    let blur_cubes_h = (height + blur::BLOCK_WIDTH - 1) / blur::BLOCK_WIDTH;

    unsafe {
        // 1. v-pass on src.
        blur::blur_pass_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(blur_cubes_w.max(1), 1, 1),
            CubeDim::new_1d(blur::BLOCK_WIDTH),
            ArrayArg::from_raw_parts(src_h, n),
            ArrayArg::from_raw_parts(v_h.clone(), n),
            width,
            height,
        );
        // 2. transpose v_h → t_h.
        transpose::transpose_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(cubes_1d.max(1), 1, 1),
            CubeDim::new_1d(TPB_1D),
            ArrayArg::from_raw_parts(v_h.clone(), n),
            ArrayArg::from_raw_parts(t_h.clone(), n),
            width,
            height,
        );
        // 3. v-pass on transposed.
        blur::blur_pass_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(blur_cubes_h.max(1), 1, 1),
            CubeDim::new_1d(blur::BLOCK_WIDTH),
            ArrayArg::from_raw_parts(t_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            height,
            width,
        );
    }

    let out = f32::from_bytes(&client.read_one(out_h).unwrap()).to_vec();
    // GPU output is in transposed orientation (height × width). Transpose
    // back so we can compare elementwise to the CPU output.
    let mut un_t = vec![0.0_f32; n];
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            un_t[y * w + x] = out[x * h + y];
        }
    }

    let cpu = cpu_blur(src, w, h);

    let mut max_abs = 0.0_f32;
    let mut max_idx = 0;
    for i in 0..n {
        let d = (un_t[i] - cpu[i]).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }
    let pass = max_abs <= 1e-4;
    println!(
        "[{}] {}×{}: max |gpu - cpu| = {:.3e} at idx {}  {}",
        if pass { "ok " } else { "FAIL" },
        width,
        height,
        max_abs,
        max_idx,
        name
    );
    pass
}

fn build(width: u32, height: u32, mut f: impl FnMut(usize, usize) -> f32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut v = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            v[y * w + x] = f(x, y);
        }
    }
    v
}

fn main() {
    let mut all_pass = true;

    all_pass &= run_case(32, 32, "constant 0.5", &build(32, 32, |_, _| 0.5));
    all_pass &= run_case(
        32,
        32,
        "delta at (8, 12)",
        &build(32, 32, |x, y| if x == 8 && y == 12 { 1.0 } else { 0.0 }),
    );
    all_pass &= run_case(64, 32, "x-gradient", &build(64, 32, |x, _| x as f32 / 63.0));
    all_pass &= run_case(32, 64, "y-gradient", &build(32, 64, |_, y| y as f32 / 63.0));

    let mut state = 0x5EEDu32;
    let img256 = build(256, 256, |_, _| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32) / (u32::MAX as f32)
    });
    all_pass &= run_case(256, 256, "random 256x256", &img256);

    let mut state2 = 0xCAFEu32;
    let img1k = build(1024, 768, |_, _| {
        state2 ^= state2 << 13;
        state2 ^= state2 >> 17;
        state2 ^= state2 << 5;
        (state2 as f32) / (u32::MAX as f32)
    });
    all_pass &= run_case(1024, 768, "random 1024x768", &img1k);

    if !all_pass {
        std::process::exit(1);
    }
    println!("All blur parity cases pass (tolerance 1e-4 abs)");
}
