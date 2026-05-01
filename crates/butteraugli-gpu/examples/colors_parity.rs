//! Validate the colors kernels (sRGB→linear, opsin dynamics) against a
//! CPU reference. Runs whatever backend the dev features select.

use butteraugli_gpu::kernels::colors::{
    opsin_dynamics_planar_kernel, srgb_u8_to_linear_planar_kernel,
};
use cubecl::prelude::*;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

// CPU reference implementations — same formulas as the kernel.
const OPSIN_BIAS_X: f32 = 1.7557483643287353;
const OPSIN_BIAS_Y: f32 = 1.7557483643287353;
const OPSIN_BIAS_B: f32 = 12.226454707163354;
const GAMMA_MUL: f32 = 19.245013259874995;
const GAMMA_ADD: f32 = 9.971063576929914;
const GAMMA_SUB: f32 = 23.16046239805755;

fn cpu_srgb_to_linear(v: u8) -> f32 {
    let f = v as f32 / 255.0;
    if f <= 0.04045 {
        f / 12.92
    } else {
        ((f + 0.055) / 1.055).powf(2.4)
    }
}

fn cpu_gamma(v: f32) -> f32 {
    GAMMA_MUL * (v + GAMMA_ADD).ln() - GAMMA_SUB
}

fn cpu_opsin_absorbance(r: f32, g: f32, b: f32, clamp: bool) -> (f32, f32, f32) {
    let mut x = 0.299_565_5_f32 * r + 0.633_730_9_f32 * g + 0.077_705_62_f32 * b + OPSIN_BIAS_X;
    let mut y = 0.221_586_91_f32 * r + 0.693_913_88_f32 * g + 0.098_731_36_f32 * b + OPSIN_BIAS_Y;
    let mut z = 0.02 * r + 0.02 * g + 0.204_801_29_f32 * b + OPSIN_BIAS_B;
    if clamp {
        x = x.max(OPSIN_BIAS_X);
        y = y.max(OPSIN_BIAS_Y);
        z = z.max(OPSIN_BIAS_B);
    }
    (x, y, z)
}

fn cpu_opsin_dynamics(
    src: &mut [(f32, f32, f32)],
    blur: &[(f32, f32, f32)],
    intensity_multiplier: f32,
) {
    for (px, &(br, bg, bb)) in src.iter_mut().zip(blur.iter()) {
        let r = px.0 * intensity_multiplier;
        let g = px.1 * intensity_multiplier;
        let b = px.2 * intensity_multiplier;
        let br = br * intensity_multiplier;
        let bg = bg * intensity_multiplier;
        let bb = bb * intensity_multiplier;

        let (bx, by, bz) = cpu_opsin_absorbance(br, bg, bb, true);
        let bx = bx.max(1e-4);
        let by = by.max(1e-4);
        let bz = bz.max(1e-4);
        let sens_x = (cpu_gamma(bx) / bx).max(1e-4);
        let sens_y = (cpu_gamma(by) / by).max(1e-4);
        let sens_z = (cpu_gamma(bz) / bz).max(1e-4);

        let (mut sx, mut sy, mut sz) = cpu_opsin_absorbance(r, g, b, false);
        sx *= sens_x;
        sy *= sens_y;
        sz *= sens_z;
        sx = sx.max(OPSIN_BIAS_X);
        sy = sy.max(OPSIN_BIAS_Y);
        sz = sz.max(OPSIN_BIAS_B);

        *px = (sx - sy, sx + sy, sz);
    }
}

fn main() {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 1024 pixels of synthetic sRGB data — coverage of dark, mid, bright.
    let n_pixels: usize = 1024;
    let src: Vec<u8> = (0..n_pixels * 3)
        .map(|i| ((i.wrapping_mul(31) + 7) % 256) as u8)
        .collect();

    // GPU sRGB → linear planar
    let src_handle = client.create_from_slice(&src);
    let r_handle = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_pixels]));
    let g_handle = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_pixels]));
    let b_handle = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_pixels]));

    const TPB: u32 = 256;
    let cubes = ((n_pixels as u32) + TPB - 1) / TPB;
    unsafe {
        srgb_u8_to_linear_planar_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src_handle.clone(), src.len()),
            ArrayArg::from_raw_parts(r_handle.clone(), n_pixels),
            ArrayArg::from_raw_parts(g_handle.clone(), n_pixels),
            ArrayArg::from_raw_parts(b_handle.clone(), n_pixels),
        );
    }

    let r_bytes = client.read_one(r_handle.clone()).expect("r");
    let g_bytes = client.read_one(g_handle.clone()).expect("g");
    let b_bytes = client.read_one(b_handle.clone()).expect("b");
    let r_gpu = f32::from_bytes(&r_bytes);
    let g_gpu = f32::from_bytes(&g_bytes);
    let b_gpu = f32::from_bytes(&b_bytes);

    let mut max_srgb_diff = 0.0_f32;
    for i in 0..n_pixels {
        let r_cpu = cpu_srgb_to_linear(src[i * 3]);
        let g_cpu = cpu_srgb_to_linear(src[i * 3 + 1]);
        let b_cpu = cpu_srgb_to_linear(src[i * 3 + 2]);
        max_srgb_diff = max_srgb_diff
            .max((r_gpu[i] - r_cpu).abs())
            .max((g_gpu[i] - g_cpu).abs())
            .max((b_gpu[i] - b_cpu).abs());
    }
    println!("[srgb_to_linear] max abs diff over {n_pixels} pixels: {max_srgb_diff:.2e}");

    // 2) Opsin dynamics: feed linear-RGB result and a "blurred" (=copy) version,
    //    then compare to CPU opsin-dynamics output.
    let blur_r = client.create_from_slice(f32::as_bytes(r_gpu));
    let blur_g = client.create_from_slice(f32::as_bytes(g_gpu));
    let blur_b = client.create_from_slice(f32::as_bytes(b_gpu));

    // Re-upload writable copies of r/g/b so we don't poison the read-only
    // handles above.
    let work_r = client.create_from_slice(f32::as_bytes(r_gpu));
    let work_g = client.create_from_slice(f32::as_bytes(g_gpu));
    let work_b = client.create_from_slice(f32::as_bytes(b_gpu));

    let intensity_multiplier = 80.0_f32 / 255.0_f32;

    unsafe {
        opsin_dynamics_planar_kernel::launch_unchecked::<Backend>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(work_r.clone(), n_pixels),
            ArrayArg::from_raw_parts(work_g.clone(), n_pixels),
            ArrayArg::from_raw_parts(work_b.clone(), n_pixels),
            ArrayArg::from_raw_parts(blur_r, n_pixels),
            ArrayArg::from_raw_parts(blur_g, n_pixels),
            ArrayArg::from_raw_parts(blur_b, n_pixels),
            intensity_multiplier,
        );
    }

    let xb = client.read_one(work_r).expect("x");
    let yb = client.read_one(work_g).expect("y");
    let bb = client.read_one(work_b).expect("b");
    let x_gpu = f32::from_bytes(&xb);
    let y_gpu = f32::from_bytes(&yb);
    let b_gpu_xyb = f32::from_bytes(&bb);

    // CPU reference: pre-build (r,g,b) tuples from the linear values we
    // already computed, then apply opsin dynamics with blur == self.
    let mut cpu_pixels: Vec<(f32, f32, f32)> = (0..n_pixels)
        .map(|i| {
            (
                cpu_srgb_to_linear(src[i * 3]),
                cpu_srgb_to_linear(src[i * 3 + 1]),
                cpu_srgb_to_linear(src[i * 3 + 2]),
            )
        })
        .collect();
    let cpu_blur = cpu_pixels.clone();
    cpu_opsin_dynamics(&mut cpu_pixels, &cpu_blur, intensity_multiplier);

    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;
    let mut max_b = 0.0f32;
    for i in 0..n_pixels {
        max_x = max_x.max((x_gpu[i] - cpu_pixels[i].0).abs());
        max_y = max_y.max((y_gpu[i] - cpu_pixels[i].1).abs());
        max_b = max_b.max((b_gpu_xyb[i] - cpu_pixels[i].2).abs());
    }
    println!("[opsin_dynamics] max abs diff X={max_x:.2e}  Y={max_y:.2e}  B={max_b:.2e}");
}
