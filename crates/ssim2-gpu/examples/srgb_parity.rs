//! sRGB → linear parity vs the published `ssimulacra2` LUT (which is the
//! same byte-resolution transfer as `yuvxyb::srgb_gamma_to_lin`).
//!
//! Builds a 256-byte test buffer (one of each value), runs the GPU
//! kernel, and compares against the CPU formula — should be bit-exact
//! since the kernel computes the same expression at byte resolution.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use ssim2_gpu::kernels::srgb;

fn cpu_srgb_to_linear(byte: u8) -> f32 {
    const SRGB_ALPHA: f32 = 1.055_010_7;
    const SRGB_BETA: f32 = 0.003_041_282_5;
    let f = (byte as f32) / 255.0;
    if f < 12.92 * SRGB_BETA {
        f / 12.92
    } else {
        ((f + (SRGB_ALPHA - 1.0)) / SRGB_ALPHA).powf(2.4)
    }
}

fn main() {
    let client = CudaRuntime::client(&Default::default());

    // 16 × 16 = 256 pixels, packed RGB; each pixel's channels are the
    // same byte i for full coverage of the LUT.
    let n_pixels = 256_usize;
    let mut srgb_bytes = Vec::with_capacity(n_pixels * 3);
    for i in 0..256_u32 {
        let v = i as u8;
        srgb_bytes.push(v);
        srgb_bytes.push(v);
        srgb_bytes.push(v);
    }

    let src_handle = client.create_from_slice(&srgb_bytes);
    let dst_r = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_pixels]));
    let dst_g = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_pixels]));
    let dst_b = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_pixels]));

    const TPB: u32 = 256;
    let cubes = ((n_pixels as u32) + TPB - 1) / TPB;
    unsafe {
        srgb::srgb_u8_to_linear_planar_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src_handle, n_pixels * 3),
            ArrayArg::from_raw_parts(dst_r.clone(), n_pixels),
            ArrayArg::from_raw_parts(dst_g.clone(), n_pixels),
            ArrayArg::from_raw_parts(dst_b.clone(), n_pixels),
        );
    }

    let r = f32::from_bytes(&client.read_one(dst_r).unwrap()).to_vec();
    let g = f32::from_bytes(&client.read_one(dst_g).unwrap()).to_vec();
    let b = f32::from_bytes(&client.read_one(dst_b).unwrap()).to_vec();

    let mut max_abs = 0.0_f32;
    let mut bit_exact = 0_usize;
    for i in 0..n_pixels {
        let exp = cpu_srgb_to_linear(i as u8);
        let abs = [(r[i] - exp).abs(), (g[i] - exp).abs(), (b[i] - exp).abs()];
        for a in abs {
            if a == 0.0 {
                bit_exact += 1;
            }
            if a > max_abs {
                max_abs = a;
            }
        }
    }

    println!("sRGB parity over 256 byte values × 3 channels = 768 samples");
    println!("  max |gpu - cpu|: {max_abs:.3e}");
    println!("  bit-exact (==0): {bit_exact} / 768");
    if max_abs <= 1e-6 {
        println!("  PASS (tolerance 1e-6)");
    } else {
        println!("  FAIL (tolerance 1e-6)");
        std::process::exit(1);
    }
}
