//! Linear RGB → positive XYB parity vs the CPU `ssimulacra2` /
//! `yuvxyb` pipeline (linear_rgb_to_xyb followed by make_positive_xyb).
//!
//! Tolerance allows ~3 ulp drift from the `cbrtf → powf(_,1/3)`
//! substitution called out in the porting plan.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use ssim2_gpu::kernels::xyb;

const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;
const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69;
const K_M21: f32 = 0.204_767_45;
const K_M22: f32 = 1.0 - K_M20 - K_M21;
const K_B0: f32 = 0.003_793_073_4;
const K_B0_ROOT: f32 = 0.155_954_2;

fn cpu_xyb(r: f32, g: f32, b: f32) -> [f32; 3] {
    let rg = K_M00 * r + K_M01 * g + K_M02 * b + K_B0;
    let gr = K_M10 * r + K_M11 * g + K_M12 * b + K_B0;
    let bb = K_M20 * r + K_M21 * g + K_M22 * b + K_B0;
    let rg_c = rg.max(0.0).cbrt() - K_B0_ROOT;
    let gr_c = gr.max(0.0).cbrt() - K_B0_ROOT;
    let b_c = bb.max(0.0).cbrt() - K_B0_ROOT;
    let x = 0.5 * (rg_c - gr_c);
    let y = 0.5 * (rg_c + gr_c);
    [14.0 * x + 0.42, y + 0.01, b_c - y + 0.55]
}

fn main() {
    let client = CudaRuntime::client(&Default::default());

    // 1024 random-ish samples covering the typical [0, 1] linear-RGB
    // range plus a few near-zero / near-one corners.
    let n = 1024_usize;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut state = 0xC0FFEEu32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32) / (u32::MAX as f32)
    };
    for _ in 0..n {
        r.push(next());
        g.push(next());
        b.push(next());
    }

    let src_r = client.create_from_slice(f32::as_bytes(&r));
    let src_g = client.create_from_slice(f32::as_bytes(&g));
    let src_b = client.create_from_slice(f32::as_bytes(&b));
    let dst_x = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let dst_y = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));
    let dst_bb = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]));

    const TPB: u32 = 256;
    let cubes = ((n as u32) + TPB - 1) / TPB;
    unsafe {
        xyb::linear_to_xyb_planar_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(src_r, n),
            ArrayArg::from_raw_parts(src_g, n),
            ArrayArg::from_raw_parts(src_b, n),
            ArrayArg::from_raw_parts(dst_x.clone(), n),
            ArrayArg::from_raw_parts(dst_y.clone(), n),
            ArrayArg::from_raw_parts(dst_bb.clone(), n),
        );
    }

    let xs = f32::from_bytes(&client.read_one(dst_x).unwrap()).to_vec();
    let ys = f32::from_bytes(&client.read_one(dst_y).unwrap()).to_vec();
    let bs = f32::from_bytes(&client.read_one(dst_bb).unwrap()).to_vec();

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for i in 0..n {
        let exp = cpu_xyb(r[i], g[i], b[i]);
        let got = [xs[i], ys[i], bs[i]];
        for c in 0..3 {
            let a = (got[c] - exp[c]).abs();
            let rel = if exp[c].abs() > 1e-6 { a / exp[c].abs() } else { 0.0 };
            if a > max_abs {
                max_abs = a;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }

    println!("XYB parity over {n} random linear-RGB samples × 3 channels");
    println!("  max abs diff: {max_abs:.3e}");
    println!("  max rel diff: {max_rel:.3e}");
    if max_abs <= 1e-5 {
        println!("  PASS (tolerance 1e-5 abs)");
    } else {
        println!("  FAIL (tolerance 1e-5 abs)");
        std::process::exit(1);
    }
}
