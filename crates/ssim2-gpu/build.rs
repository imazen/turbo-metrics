//! Compute the Charalampidis [2016] truncated-cosine recursive Gaussian
//! coefficients at build time and emit them as a Rust constants file.
//!
//! Verbatim port of `crates/ssimulacra2-cuda-kernel/build.rs` (which is
//! itself the same algorithm as the CPU `ssimulacra2` crate's build.rs).
//!
//! Output is `$OUT_DIR/recursive_gaussian.rs`, included by `kernels::blur`.

use nalgebra::{Matrix3, Matrix3x1};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::{env, io};

fn main() {
    let out_dir = env::var("OUT_DIR").expect("can read OUT_DIR");
    init_recursive_gaussian(&out_dir).expect("init recursive gaussian");
}

fn write_const_f32<W: Write>(w: &mut W, name: &str, val: f32) -> io::Result<()> {
    writeln!(w, "pub const {name}: f32 = {val}_f32;")
}

fn write_const_usize<W: Write>(w: &mut W, name: &str, val: usize) -> io::Result<()> {
    writeln!(w, "pub const {name}: usize = {val}_usize;")
}

fn init_recursive_gaussian(out_path: &str) -> io::Result<()> {
    const SIGMA: f64 = 1.5_f64;

    let radius = 3.2795_f64.mul_add(SIGMA, 0.2546).round();

    let pi_div_2r = PI / (2.0 * radius);
    let omega = [pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r];

    let p_1 = 1.0 / (0.5 * omega[0]).tan();
    let p_3 = -1.0 / (0.5 * omega[1]).tan();
    let p_5 = 1.0 / (0.5 * omega[2]).tan();

    let r_1 = p_1 * p_1 / omega[0].sin();
    let r_3 = -p_3 * p_3 / omega[1].sin();
    let r_5 = p_5 * p_5 / omega[2].sin();

    let neg_half_sigma2 = -0.5 * SIGMA * SIGMA;
    let recip_radius = 1.0 / radius;
    let mut rho = [0.0_f64; 3];
    for i in 0..3 {
        rho[i] = (neg_half_sigma2 * omega[i] * omega[i]).exp() * recip_radius;
    }

    let d_13 = p_1.mul_add(r_3, -r_1 * p_3);
    let d_35 = p_3.mul_add(r_5, -r_3 * p_5);
    let d_51 = p_5.mul_add(r_1, -r_5 * p_1);

    let recip_d13 = 1.0 / d_13;
    let zeta_15 = d_35 * recip_d13;
    let zeta_35 = d_51 * recip_d13;

    let a = Matrix3::from_row_slice(&[p_1, p_3, p_5, r_1, r_3, r_5, zeta_15, zeta_35, 1.0])
        .try_inverse()
        .expect("Has inverse");
    let gamma = Matrix3x1::from_column_slice(&[
        1.0,
        radius.mul_add(radius, -SIGMA * SIGMA),
        zeta_15.mul_add(rho[0], zeta_35 * rho[1]) + rho[2],
    ]);
    let beta = a * gamma;

    let sum = beta[2].mul_add(p_5, beta[0].mul_add(p_1, beta[1] * p_3));
    assert!((sum - 1.0).abs() < 1E-12);

    let mut n2 = [0.0_f64; 3];
    let mut d1 = [0.0_f64; 3];
    let mut mul_prev = [0.0_f32; 3 * 4];
    let mut mul_prev2 = [0.0_f32; 3 * 4];
    let mut mul_in = [0.0_f32; 3 * 4];
    for i in 0..3 {
        n2[i] = -beta[i] * (omega[i] * (radius + 1.0)).cos();
        d1[i] = -2.0 * omega[i].cos();

        let d_2 = d1[i] * d1[i];

        mul_prev[4 * i] = -d1[i] as f32;
        mul_prev[4 * i + 1] = (d_2 - 1.0) as f32;
        mul_prev[4 * i + 2] = (-d_2).mul_add(d1[i], 2.0 * d1[i]) as f32;
        mul_prev[4 * i + 3] = d_2.mul_add(d_2, 3.0_f64.mul_add(-d_2, 1.0)) as f32;
        mul_prev2[4 * i] = -1.0;
        mul_prev2[4 * i + 1] = d1[i] as f32;
        mul_prev2[4 * i + 2] = (-d_2 + 1.0) as f32;
        mul_prev2[4 * i + 3] = d_2.mul_add(d1[i], -2.0 * d1[i]) as f32;
        mul_in[4 * i] = n2[i] as f32;
        mul_in[4 * i + 1] = (-d1[i] * n2[i]) as f32;
        mul_in[4 * i + 2] = d_2.mul_add(n2[i], -n2[i]) as f32;
        mul_in[4 * i + 3] = (-d_2 * d1[i]).mul_add(n2[i], 2.0 * d1[i] * n2[i]) as f32;
    }

    let file_path = Path::new(out_path).join("recursive_gaussian.rs");
    let mut out_file = File::create(file_path)?;

    write_const_usize(&mut out_file, "RADIUS", radius as usize)?;

    write_const_f32(&mut out_file, "VERT_MUL_IN_1", n2[0] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_IN_3", n2[1] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_IN_5", n2[2] as f32)?;

    write_const_f32(&mut out_file, "VERT_MUL_PREV_1", d1[0] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_PREV_3", d1[1] as f32)?;
    write_const_f32(&mut out_file, "VERT_MUL_PREV_5", d1[2] as f32)?;

    write_const_f32(&mut out_file, "MUL_IN_1", mul_in[0])?;
    write_const_f32(&mut out_file, "MUL_IN_3", mul_in[4])?;
    write_const_f32(&mut out_file, "MUL_IN_5", mul_in[8])?;

    write_const_f32(&mut out_file, "MUL_PREV_1", mul_prev[0])?;
    write_const_f32(&mut out_file, "MUL_PREV_3", mul_prev[4])?;
    write_const_f32(&mut out_file, "MUL_PREV_5", mul_prev[8])?;

    write_const_f32(&mut out_file, "MUL_PREV2_1", mul_prev2[0])?;
    write_const_f32(&mut out_file, "MUL_PREV2_3", mul_prev2[4])?;
    write_const_f32(&mut out_file, "MUL_PREV2_5", mul_prev2[8])?;

    Ok(())
}
