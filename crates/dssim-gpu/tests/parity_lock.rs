//! Integration tests for `dssim-gpu` against the published `dssim-core`
//! v3.4 CPU reference.
//!
//! Backend selection mirrors `ssim2-gpu/tests/parity_lock.rs`: CUDA
//! preferred, WGPU as the cross-vendor fallback. cubecl-cpu can't run
//! these kernels (no `CUBE_COUNT` builtin, no `Atomic<f32>`); use the
//! per-kernel parity examples or the CUDA / WGPU backends instead.

use cubecl::Runtime;
use dssim_core::{Dssim as DssimCpu, ToRGBAPLU};
use dssim_gpu::{Dssim, Error};
use imgref::ImgVec;
use rgb::RGB;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "dssim-gpu integration tests require either the `cuda` or `wgpu` feature to select a runtime"
);

// Helper kept as a macro because the precise client type changes per
// backend; inlining `Backend::client(&Default::default())` per call is
// the same shape ssim2-gpu uses.
macro_rules! make_client {
    () => {
        Backend::client(&Default::default())
    };
}

// ───────────────────────── helpers ─────────────────────────

fn cpu_dssim(ref_data: &[u8], dis_data: &[u8], w: usize, h: usize) -> f64 {
    let dssim = DssimCpu::new();
    let to_rgb = |buf: &[u8]| -> Vec<RGB<u8>> {
        buf.chunks_exact(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect()
    };
    let ref_rgb = to_rgb(ref_data).to_rgblu();
    let dis_rgb = to_rgb(dis_data).to_rgblu();
    let ref_img = ImgVec::new(ref_rgb, w, h);
    let dis_img = ImgVec::new(dis_rgb, w, h);
    let ref_prep = dssim.create_image(&ref_img).unwrap();
    let dis_prep = dssim.create_image(&dis_img).unwrap();
    let (score, _) = dssim.compare(&ref_prep, dis_prep);
    score.into()
}

fn solid(w: usize, h: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for _ in 0..w * h {
        v.push(r);
        v.push(g);
        v.push(b);
    }
    v
}

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            let g = ((x * 255) / w) as u8;
            v.push(g);
            v.push(g);
            v.push(g);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = Vec::with_capacity(data.len());
    let mut seed = Wrapping(12345_u32);
    for &v in data {
        seed = seed * Wrapping(1103515245_u32) + Wrapping(12345_u32);
        let noise = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        out.push((v as i16 + noise).clamp(0, 255) as u8);
    }
    out
}

// ───────────────────────── parity ─────────────────────────

#[test]
fn identical_image_is_zero() {
    let w = 32;
    let h = 32;
    let img = gradient(w, h);
    let mut d = Dssim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let r = d.compute(&img, &img).unwrap().score;
    eprintln!("identical: gpu = {r:.6}");
    assert!(r < 1e-3, "DSSIM for identical input should be ~0, got {r}");
}

#[test]
fn black_vs_white_is_significant() {
    let w = 64;
    let h = 64;
    let black = solid(w, h, 0, 0, 0);
    let white = solid(w, h, 255, 255, 255);
    let cpu = cpu_dssim(&black, &white, w, h);
    let mut d = Dssim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let gpu = d.compute(&black, &white).unwrap().score;
    eprintln!("black-vs-white: cpu = {cpu:.6}, gpu = {gpu:.6}");
    assert!(cpu > 0.1, "cpu sanity: {cpu}");
    assert!(gpu > 0.1, "gpu produced ~0 on a pair the CPU rates at {cpu}");
    let rel = (gpu - cpu).abs() / cpu;
    assert!(
        rel < 0.05,
        "gpu = {gpu:.6} differs from cpu = {cpu:.6} by {:.2} %",
        rel * 100.0
    );
}

#[test]
fn small_distortion_is_close() {
    let w = 64;
    let h = 64;
    let r = gradient(w, h);
    let d_data = add_noise(&r, 8);
    let cpu = cpu_dssim(&r, &d_data, w, h);
    let mut d = Dssim::<Backend>::new(make_client!(), w as u32, h as u32).unwrap();
    let gpu = d.compute(&r, &d_data).unwrap().score;
    eprintln!("noisy-gradient: cpu = {cpu:.6}, gpu = {gpu:.6}");
    let rel = if cpu > 1e-6 {
        (gpu - cpu).abs() / cpu
    } else {
        (gpu - cpu).abs()
    };
    assert!(
        rel < 0.1,
        "gpu = {gpu:.6} differs from cpu = {cpu:.6} by {:.2} %",
        rel * 100.0
    );
}

#[test]
fn cached_reference_matches_direct() {
    let w = 64_u32;
    let h = 64_u32;
    let r = gradient(w as usize, h as usize);
    let dist = add_noise(&r, 12);

    let mut d_one = Dssim::<Backend>::new(make_client!(), w, h).unwrap();
    let direct = d_one.compute(&r, &dist).unwrap().score;

    let mut d_two = Dssim::<Backend>::new(make_client!(), w, h).unwrap();
    d_two.set_reference(&r).unwrap();
    let cached = d_two.compute_with_reference(&dist).unwrap().score;

    eprintln!("direct = {direct:.6}, cached = {cached:.6}");
    let abs = (direct - cached).abs();
    assert!(
        abs < 1e-5,
        "cached path drifted from direct by {abs:.2e} (direct={direct}, cached={cached})"
    );
}

// ───────────────────────── error paths ─────────────────────────

#[test]
fn dimension_mismatch_is_reported() {
    let mut d = Dssim::<Backend>::new(make_client!(), 32, 32).unwrap();
    let small = vec![0_u8; 31 * 32 * 3];
    let r = d.compute(&small, &small);
    assert!(matches!(r, Err(Error::DimensionMismatch { .. })));
}

#[test]
fn invalid_image_size_rejected() {
    let r = Dssim::<Backend>::new(make_client!(), 4, 4);
    assert!(matches!(r, Err(Error::InvalidImageSize)));
}

#[test]
fn no_cached_reference_rejected() {
    let mut d = Dssim::<Backend>::new(make_client!(), 32, 32).unwrap();
    let buf = vec![0_u8; 32 * 32 * 3];
    let r = d.compute_with_reference(&buf);
    assert!(matches!(r, Err(Error::NoCachedReference)));
}

#[test]
fn clear_reference_drops_cache() {
    let mut d = Dssim::<Backend>::new(make_client!(), 32, 32).unwrap();
    let r = vec![128_u8; 32 * 32 * 3];
    d.set_reference(&r).unwrap();
    assert!(d.has_cached_reference());
    d.clear_reference();
    assert!(!d.has_cached_reference());
}

// ───────────────────────── corpus parity ─────────────────────────

const CORPUS_DIR: &str = "../dssim-cuda/test_data";

fn load_png(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("decode png").to_rgb8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

fn jpeg_to_rgb_at_size(jpeg_path: &std::path::Path, w: u32, h: u32) -> Vec<u8> {
    let img = image::open(jpeg_path).expect("decode jpeg").to_rgb8();
    assert_eq!(img.dimensions(), (w, h), "corpus dim mismatch");
    img.into_raw()
}

#[test]
fn jpeg_corpus_q70_q90() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DIR);
    let src = dir.join("source.png");
    if !src.exists() {
        eprintln!("skipping: corpus dir absent at {}", dir.display());
        return;
    }
    let (ref_data, w, h) = load_png(&src);
    let mut d = Dssim::<Backend>::new(make_client!(), w, h).unwrap();
    for q in &["q70.jpg", "q90.jpg"] {
        let dis_path = dir.join(q);
        let dis_data = jpeg_to_rgb_at_size(&dis_path, w, h);
        let cpu = cpu_dssim(&ref_data, &dis_data, w as usize, h as usize);
        let gpu = d.compute(&ref_data, &dis_data).unwrap().score;
        let rel = if cpu > 1e-6 {
            (gpu - cpu).abs() / cpu
        } else {
            (gpu - cpu).abs()
        };
        eprintln!("{q}: cpu = {cpu:.6}, gpu = {gpu:.6}, rel = {:.3} %", rel * 100.0);
        assert!(
            rel < 0.05,
            "{q}: gpu = {gpu:.6} differs from cpu = {cpu:.6} by {:.2} %",
            rel * 100.0
        );
    }
}
