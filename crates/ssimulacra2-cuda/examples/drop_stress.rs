//! Minimal repro for the Ssimulacra2 drop panic.
//!
//! Loops over several 2000×1200+ sizes, creating a fresh `Ssimulacra2`
//! instance per size, running N scores, and then dropping it + its
//! backing buffers before moving to the next size. If a drop fails,
//! the `Image::drop` impl currently panics via `unwrap()` on the
//! cudaFreeAsync result, which gives a non-unwinding abort.
//!
//! After the library fix, this should run through all sizes without
//! panicking. Any free-time errors are logged to stderr but do not
//! abort the process.

use std::time::Instant;

use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{C, Image, ImgMut};
use ssimulacra2_cuda::Ssimulacra2;

fn main() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("init cuda");

    // Lowest-priority stream (matches ssim2_only_v3's usage pattern).
    let stream = {
        let range = CuStream::priority_range().expect("priority range");
        CuStream::new_with_priority(range.least).expect("stream")
    };

    // Sizes that grow past the 2000×1200 threshold reported by ssim2_only_v3.
    // Repeated + shuffled to mimic a sorted-by-size walk of ~60-70 sources.
    let base_sizes: &[(u32, u32)] = &[
        (1024, 768),
        (1600, 900),
        (2000, 1200),
        (2048, 1536),
        (1920, 1080),
        (2200, 1400),
        (2400, 1800),
        (2560, 1440),
        (2800, 1900),
        (3000, 2000),
        (3200, 2100),
        (3500, 2333),
        (4000, 3000),
        (4500, 3000),
        (5000, 3500),
    ];
    let mut sizes: Vec<(u32, u32, usize)> = Vec::new();
    // Small-first walk (what ssim2_only_v3 does), multiple passes to
    // stress the stream-ordered pool.
    for _ in 0..5 {
        for &(w, h) in base_sizes {
            sizes.push((w, h, 3));
        }
    }

    for (idx, (w, h, n_scores)) in sizes.iter().copied().enumerate() {
        let px = (w as u64) * (h as u64);
        let t0 = Instant::now();

        // Host RGB buffers for the one ref + (n_scores) distorteds.
        let ref_rgb = make_rgb(w, h, 0);
        let dis_rgbs: Vec<Vec<u8>> = (0..n_scores).map(|s| make_rgb(w, h, s as u8 + 1)).collect();

        // Per-source GPU state matching ssim2_only_v3's lifecycle.
        let mut ref_dev: Image<u8, C<3>> = Image::malloc(w, h).expect("ref_dev malloc");
        let mut dis_dev: Image<u8, C<3>> = Image::malloc(w, h).expect("dis_dev malloc");
        let mut ref_linear: Image<f32, C<3>> = Image::malloc(w, h).expect("ref_linear malloc");
        let mut dis_linear: Image<f32, C<3>> = Image::malloc(w, h).expect("dis_linear malloc");

        let mut ssim = match Ssimulacra2::new(&ref_linear, &dis_linear, &stream) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "[drop_stress] size {idx} ({w}x{h}) ssim new failed ({:?}); skipping",
                    e
                );
                // Drop the caller buffers explicitly so we start the next
                // iteration with maximum free memory.
                drop(ref_linear);
                drop(dis_linear);
                drop(ref_dev);
                drop(dis_dev);
                continue;
            }
        };

        // Populate reference + cache it.
        ref_dev
            .copy_from_cpu(&ref_rgb, stream.inner() as _)
            .expect("ref h2d");
        ssim.kernel()
            .srgb_to_linear(&stream, &ref_dev, &mut ref_linear);
        ssim.set_reference_linear(&ref_linear, &stream)
            .expect("set ref");
        stream.sync().expect("ref prep sync");

        // Score each distorted.
        for (k, dis_rgb) in dis_rgbs.iter().enumerate() {
            dis_dev
                .copy_from_cpu(dis_rgb, stream.inner() as _)
                .expect("dis h2d");
            ssim.kernel()
                .srgb_to_linear(&stream, &dis_dev, &mut dis_linear);
            let score = ssim
                .compute_with_reference_linear(&ref_linear, &dis_linear, &stream)
                .expect("compute");
            eprintln!("  [size {idx}] {w}x{h} ({px} px) dis[{k}] score={score:.3}");
        }

        let score_dt = t0.elapsed().as_secs_f64();

        // Explicit drop dance — mirrors ssim2_only_v3. This is where the
        // panic used to fire around the 7th or 8th source on a clean run.
        drop(ssim);
        drop(ref_linear);
        drop(dis_linear);
        drop(ref_dev);
        drop(dis_dev);

        let total_dt = t0.elapsed().as_secs_f64();
        let free_dt = total_dt - score_dt;
        eprintln!(
            "[drop_stress] size {idx}: {w}x{h} ({px} px) — score_dt={score_dt:.2}s  drop_dt={free_dt:.2}s"
        );
    }

    eprintln!("[drop_stress] completed all sizes without panic");
}

/// Deterministic, non-uniform RGB content so the pipeline sees real
/// gradients (a flat buffer would score trivially).
fn make_rgb(w: u32, h: u32, seed: u8) -> Vec<u8> {
    let n = (w as usize) * (h as usize) * 3;
    let mut out = vec![0u8; n];
    for y in 0..(h as usize) {
        for x in 0..(w as usize) {
            let i = (y * (w as usize) + x) * 3;
            out[i] = ((x.wrapping_add(seed as usize)) as u8).wrapping_mul(3);
            out[i + 1] = ((y.wrapping_add(seed as usize)) as u8).wrapping_mul(5);
            out[i + 2] = ((x ^ y) as u8).wrapping_add(seed).wrapping_mul(7);
        }
    }
    out
}
