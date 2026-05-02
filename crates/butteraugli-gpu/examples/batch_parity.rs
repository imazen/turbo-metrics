//! Verify ButteraugliBatch produces the same scores as N sequential
//! Butteraugli::compute_with_reference calls. Then time it for the
//! amortised launch-overhead benefit at small sizes.

use std::time::Instant;

use butteraugli_gpu::{Butteraugli, ButteraugliBatch};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

fn make_image(w: u32, h: u32, salt: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            out.push(((x.wrapping_mul(7).wrapping_add(salt)) & 0xff) as u8);
            out.push(((y.wrapping_mul(11).wrapping_add(salt.wrapping_mul(3))) & 0xff) as u8);
            out.push((((x ^ y).wrapping_mul(13).wrapping_add(salt.wrapping_mul(5))) & 0xff) as u8);
        }
    }
    out
}

fn main() {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // Sanity: identical images → all-zero batched scores.
    {
        let w = 256u32;
        let h = 256u32;
        let n = 4;
        let bytes_per = (w * h * 3) as usize;
        let ref_rgb = make_image(w, h, 1);
        let mut dist_batch = Vec::with_capacity(n * bytes_per);
        for _ in 0..n {
            dist_batch.extend_from_slice(&ref_rgb);
        }
        let mut bu_batch = ButteraugliBatch::<Backend>::new(client.clone(), w, h, n);
        bu_batch.set_reference(&ref_rgb);
        let scores = bu_batch.compute_batch_with_reference(&dist_batch);
        println!("identical-image sanity, n={}: {:?}", n, scores);

        // Same with single-image cached path for comparison.
        let mut bu_single = Butteraugli::<Backend>::new_multires(client.clone(), w, h);
        bu_single.set_reference(&ref_rgb);
        let r = bu_single.compute_with_reference(&ref_rgb);
        println!(
            "identical-image, single cached: score={}, pnorm_3={}",
            r.score, r.pnorm_3
        );
    }
    // Sanity 2: flat-128 input. All freq bands should be ≈ 0.
    {
        let w = 256u32;
        let h = 256u32;
        let n = 2;
        let flat = vec![128_u8; (w * h * 3) as usize];
        let mut dist_batch = Vec::with_capacity(n * (w * h * 3) as usize);
        for _ in 0..n {
            dist_batch.extend_from_slice(&flat);
        }
        let mut bu_batch = ButteraugliBatch::<Backend>::new(client.clone(), w, h, n);
        bu_batch.set_reference(&flat);
        let scores = bu_batch.compute_batch_with_reference(&dist_batch);
        println!("flat-128 sanity, n={}: {:?}", n, scores);
        let mut bu_single = Butteraugli::<Backend>::new_multires(client.clone(), w, h);
        bu_single.set_reference(&flat);
        let r = bu_single.compute_with_reference(&flat);
        println!(
            "flat-128 single cached: score={}, pnorm_3={}",
            r.score, r.pnorm_3
        );
    }

    let cases: &[(u32, u32, usize)] = &[(256, 256, 8), (512, 512, 8), (1024, 1024, 4)];

    for &(w, h, n) in cases {
        let bytes_per = (w * h * 3) as usize;
        let ref_rgb = make_image(w, h, 0xC0FFEE);
        let mut dist_batch = Vec::with_capacity(n * bytes_per);
        for i in 0..n {
            let img = make_image(w, h, 0x100 + i as u32);
            dist_batch.extend_from_slice(&img);
        }

        // Reference: per-image cached_with_reference.
        let mut bu_single = Butteraugli::<Backend>::new_multires(client.clone(), w, h);
        bu_single.set_reference(&ref_rgb);
        let mut single_scores = Vec::with_capacity(n);
        let t = Instant::now();
        for i in 0..n {
            let dist = &dist_batch[i * bytes_per..(i + 1) * bytes_per];
            let r = bu_single.compute_with_reference(dist);
            single_scores.push(r.score);
        }
        let single_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Batched
        let mut bu_batch = ButteraugliBatch::<Backend>::new(client.clone(), w, h, n);
        bu_batch.set_reference(&ref_rgb);
        // warm-up
        let _ = bu_batch.compute_batch_with_reference(&dist_batch);
        let t = Instant::now();
        let batched_scores = bu_batch.compute_batch_with_reference(&dist_batch);
        let batch_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("\n=== {}×{}, batch n={} ===", w, h, n);
        let mut max_rel = 0.0f64;
        let mut max_abs = 0.0f64;
        for (i, (&s, &b)) in single_scores.iter().zip(batched_scores.iter()).enumerate() {
            let abs = (s as f64 - b as f64).abs();
            let rel = if s.abs() > 1e-6 {
                abs / s.abs() as f64
            } else {
                0.0
            };
            if rel > max_rel {
                max_rel = rel;
            }
            if abs > max_abs {
                max_abs = abs;
            }
            println!(
                "  [{i}] single={:.4}  batched={:.4}  Δ={:.2e}  (rel={:.2e})",
                s, b, abs, rel
            );
        }
        println!(
            "  max abs diff = {:.4e}  max rel diff = {:.4e}",
            max_abs, max_rel
        );
        println!(
            "  time:   N×compute_with_reference = {:.2} ms  ({:.2} ms/image)",
            single_ms,
            single_ms / n as f64
        );
        println!(
            "  time:   compute_batch_with_reference = {:.2} ms  ({:.2} ms/image)",
            batch_ms,
            batch_ms / n as f64
        );
        println!(
            "  speedup: {:.2}× per image",
            (single_ms / n as f64) / (batch_ms / n as f64)
        );
    }
}
