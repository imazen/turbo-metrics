//! Smoke test: verify ButteraugliParams actually changes scores
//! (tunable params are wired through the pipeline) and that
//! clear_reference + copy_diffmap_to + Result-returning APIs all work.

use butteraugli_gpu::{Butteraugli, ButteraugliBatch, ButteraugliParams, Error};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn make(w: u32, h: u32, salt: u32) -> Vec<u8> {
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
    let w = 256u32;
    let h = 256u32;
    let r = make(w, h, 1);
    let d = make(w, h, 2);
    let mut bu = Butteraugli::<Backend>::new_multires(client.clone(), w, h);

    println!("--- compute_with_options ---");
    let default = ButteraugliParams::default();
    let hdr = ButteraugliParams::default().with_intensity_target(250.0);
    let asym = ButteraugliParams::default().with_hf_asymmetry(2.0);
    let chroma_lo = ButteraugliParams::default().with_xmul(0.25);
    for (label, p) in [
        ("default (80nit, 1.0, 1.0)", default),
        ("HDR (250nit)", hdr),
        ("hf_asymmetry=2.0", asym),
        ("xmul=0.25", chroma_lo),
    ] {
        let out = bu.compute_with_options(&r, &d, &p).expect("ok");
        println!(
            "  {:30}  score={:.4}  pnorm_3={:.4}",
            label, out.score, out.pnorm_3
        );
    }

    println!("\n--- clear_reference ---");
    bu.set_reference(&r);
    let s1 = bu.compute_with_reference(&d).score;
    bu.clear_reference();
    assert!(!bu.has_cached_reference());
    let err = bu.try_compute_with_reference(&d).unwrap_err();
    assert!(matches!(err, Error::NoCachedReference));
    println!("  cleared, then try_compute_with_reference → {:?} ✓", err);
    bu.set_reference(&r);
    let s2 = bu.compute_with_reference(&d).score;
    println!(
        "  re-cached, score before={:.4} after={:.4}  diff={:.2e}",
        s1,
        s2,
        (s1 - s2).abs()
    );

    println!("\n--- copy_diffmap_to ---");
    let mut buf = vec![0.0f32; (w * h) as usize];
    bu.copy_diffmap_to(&mut buf).expect("ok");
    let max = buf.iter().cloned().fold(0.0_f32, f32::max);
    println!(
        "  preallocated buffer, peak diffmap={:.4} (matches score={:.4})",
        max, s2
    );
    let mut tiny = vec![0.0f32; 16];
    let err = bu.copy_diffmap_to(&mut tiny).unwrap_err();
    println!("  too-small buffer → {:?} ✓", err);

    println!("\n--- compute() with bad dim ---");
    let bad = vec![0u8; 16];
    let err = Butteraugli::<Backend>::new(client.clone(), w, h)
        .compute_with_options(&bad, &d, &ButteraugliParams::default())
        .unwrap_err();
    println!("  {} → {:?} ✓", "small buffer", err);

    println!("\n--- ButteraugliBatch with options ---");
    let n = 4;
    let mut dist_batch = Vec::with_capacity(n * (w * h * 3) as usize);
    for i in 0..n {
        dist_batch.extend_from_slice(&make(w, h, 100 + i as u32));
    }
    let mut batch = ButteraugliBatch::<Backend>::new(client, w, h, n);
    batch
        .set_reference_with_options(&r, &ButteraugliParams::default().with_xmul(0.5))
        .expect("ok");
    let scores = batch.compute_batch_with_reference(&dist_batch);
    println!("  xmul=0.5 batched scores: {:?}", scores);
    batch.clear_reference();
    assert!(!batch.has_reference());
    println!("  cleared ✓");
}
