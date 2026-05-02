//! End-to-end smoke test: run `Ssim2::compute` on a synthetic image
//! pair and print the score. Builds a 256×256 reference (gradient + a
//! few stripes), and three distorted variants at increasing magnitude.
//!
//! No parity check here — see `parity_real_image` for that. This is the
//! "did the pipeline run end-to-end without panicking" gate.

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::Ssim2;

fn build_pair(width: u32, height: u32, mag: u8) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let mut a = vec![0u8; w * h * 3];
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255 / w.max(1)) & 0xff) as u8;
            let g = ((y * 255 / h.max(1)) & 0xff) as u8;
            let bb = (((x + y) * 255 / (w + h).max(1)) & 0xff) as u8;
            let i = (y * w + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;
            // Distorted: 8×8 block-aligned ring + magnitude offset.
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 { mag as i32 } else { -(mag as i32) };
            b[i] = (r as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (g as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (bb as i32 + pert).clamp(0, 255) as u8;
        }
    }
    (a, b)
}

fn main() {
    let client = CudaRuntime::client(&Default::default());

    let width = 256_u32;
    let height = 256_u32;

    let mut s = Ssim2::<CudaRuntime>::new(client, width, height).expect("Ssim2::new");

    for mag in [0u8, 1, 4, 12, 32] {
        let (a, b) = build_pair(width, height, mag);
        let result = s.compute(&a, &b).expect("compute");
        println!("mag = {mag:>2}: ssim2 score = {:.4}", result.score);
    }
}
