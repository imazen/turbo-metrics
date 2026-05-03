//! Quick end-to-end smoke: score one synthetic image pair and print
//! the result. Defaults to the CUDA backend; pass
//! `--no-default-features --features wgpu` to run on cross-vendor wgpu.

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use dssim_gpu::Dssim;

fn main() {
    let w = 64_u32;
    let h = 64_u32;
    let mut ref_buf = Vec::with_capacity((w * h * 3) as usize);
    let mut dis_buf = Vec::with_capacity((w * h * 3) as usize);
    for _y in 0..h {
        for x in 0..w {
            let g = ((x * 255) / w) as u8;
            ref_buf.extend_from_slice(&[g, g, g]);
            // distortion: shift left by 1 column.
            let g2 = (((x.saturating_sub(1)) * 255) / w) as u8;
            dis_buf.extend_from_slice(&[g2, g2, g2]);
        }
    }

    let client = Backend::client(&Default::default());
    let mut d = Dssim::<Backend>::new(client, w, h).unwrap();
    let r = d.compute(&ref_buf, &dis_buf).unwrap();
    println!("DSSIM = {:.6}", r.score);

    let r2 = d.compute(&ref_buf, &ref_buf).unwrap();
    println!("self-DSSIM = {:.6}", r2.score);
}
