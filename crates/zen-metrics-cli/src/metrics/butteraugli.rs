#![forbid(unsafe_code)]

//! CPU butteraugli scoring via the `butteraugli` crate.
//!
//! Returns both aggregations from a single `compute()` call:
//! - **max-norm** (`ButteraugliResult::score`) — the per-block maximum,
//!   sensitive to localised distortion.
//! - **3-norm** (`ButteraugliResult::pnorm_3`) — the libjxl-style
//!   `butteraugli_main --pnorm` aggregation used by the Cloudinary CID22
//!   paper.
//!
//! Both numbers are byproducts of the same internal heatmap, so it is
//! strictly cheaper to emit both than to run butteraugli twice.

use crate::decode::Rgb8Image;

/// Compute butteraugli once and return `(max_norm, pnorm_3)`.
pub fn score_both(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    use rgb::FromSlice;
    let w = reference.width as usize;
    let h = reference.height as usize;

    let ref_pixels = reference.pixels.as_rgb();
    let dst_pixels = distorted.pixels.as_rgb();
    if ref_pixels.len() != w * h || dst_pixels.len() != w * h {
        return Err("butteraugli: pixel count != width*height".into());
    }
    let img1 = imgref::ImgRef::new(ref_pixels, w, h);
    let img2 = imgref::ImgRef::new(dst_pixels, w, h);
    let params = butteraugli::ButteraugliParams::new();
    let result =
        butteraugli::butteraugli(img1, img2, &params).map_err(|e| format!("butteraugli: {e}"))?;
    Ok((result.score as f64, result.pnorm_3))
}
