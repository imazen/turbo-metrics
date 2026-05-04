#![forbid(unsafe_code)]

//! CPU butteraugli score via the `butteraugli` crate.
//!
//! Returns the libjxl-style **3-norm** aggregation
//! (`ButteraugliResult::pnorm_3`), matching `butteraugli_main --pnorm` and
//! the Cloudinary CID22 paper. Lower is better.

use crate::decode::Rgb8Image;

pub fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
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
    Ok(result.pnorm_3)
}

pub fn score_max(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    use rgb::FromSlice;
    let w = reference.width as usize;
    let h = reference.height as usize;

    let ref_pixels = reference.pixels.as_rgb();
    let dst_pixels = distorted.pixels.as_rgb();
    if ref_pixels.len() != w * h || dst_pixels.len() != w * h {
        return Err("butteraugli-max: pixel count != width*height".into());
    }
    let img1 = imgref::ImgRef::new(ref_pixels, w, h);
    let img2 = imgref::ImgRef::new(dst_pixels, w, h);
    let params = butteraugli::ButteraugliParams::new();
    let result = butteraugli::butteraugli(img1, img2, &params)
        .map_err(|e| format!("butteraugli-max: {e}"))?;
    Ok(result.score as f64)
}
