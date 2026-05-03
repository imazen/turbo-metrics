#![forbid(unsafe_code)]

//! CPU butteraugli score via the `butteraugli` crate.
//!
//! Returns the max-norm score (`ButteraugliResult::score`). Lower is better:
//! < 1.0 is "good", > 2.0 is "bad". Mirrors libjxl's default scoring.

use crate::decode::Rgb8Image;

pub fn score(reference: &Rgb8Image, distorted: &Rgb8Image) -> Result<f64, Box<dyn std::error::Error>> {
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
    Ok(result.score)
}
