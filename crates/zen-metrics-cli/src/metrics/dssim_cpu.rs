#![forbid(unsafe_code)]

//! CPU DSSIM score via the `dssim-core` crate.
//!
//! DSSIM is in dissimilarity space — higher = more different. The CLI
//! reports the raw `Val::into()` f64 like `dssim-core`'s own examples.

use crate::decode::Rgb8Image;

pub fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    use rgb::FromSlice;

    let w = reference.width as usize;
    let h = reference.height as usize;

    let attr = dssim_core::Dssim::new();
    let ref_img = attr
        .create_image_rgb(reference.pixels.as_rgb(), w, h)
        .ok_or("dssim: failed to create reference image")?;
    let dst_img = attr
        .create_image_rgb(distorted.pixels.as_rgb(), w, h)
        .ok_or("dssim: failed to create distorted image")?;
    let (val, _maps) = attr.compare(&ref_img, dst_img);
    Ok(val.into())
}
