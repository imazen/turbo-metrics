#![forbid(unsafe_code)]

//! CPU DSSIM score via the `dssim-core` crate (canonical Rust DSSIM,
//! Kornel's `dssim` v3.4).
//!
//! DSSIM is a "distance" metric — `0.0` for identical images, larger
//! values mean more distortion. Output here mirrors `dssim_core::Dssim::compare`'s
//! returned `Val` cast to `f64`.

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
        return Err("dssim: pixel count != width*height".into());
    }
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err("dssim: reference and distorted dimensions differ".into());
    }

    let d = dssim_core::Dssim::new();
    let ref_img = d
        .create_image_rgb(ref_pixels, w, h)
        .ok_or("dssim: failed to build reference image (zero size?)")?;
    let dist_img = d
        .create_image_rgb(dst_pixels, w, h)
        .ok_or("dssim: failed to build distorted image (zero size?)")?;
    let (val, _maps) = d.compare(&ref_img, dist_img);
    Ok(f64::from(val))
}
