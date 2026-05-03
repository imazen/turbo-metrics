#![forbid(unsafe_code)]

//! CPU SSIMULACRA2 via the `ssimulacra2` crate. The metric expects
//! `LinearRgb`; we go through the re-exported `Rgb` so we use the same
//! `yuvxyb` version `ssimulacra2` was built against.

use crate::decode::Rgb8Image;

pub fn score(reference: &Rgb8Image, distorted: &Rgb8Image) -> Result<f64, Box<dyn std::error::Error>> {
    let lin_ref = to_linear_rgb(reference)?;
    let lin_dst = to_linear_rgb(distorted)?;
    let s = ssimulacra2::compute_frame_ssimulacra2(lin_ref, lin_dst)
        .map_err(|e| format!("ssimulacra2: {e:?}"))?;
    Ok(s)
}

fn to_linear_rgb(img: &Rgb8Image) -> Result<ssimulacra2::LinearRgb, Box<dyn std::error::Error>> {
    if img.width == 0 || img.height == 0 {
        return Err("ssimulacra2: image has zero dimension".into());
    }
    let w = img.width as usize;
    let h = img.height as usize;
    let mut data = Vec::with_capacity(img.pixels.len() / 3);
    for chunk in img.pixels.chunks_exact(3) {
        data.push([
            chunk[0] as f32 / 255.0,
            chunk[1] as f32 / 255.0,
            chunk[2] as f32 / 255.0,
        ]);
    }
    let rgb = ssimulacra2::Rgb::new(
        data,
        w,
        h,
        ssimulacra2::TransferCharacteristic::SRGB,
        ssimulacra2::ColorPrimaries::BT709,
    )
    .map_err(|e| format!("ssimulacra2::Rgb::new: {e:?}"))?;
    ssimulacra2::LinearRgb::try_from(rgb)
        .map_err(|e| format!("ssimulacra2::LinearRgb::try_from: {e:?}").into())
}
