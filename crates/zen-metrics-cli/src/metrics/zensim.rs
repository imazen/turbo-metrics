#![forbid(unsafe_code)]

//! CPU zensim score via the `zensim` crate.

use crate::decode::Rgb8Image;

pub fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    use zensim::{PixelFormat, StridedBytes, Zensim};

    let z = Zensim::new(zensim::ZensimProfile::latest());
    let w = reference.width as usize;
    let h = reference.height as usize;
    let stride = w * 3;

    let src = StridedBytes::try_new(&reference.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid reference RGB slice: {e:?}"))?;
    let dst = StridedBytes::try_new(&distorted.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid distorted RGB slice: {e:?}"))?;
    let result = z
        .compute(&src, &dst)
        .map_err(|e| format!("zensim: {e:?}"))?;
    Ok(result.score())
}

/// Score plus the full extended 300-feature vector (4 scales × 3 channels ×
/// 25 features/channel at the default profile).
///
/// The `score` returned here is identical to [`score`] — the extra 72
/// masked features have zero weight in the trained profile, so adding
/// them changes neither the weighted distance nor the score. The extra
/// cost on top of [`score`] is the masking pass over the same multi-scale
/// stats; the score-relevant work is shared.
///
/// Only used by the `sweep` subcommand when the user passes
/// `--feature-output <path.parquet>`.
#[cfg(feature = "sweep")]
pub fn score_with_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    use zensim::{PixelFormat, StridedBytes, Zensim};

    let z = Zensim::new(zensim::ZensimProfile::latest());
    let w = reference.width as usize;
    let h = reference.height as usize;
    let stride = w * 3;

    let src = StridedBytes::try_new(&reference.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid reference RGB slice: {e:?}"))?;
    let dst = StridedBytes::try_new(&distorted.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid distorted RGB slice: {e:?}"))?;
    let result = z
        .compute_extended_features(&src, &dst)
        .map_err(|e| format!("zensim: {e:?}"))?;
    let score = result.score();
    let features = result.into_features();
    Ok((score, features))
}
