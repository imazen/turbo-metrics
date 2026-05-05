#![forbid(unsafe_code)]

//! Codec encode dispatch for the sweep driver.
//!
//! Each codec gets its own `encode_*` function that takes the source RGB8
//! image, the integer quality, and the per-cell knob tuple JSON, and
//! returns the encoded bytes. The functions exist as a thin orchestration
//! layer — they translate JSON into typed builder calls and never modify
//! codec source.
//!
//! Knob coverage per codec (zen-metrics-cli 0.3.0):
//!
//! ## zenwebp 0.4.5 (`__expert` enabled)
//! Public builders: `method`, `segments`, `sns_strength`, `filter_strength`,
//! `lossless`. Expert (via `LossyConfig::with_internal_params`):
//! `partition_limit`, `multi_pass_stats`, `smooth_segment_map`, `sharp_yuv`
//! (`"off"` / `"on"`).
//!
//! ## zenavif 0.1.7 (`__expert` enabled)
//! Public builders: `speed` (0..=10), `lossless`. Expert (via
//! `EncoderConfig::with_internal_params`): `partition_range` (`[min, max]`
//! pair, members in {4, 8, 16, 32, 64}), `complex_prediction_modes`,
//! `lrf`, `fast_deblock`.
//!
//! ## zenjxl 0.2.1 (`__expert` enabled, but `JxlEncoderConfig` does not
//! currently expose `with_internal_params` for the lossy path — internal
//! knobs live on `jxl_encoder::LossyConfig` and are not reachable from
//! the wrapper). The wrapper handles: `effort` (1..=10), `lossless`,
//! `noise`, `distance`. For lossy, the sweep also drops down to
//! `jxl_encoder::LossyConfig` directly when any expert knob is present:
//! `butteraugli_iters`, `zensim_iters`, `ssim2_iters`, `pixel_domain_loss`,
//! `patches`, `gaborish`, `error_diffusion`, `denoise`, `lf_frame`,
//! `force_strategy` (u8 or null), `progressive` ("off"/"dc-only"/"two-pass").
//! These bypass the macro-knob `effort` bundling so the picker can compose
//! independent decisions (e.g. effort=5 DCT search + 2 butteraugli iters
//! without LZ77).
//!
//! Adding a codec or knob is a local change to this file: extend the
//! match arms, document it in the module header, and add a test under
//! `tests/cli.rs` if the encoder behaviour is observable.

use crate::decode::Rgb8Image;
use serde_json::{Map, Value};
use std::error::Error;
use std::time::Instant;

/// Codec selector for the sweep CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CodecKind {
    /// `zenwebp` lossy/lossless WebP encoder.
    Zenwebp,
    /// `zenavif` AV1-still encoder via ravif.
    Zenavif,
    /// `zenjxl` JPEG XL encoder via jxl-encoder.
    Zenjxl,
}

impl CodecKind {
    pub fn name(self) -> &'static str {
        match self {
            CodecKind::Zenwebp => "zenwebp",
            CodecKind::Zenavif => "zenavif",
            CodecKind::Zenjxl => "zenjxl",
        }
    }
}

/// Encoded output bundle.
pub struct EncodedCell {
    pub bytes: Vec<u8>,
    pub encode_ms: f64,
}

/// Encode `source` with `codec`, quality `q`, and knob assignment `knobs`.
/// Errors propagate as `Box<dyn Error>` so the sweep runner can record a
/// per-cell failure without halting the rest of the grid.
pub fn encode(
    codec: CodecKind,
    source: &Rgb8Image,
    q: u32,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    match codec {
        CodecKind::Zenwebp => encode_webp(source, q, knobs),
        CodecKind::Zenavif => encode_avif(source, q, knobs),
        CodecKind::Zenjxl => encode_jxl(source, q, knobs),
    }
}

// ── zenwebp ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "sweep", feature = "webp"))]
fn encode_webp(
    source: &Rgb8Image,
    q: u32,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout};

    let lossless = knobs
        .get("lossless")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let cfg = if lossless {
        EncoderConfig::new_lossless().with_quality(q as f32)
    } else {
        let mut lossy = LossyConfig::new().with_quality(q as f32);
        if let Some(m) = knobs.get("method").and_then(Value::as_u64) {
            lossy = lossy.with_method(m.min(6) as u8);
        }
        if let Some(s) = knobs.get("segments").and_then(Value::as_u64) {
            lossy = lossy.with_segments(s.clamp(1, 4) as u8);
        }
        if let Some(s) = knobs.get("sns_strength").and_then(Value::as_u64) {
            lossy = lossy.with_sns_strength(s.min(100) as u8);
        }
        if let Some(s) = knobs.get("filter_strength").and_then(Value::as_u64) {
            lossy = lossy.with_filter_strength(s.min(100) as u8);
        }
        // Expert knobs — only built when at least one is present, so the
        // default codepath is exercised as-is when all knobs are absent.
        #[cfg(feature = "sweep")]
        {
            let mut params = zenwebp::InternalParams::default();
            let mut any = false;
            if let Some(v) = knobs.get("partition_limit").and_then(Value::as_u64) {
                params.partition_limit = Some(v.min(100) as u8);
                any = true;
            }
            if let Some(v) = knobs.get("multi_pass_stats").and_then(Value::as_bool) {
                params.multi_pass_stats = Some(v);
                any = true;
            }
            if let Some(v) = knobs.get("smooth_segment_map").and_then(Value::as_bool) {
                params.smooth_segment_map = Some(v);
                any = true;
            }
            if let Some(v) = knobs.get("sharp_yuv").and_then(Value::as_str) {
                params.sharp_yuv = Some(match v {
                    "off" => zenwebp::SharpYuvSetting::Off,
                    "on" => zenwebp::SharpYuvSetting::On,
                    other => {
                        return Err(format!(
                            "zenwebp sharp_yuv must be \"off\" or \"on\"; got {other:?}"
                        )
                        .into());
                    }
                });
                any = true;
            }
            if any {
                lossy = lossy.with_internal_params(params);
            }
        }
        EncoderConfig::Lossy(lossy)
    };

    let start = Instant::now();
    let bytes = EncodeRequest::new(
        &cfg,
        &source.pixels,
        PixelLayout::Rgb8,
        source.width,
        source.height,
    )
    .encode()
    .map_err(|e| format!("zenwebp encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell { bytes, encode_ms })
}

#[cfg(not(all(feature = "sweep", feature = "webp")))]
fn encode_webp(
    _source: &Rgb8Image,
    _q: u32,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenwebp encode is disabled (rebuild with `--features sweep`)".into())
}

// ── zenavif ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "sweep", feature = "avif"))]
fn encode_avif(
    source: &Rgb8Image,
    q: u32,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use imgref::ImgRef;
    use zenavif::EncoderConfig;

    // Build an ImgRef<Rgb<u8>> over the source buffer without copying.
    let pixels: &[rgb::Rgb<u8>] = bytemuck_cast_rgb(&source.pixels);
    let img = ImgRef::new(pixels, source.width as usize, source.height as usize);

    let mut cfg = EncoderConfig::new().quality(q as f32);
    if let Some(s) = knobs.get("speed").and_then(Value::as_u64) {
        cfg = cfg.speed(s.min(10) as u8);
    }
    if let Some(b) = knobs.get("lossless").and_then(Value::as_bool) {
        cfg = cfg.with_lossless(b);
    }
    // Expert knobs.
    #[cfg(feature = "sweep")]
    {
        let mut params = zenavif::expert::InternalParams::default();
        let mut any = false;
        if let Some(arr) = knobs.get("partition_range").and_then(Value::as_array)
            && arr.len() == 2
        {
            let min = arr[0].as_u64().unwrap_or(0) as u8;
            let max = arr[1].as_u64().unwrap_or(0) as u8;
            params.partition_range = Some((min, max));
            any = true;
        }
        if let Some(v) = knobs
            .get("complex_prediction_modes")
            .and_then(Value::as_bool)
        {
            params.complex_prediction_modes = Some(v);
            any = true;
        }
        if let Some(v) = knobs.get("lrf").and_then(Value::as_bool) {
            params.lrf = Some(v);
            any = true;
        }
        if let Some(v) = knobs.get("fast_deblock").and_then(Value::as_bool) {
            params.fast_deblock = Some(v);
            any = true;
        }
        if any {
            cfg = cfg.with_internal_params(params);
        }
    }

    let start = Instant::now();
    let encoded = zenavif::encode_rgb8(
        img,
        &cfg,
        almost_enough::StopToken::new(enough::Unstoppable),
    )
    .map_err(|e| format!("zenavif encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell {
        bytes: encoded.avif_file,
        encode_ms,
    })
}

#[cfg(not(all(feature = "sweep", feature = "avif")))]
fn encode_avif(
    _source: &Rgb8Image,
    _q: u32,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenavif encode is disabled (rebuild with `--features sweep`)".into())
}

// ── zenjxl ──────────────────────────────────────────────────────────────

/// Expert lossy knobs that, if any are present, route encode through
/// `jxl_encoder::LossyConfig` directly instead of the zenjxl wrapper.
#[cfg(all(feature = "sweep", feature = "jxl"))]
const JXL_EXPERT_KNOBS: &[&str] = &[
    "butteraugli_iters",
    "zensim_iters",
    "ssim2_iters",
    "pixel_domain_loss",
    "patches",
    "gaborish",
    "error_diffusion",
    "denoise",
    "lf_frame",
    "force_strategy",
    "max_strategy_size",
    "progressive",
    "lz77",
];

#[cfg(all(feature = "sweep", feature = "jxl"))]
fn encode_jxl(
    source: &Rgb8Image,
    q: u32,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    let lossless = knobs
        .get("lossless")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // Expert lossy path: any of the new knobs trigger a direct LossyConfig
    // build that bypasses the wrapper's effort macro-knob. Lossless stays
    // on the wrapper; the new knobs don't apply to modular mode.
    if !lossless && knobs.keys().any(|k| JXL_EXPERT_KNOBS.contains(&k.as_str())) {
        return encode_jxl_expert(source, q, knobs);
    }

    encode_jxl_wrapper(source, q, knobs, lossless)
}

#[cfg(all(feature = "sweep", feature = "jxl"))]
fn encode_jxl_wrapper(
    source: &Rgb8Image,
    q: u32,
    knobs: &Map<String, Value>,
    lossless: bool,
) -> Result<EncodedCell, Box<dyn Error>> {
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    use zenjxl::JxlEncoderConfig;
    use zenpixels::{PixelDescriptor, PixelSlice};

    let mut cfg = if lossless {
        JxlEncoderConfig::new().with_lossless(true)
    } else {
        let mut c = JxlEncoderConfig::new().with_generic_quality(q as f32);
        if let Some(d) = knobs.get("distance").and_then(Value::as_f64) {
            c = c.with_distance(d as f32);
        }
        if let Some(b) = knobs.get("noise").and_then(Value::as_bool) {
            c = c.with_noise(b);
        }
        c
    };
    if let Some(e) = knobs.get("effort").and_then(Value::as_u64) {
        cfg = cfg.with_generic_effort(e.clamp(1, 10) as i32);
    }

    let stride = (source.width as usize) * 3;
    let slice = PixelSlice::new(
        &source.pixels,
        source.width,
        source.height,
        stride,
        PixelDescriptor::RGB8_SRGB,
    )
    .map_err(|e| format!("zenjxl: pixel slice construction failed: {e}"))?;

    let start = Instant::now();
    let encoder = cfg
        .job()
        .encoder()
        .map_err(|e| format!("zenjxl encoder construction failed: {e}"))?;
    let output = encoder
        .encode(slice)
        .map_err(|e| format!("zenjxl encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell {
        bytes: output.into_vec(),
        encode_ms,
    })
}

/// Direct `LossyConfig` path for expert knobs the wrapper doesn't expose.
/// Decouples the macro-knob `effort` from independent decisions like
/// `butteraugli_iters`, `pixel_domain_loss`, `patches`, etc.
#[cfg(all(feature = "sweep", feature = "jxl"))]
fn encode_jxl_expert(
    source: &Rgb8Image,
    q: u32,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use jxl_encoder::{LossyConfig, PixelLayout, ProgressiveMode};

    // Resolve distance: explicit `distance` knob wins, else fall back to
    // the generic q→distance mapping zenjxl uses.
    let distance = if let Some(d) = knobs.get("distance").and_then(Value::as_f64) {
        d as f32
    } else {
        jxl_encoder::quality_to_distance(jxl_encoder::calibrated_jxl_quality(q as f32))
    };

    let mut cfg = LossyConfig::new(distance);

    if let Some(e) = knobs.get("effort").and_then(Value::as_u64) {
        cfg = cfg.with_effort(e.clamp(1, 10) as u8);
    }
    if let Some(b) = knobs.get("noise").and_then(Value::as_bool) {
        cfg = cfg.with_noise(b);
    }
    if let Some(b) = knobs.get("denoise").and_then(Value::as_bool) {
        cfg = cfg.with_denoise(b);
    }
    if let Some(b) = knobs.get("gaborish").and_then(Value::as_bool) {
        cfg = cfg.with_gaborish(b);
    }
    if let Some(b) = knobs.get("patches").and_then(Value::as_bool) {
        cfg = cfg.with_patches(b);
    }
    if let Some(b) = knobs.get("pixel_domain_loss").and_then(Value::as_bool) {
        cfg = cfg.with_pixel_domain_loss(b);
    }
    if let Some(b) = knobs.get("error_diffusion").and_then(Value::as_bool) {
        cfg = cfg.with_error_diffusion(b);
    }
    if let Some(b) = knobs.get("lf_frame").and_then(Value::as_bool) {
        cfg = cfg.with_lf_frame(b);
    }
    if let Some(b) = knobs.get("lz77").and_then(Value::as_bool) {
        cfg = cfg.with_lz77(b);
    }
    if let Some(n) = knobs.get("butteraugli_iters").and_then(Value::as_u64) {
        cfg = cfg.with_butteraugli_iters(n.min(16) as u32);
    }
    if let Some(n) = knobs.get("zensim_iters").and_then(Value::as_u64) {
        cfg = cfg.with_zensim_iters(n.min(16) as u32);
    }
    if let Some(n) = knobs.get("ssim2_iters").and_then(Value::as_u64) {
        cfg = cfg.with_ssim2_iters(n.min(16) as u32);
    }
    // `force_strategy` accepts null (unset) or u8 in 0..=18 (DCT strategy id).
    if let Some(v) = knobs.get("force_strategy") {
        match v {
            Value::Null => cfg = cfg.with_force_strategy(None),
            Value::Number(n) => {
                if let Some(s) = n.as_u64() {
                    cfg = cfg.with_force_strategy(Some(s.min(18) as u8));
                }
            }
            _ => {}
        }
    }
    if let Some(v) = knobs.get("max_strategy_size") {
        match v {
            Value::Null => cfg = cfg.with_max_strategy_size(None),
            Value::Number(n) => {
                if let Some(s) = n.as_u64() {
                    cfg = cfg.with_max_strategy_size(Some(s.min(255) as u8));
                }
            }
            _ => {}
        }
    }
    if let Some(s) = knobs.get("progressive").and_then(Value::as_str) {
        let mode = match s {
            "off" | "single" => ProgressiveMode::Single,
            "two-pass" | "two_pass" | "twopass" | "qac-fac" => ProgressiveMode::QuantizedAcFullAc,
            "three-pass" | "three_pass" | "threepass" | "dc-vlf-lf-ac" => {
                ProgressiveMode::DcVlfLfAc
            }
            other => {
                return Err(format!(
                    "zenjxl: unknown progressive mode '{other}' (want single|two-pass|three-pass)"
                )
                .into());
            }
        };
        cfg = cfg.with_progressive(mode);
    }

    let stride = (source.width as usize) * 3;
    if source.pixels.len() < stride * source.height as usize {
        return Err("zenjxl expert: pixel buffer shorter than width*height*3".into());
    }

    let start = Instant::now();
    let bytes = cfg
        .encode(
            &source.pixels,
            source.width,
            source.height,
            PixelLayout::Rgb8,
        )
        .map_err(|e| format!("zenjxl expert encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell { bytes, encode_ms })
}

#[cfg(not(all(feature = "sweep", feature = "jxl")))]
fn encode_jxl(
    _source: &Rgb8Image,
    _q: u32,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenjxl encode is disabled (rebuild with `--features sweep`)".into())
}

// ── helpers ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "sweep", feature = "avif"))]
fn bytemuck_cast_rgb(bytes: &[u8]) -> &[rgb::Rgb<u8>] {
    // `rgb::Rgb<u8>` is `repr(C)` over three `u8` fields with no padding,
    // so a flat RGB byte buffer with length divisible by 3 maps 1:1 onto
    // a slice of `Rgb<u8>`. We use the rgb crate's own
    // `FromSlice::as_rgb` which is a safe shim that performs the cast
    // via its own `bytemuck`-style guard internally — keeps our crate
    // `#![forbid(unsafe_code)]` clean.
    use rgb::FromSlice;
    bytes.as_rgb()
}
