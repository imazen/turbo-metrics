//! DSSIM pipeline orchestration.
//!
//! Wires the kernels in `kernels::*` into the 5-scale DSSIM algorithm.
//! Public entry points:
//!
//! - [`Dssim::new`] + [`Dssim::compute`] — score one image pair from sRGB.
//! - [`Dssim::set_reference`] + [`Dssim::compute_with_reference`] — cache
//!   reference-side state and score many distorted images against it.
//!
//! Algorithm (per scale, faithful to `dssim-cuda`'s `compute_sync_inner`):
//!   1. Linear-RGB pyramid: sRGB→linear at scale 0, then 2×2 box
//!      downscale planar between scales.
//!   2. Per scale: linear RGB → planar Lab.
//!   3. Pre-blur the chroma (a, b) twice each — matches `dssim-core`'s
//!      chroma blur step.
//!   4. For every channel ∈ {L, a, b}:
//!      - mu = blur(blur(channel))
//!      - sq_blur = blur(blur_squared(channel))
//!      - cross = blur(blur_product(ref_channel, dis_channel))
//!   5. Fused 15-input SSIM map.
//!   6. Σ ssim → mean_ssim; avg = mean_ssim ^ (0.5 ^ scale_idx);
//!      Σ |ssim - avg| → mad; score_for_scale = 1 - mad.
//!   7. Final = Σ (score_for_scale · weight) / Σ weight, then
//!      `1 / ssim - 1` to convert SSIM → DSSIM.
//!
//! Buffer aliasing follows the `blur_plane_via(src, dst, scratch)` rule
//! from `CUBECL_GOTCHAS.md` — `temp1` is the only mutable scratch and
//! it's never the source or dest of the second pass.

use cubecl::prelude::*;

use crate::kernels::{blur, downscale, lab, reduction, srgb, ssim};
use crate::{Error, GpuDssimResult, NUM_SCALES, Result, SCALE_WEIGHTS};

/// Per-scale buffer set. All planes are `width × height` planar f32.
/// We keep `temp1` and `temp2` separate so the two-pass blur sequence
/// (`blur → blur` for mu, `blur_squared → blur` for variance,
/// `blur_product → blur` for covariance) never aliases its source and
/// destination through the same scratch.
struct Scale {
    width: u32,
    height: u32,
    n: usize,

    /// Linear RGB at this scale (planar).
    ref_lin: [cubecl::server::Handle; 3],
    dis_lin: [cubecl::server::Handle; 3],

    /// Lab planes (post conversion + chroma pre-blur).
    ref_lab: [cubecl::server::Handle; 3],
    dis_lab: [cubecl::server::Handle; 3],

    /// Cached reference-side fully-blurred outputs (used by
    /// `compute_with_reference`).
    ref_mu: [cubecl::server::Handle; 3],
    ref_sq_blur: [cubecl::server::Handle; 3],

    /// Distorted-side and cross outputs (overwritten per call).
    dis_mu: [cubecl::server::Handle; 3],
    dis_sq_blur: [cubecl::server::Handle; 3],
    cross_blur: [cubecl::server::Handle; 3],

    /// Two scratch planes for the two-pass-blur shape.
    temp1: cubecl::server::Handle,
    temp2: cubecl::server::Handle,

    /// Per-pixel SSIM map (also reused as input for `abs_diff_scalar`).
    ssim_map: cubecl::server::Handle,
    /// Per-pixel |ssim - avg| map (input to the second reduction).
    mad_map: cubecl::server::Handle,
}

fn alloc_plane<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

fn alloc_3<R: Runtime>(client: &ComputeClient<R>, n: usize) -> [cubecl::server::Handle; 3] {
    [
        alloc_plane(client, n),
        alloc_plane(client, n),
        alloc_plane(client, n),
    ]
}

impl Scale {
    fn new<R: Runtime>(client: &ComputeClient<R>, width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            n,
            ref_lin: alloc_3(client, n),
            dis_lin: alloc_3(client, n),
            ref_lab: alloc_3(client, n),
            dis_lab: alloc_3(client, n),
            ref_mu: alloc_3(client, n),
            ref_sq_blur: alloc_3(client, n),
            dis_mu: alloc_3(client, n),
            dis_sq_blur: alloc_3(client, n),
            cross_blur: alloc_3(client, n),
            temp1: alloc_plane(client, n),
            temp2: alloc_plane(client, n),
            ssim_map: alloc_plane(client, n),
            mad_map: alloc_plane(client, n),
        }
    }
}

/// Per-instance allocations + per-call orchestration of the DSSIM
/// pipeline. Construct once for a given resolution; reuse across many
/// image pairs of that resolution.
pub struct Dssim<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    n: usize,

    /// sRGB u8 staging (uploaded as u32 because WGSL has no `u8` type).
    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    /// Per-scale buffer sets.
    scales: Vec<Scale>,

    /// Per-thread (or per-slot, in fast mode) reduction partials.
    /// Layout: 2 reductions per scale × `NUM_SCALES` scales.
    /// Slot encoding: `scale * 2 + (0 = ssim_sum, 1 = mad_sum)`.
    partials: cubecl::server::Handle,
    /// Final per-slot scalars folded by the finalizer kernel.
    sums: cubecl::server::Handle,

    has_cached_reference: bool,
}

const NUM_SLOTS: usize = NUM_SCALES * 2; // 10
const PARTIALS_LEN: usize = NUM_SLOTS * reduction::PARTIALS_PER_REDUCTION;
const SUMS_LEN: usize = NUM_SLOTS;

impl<R: Runtime> Dssim<R> {
    /// Allocate every per-instance buffer for the given image size.
    /// Returns `Err(InvalidImageSize)` for images smaller than 8×8.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize);
        }
        let n = (width as usize) * (height as usize);

        // Pyramid dims: 5 levels, halving each axis. dssim-cuda clamps
        // each axis to a minimum of 8 to keep the SSIM window valid;
        // we mirror that.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut w = width;
        let mut h = height;
        for _ in 0..NUM_SCALES {
            dims.push((w, h));
            w = (w + 1) / 2;
            h = (h + 1) / 2;
            if w < 8 {
                w = 8;
            }
            if h < 8 {
                h = 8;
            }
        }

        let scales = dims
            .iter()
            .map(|&(w, h)| Scale::new(&client, w, h))
            .collect::<Vec<_>>();

        let n_bytes = n * 3;
        let src_u8_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n_bytes]));
        let src_u8_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n_bytes]));

        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        let sums = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; SUMS_LEN]));

        Ok(Self {
            client,
            width,
            height,
            n,
            src_u8_a,
            src_u8_b,
            scales,
            partials,
            sums,
            has_cached_reference: false,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Number of active pyramid scales.
    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// Score one image pair, both sRGB packed RGB u8 of length
    /// `width × height × 3`.
    pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<GpuDssimResult> {
        self.check_dims(ref_srgb)?;
        self.check_dims(dist_srgb)?;

        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.upload_and_srgb_to_linear(false, dist_srgb);

        // Build linear pyramid first (dssim-core downsamples in linear
        // RGB, NOT in Lab — Lab is per-scale).
        self.build_linear_pyramid(true);
        self.build_linear_pyramid(false);

        self.zero_partials();

        for s in 0..self.scales.len() {
            self.run_lab(s, true);
            self.run_lab(s, false);
            self.run_chroma_preblur(s, true);
            self.run_chroma_preblur(s, false);
            self.run_blur_stats(s, true);
            self.run_blur_stats(s, false);
            self.run_cross_blur(s);
            self.run_ssim_map(s);
            self.run_sum_ssim(s);
        }
        self.run_finalize();

        // Read sums → compute mean_ssim per scale → re-launch
        // |ssim - avg| → second reduction → final score.
        let sums_host = self.read_sums();
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let ssim_sum = sums_host[s * 2] as f64;
            let mean_ssim = ssim_sum / n_pix;
            let avg = mean_ssim.max(0.0).powf(0.5_f64.powi(s as i32));
            self.run_abs_diff_and_sum(s, avg as f32);
        }
        self.run_finalize();
        let sums_host_pass2 = self.read_sums();

        let mut weighted = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let mad_sum = sums_host_pass2[s * 2 + 1] as f64;
            let mad = mad_sum / n_pix;
            let scale_score = 1.0 - mad;
            weighted += scale_score * SCALE_WEIGHTS[s];
            weight_sum += SCALE_WEIGHTS[s];
        }
        let ssim = weighted / weight_sum;
        let dssim = ssim_to_dssim(ssim);

        Ok(GpuDssimResult { score: dssim })
    }

    /// Cache reference-side state for many comparisons against a fixed
    /// reference. Subsequent `compute_with_reference` calls skip the
    /// reference-side pyramid + Lab + reference-blur work.
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_dims(ref_srgb)?;
        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.build_linear_pyramid(true);
        for s in 0..self.scales.len() {
            self.run_lab(s, true);
            self.run_chroma_preblur(s, true);
            self.run_blur_stats(s, true);
        }
        self.has_cached_reference = true;
        Ok(())
    }

    /// Drop any cached reference state.
    pub fn clear_reference(&mut self) {
        self.has_cached_reference = false;
    }

    pub fn has_cached_reference(&self) -> bool {
        self.has_cached_reference
    }

    /// Compute against the cached reference. Returns
    /// `Err(NoCachedReference)` if `set_reference` hasn't been called.
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<GpuDssimResult> {
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        self.check_dims(dist_srgb)?;

        self.upload_and_srgb_to_linear(false, dist_srgb);
        self.build_linear_pyramid(false);

        self.zero_partials();

        for s in 0..self.scales.len() {
            self.run_lab(s, false);
            self.run_chroma_preblur(s, false);
            self.run_blur_stats(s, false);
            self.run_cross_blur(s);
            self.run_ssim_map(s);
            self.run_sum_ssim(s);
        }
        self.run_finalize();

        let sums_host = self.read_sums();
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let ssim_sum = sums_host[s * 2] as f64;
            let mean_ssim = ssim_sum / n_pix;
            let avg = mean_ssim.max(0.0).powf(0.5_f64.powi(s as i32));
            self.run_abs_diff_and_sum(s, avg as f32);
        }
        self.run_finalize();
        let sums_host_pass2 = self.read_sums();

        let mut weighted = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let mad_sum = sums_host_pass2[s * 2 + 1] as f64;
            let mad = mad_sum / n_pix;
            let scale_score = 1.0 - mad;
            weighted += scale_score * SCALE_WEIGHTS[s];
            weight_sum += SCALE_WEIGHTS[s];
        }
        let ssim = weighted / weight_sum;
        Ok(GpuDssimResult {
            score: ssim_to_dssim(ssim),
        })
    }

    // ───────────────────────── helpers ─────────────────────────

    fn check_dims(&self, srgb: &[u8]) -> Result<()> {
        let expected = self.n * 3;
        if srgb.len() != expected {
            Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            })
        } else {
            Ok(())
        }
    }

    fn cube_count_1d(n: usize) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = ((n as u32) + TPB - 1) / TPB;
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn cube_dim_1d() -> CubeDim {
        CubeDim::new_1d(256)
    }

    fn upload_and_srgb_to_linear(&mut self, is_a: bool, srgb: &[u8]) {
        let n_bytes = self.n * 3;
        let widened: Vec<u32> = srgb.iter().map(|&b| b as u32).collect();
        if is_a {
            self.src_u8_a = self.client.create_from_slice(u32::as_bytes(&widened));
        } else {
            self.src_u8_b = self.client.create_from_slice(u32::as_bytes(&widened));
        }
        let (src, lin) = if is_a {
            (&self.src_u8_a, &self.scales[0].ref_lin)
        } else {
            (&self.src_u8_b, &self.scales[0].dis_lin)
        };
        unsafe {
            srgb::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(self.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), n_bytes),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
            );
        }
    }

    fn build_linear_pyramid(&self, is_a: bool) {
        for s in 1..self.scales.len() {
            let (prev_w, prev_h) = (self.scales[s - 1].width, self.scales[s - 1].height);
            let (curr_w, curr_h) = (self.scales[s].width, self.scales[s].height);
            let (prev_lin, curr_lin) = if is_a {
                (&self.scales[s - 1].ref_lin, &self.scales[s].ref_lin)
            } else {
                (&self.scales[s - 1].dis_lin, &self.scales[s].dis_lin)
            };
            let n_curr = self.scales[s].n;
            let n_prev = self.scales[s - 1].n;
            for ch in 0..3 {
                unsafe {
                    downscale::downscale_2x_plane_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n_curr),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(prev_lin[ch].clone(), n_prev),
                        ArrayArg::from_raw_parts(curr_lin[ch].clone(), n_curr),
                        prev_w,
                        prev_h,
                        curr_w,
                        curr_h,
                    );
                }
            }
        }
    }

    fn run_lab(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let (lin, lab_buf) = if is_a {
            (&s.ref_lin, &s.ref_lab)
        } else {
            (&s.dis_lin, &s.dis_lab)
        };
        unsafe {
            lab::linear_to_lab_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), s.n),
                ArrayArg::from_raw_parts(lin[1].clone(), s.n),
                ArrayArg::from_raw_parts(lin[2].clone(), s.n),
                ArrayArg::from_raw_parts(lab_buf[0].clone(), s.n),
                ArrayArg::from_raw_parts(lab_buf[1].clone(), s.n),
                ArrayArg::from_raw_parts(lab_buf[2].clone(), s.n),
            );
        }
    }

    /// Two-pass `blur_3x3` of the chroma channels (a, b) in place via
    /// scratch. Matches `dssim-cuda`'s pre-SSIM chroma blur step.
    fn run_chroma_preblur(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let lab_buf = if is_a { &s.ref_lab } else { &s.dis_lab };
        for ch in [1usize, 2] {
            self.blur_two_pass(scale, &lab_buf[ch], &lab_buf[ch], &s.temp1, &s.temp2);
        }
    }

    /// Two-pass blur via `src → temp_a → temp_b` then copy `temp_b → dst`.
    /// `dst` and `src` may be the same handle. Aliasing rule: the two
    /// scratches must be distinct from each other AND from src/dst.
    fn blur_two_pass(
        &self,
        scale: usize,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        scratch_a: &cubecl::server::Handle,
        scratch_b: &cubecl::server::Handle,
    ) {
        let s = &self.scales[scale];
        unsafe {
            // pass 1: src → scratch_a
            blur::blur_3x3_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), s.n),
                ArrayArg::from_raw_parts(scratch_a.clone(), s.n),
                s.width,
                s.height,
            );
            // pass 2: scratch_a → scratch_b
            blur::blur_3x3_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch_a.clone(), s.n),
                ArrayArg::from_raw_parts(scratch_b.clone(), s.n),
                s.width,
                s.height,
            );
            // copy scratch_b → dst (allows dst == src). Use the
            // `blur_3x3` kernel here would corrupt the result; use a
            // pointwise copy via the abs_diff_scalar(0.0) hack? No —
            // use an explicit copy kernel.
            copy_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch_b.clone(), s.n),
                ArrayArg::from_raw_parts(dst.clone(), s.n),
            );
        }
    }

    /// For each channel: `mu = blur(blur(src))`,
    /// `sq_blur = blur(blur_squared(src))`. Stores into `ref_mu` /
    /// `ref_sq_blur` (or dis variants).
    fn run_blur_stats(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let (src_lab, mu_dst, sq_dst) = if is_a {
            (&s.ref_lab, &s.ref_mu, &s.ref_sq_blur)
        } else {
            (&s.dis_lab, &s.dis_mu, &s.dis_sq_blur)
        };
        for ch in 0..3 {
            // mu pipeline.
            self.blur_two_pass(scale, &src_lab[ch], &mu_dst[ch], &s.temp1, &s.temp2);

            // sq_blur pipeline: blur_squared(src) → temp1, then blur → sq_dst.
            unsafe {
                blur::blur_squared_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(src_lab[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    s.width,
                    s.height,
                );
                blur::blur_3x3_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    ArrayArg::from_raw_parts(sq_dst[ch].clone(), s.n),
                    s.width,
                    s.height,
                );
            }
        }
    }

    /// `cross_blur[ch] = blur(blur_product(ref_lab[ch], dis_lab[ch]))`.
    fn run_cross_blur(&self, scale: usize) {
        let s = &self.scales[scale];
        for ch in 0..3 {
            unsafe {
                blur::blur_product_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.ref_lab[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.dis_lab[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    s.width,
                    s.height,
                );
                blur::blur_3x3_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    ArrayArg::from_raw_parts(s.cross_blur[ch].clone(), s.n),
                    s.width,
                    s.height,
                );
            }
        }
    }

    fn run_ssim_map(&self, scale: usize) {
        let s = &self.scales[scale];
        unsafe {
            ssim::ssim_lab_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(s.ref_mu[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_mu[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_mu[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_mu[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_mu[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_mu[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_sq_blur[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_sq_blur[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_sq_blur[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_sq_blur[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_sq_blur[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_sq_blur[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.cross_blur[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.cross_blur[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.cross_blur[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.ssim_map.clone(), s.n),
            );
        }
    }

    fn run_sum_ssim(&self, scale: usize) {
        let s = &self.scales[scale];
        let slot = (scale * 2) as u32; // ssim_sum slot
        reduction::launch_sum::<R>(
            &self.client,
            s.ssim_map.clone(),
            s.n,
            self.partials.clone(),
            PARTIALS_LEN,
            slot,
        );
    }

    /// Compute `mad_map = |ssim_map - avg|` then run the second
    /// reduction into the slot reserved for this scale.
    fn run_abs_diff_and_sum(&self, scale: usize, avg: f32) {
        let s = &self.scales[scale];
        unsafe {
            ssim::abs_diff_scalar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(s.ssim_map.clone(), s.n),
                ArrayArg::from_raw_parts(s.mad_map.clone(), s.n),
                avg,
            );
        }
        let slot = (scale * 2 + 1) as u32; // mad_sum slot
        reduction::launch_sum::<R>(
            &self.client,
            s.mad_map.clone(),
            s.n,
            self.partials.clone(),
            PARTIALS_LEN,
            slot,
        );
    }

    fn run_finalize(&self) {
        reduction::launch_finalize::<R>(
            &self.client,
            self.partials.clone(),
            PARTIALS_LEN,
            self.sums.clone(),
            SUMS_LEN,
            NUM_SLOTS as u32,
        );
    }

    /// Reset both reduction buffers. Required because the fast-mode
    /// path uses `Atomic<f32>::fetch_add` and inherits the previous
    /// call's accumulator otherwise.
    fn zero_partials(&mut self) {
        self.partials = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        self.sums = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; SUMS_LEN]));
    }

    fn read_sums(&mut self) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(self.sums.clone())
            .expect("read sums buffer");
        f32::from_bytes(&bytes).to_vec()
    }
}

/// Pointwise copy kernel `dst[i] = src[i]`. Used in the chroma
/// pre-blur double-pass to allow `dst == src`.
#[cube(launch_unchecked)]
pub fn copy_kernel(src: &Array<f32>, dst: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= dst.len() {
        terminate!();
    }
    dst[i] = src[i];
}

/// Convert SSIM (0-1, higher better) to DSSIM (0+, lower better).
/// Verbatim from `dssim-cuda::ssim_to_dssim`.
fn ssim_to_dssim(ssim: f64) -> f64 {
    1.0 / ssim.max(f64::EPSILON) - 1.0
}
