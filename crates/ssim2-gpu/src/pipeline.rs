//! SSIMULACRA2 pipeline orchestration.
//!
//! Wires the kernels in `kernels::*` together into the 6-octave
//! SSIMULACRA2 algorithm. Public entry points:
//!
//! - [`Ssim2::new`] + [`Ssim2::compute`] — score one image pair from sRGB.
//! - [`Ssim2::set_reference`] + [`Ssim2::compute_with_reference`] — cache
//!   reference-side state and score many distorted images against it
//!   (encoder rate-distortion search).
//!
//! Algorithm (per scale, per channel):
//!   1. linear RGB pyramid (downscale_2x).
//!   2. linear → positive XYB.
//!   3. sigma11 = ref²; sigma22 = dis²; sigma12 = ref·dis.
//!   4. Blur 5 planes: ref, dis, sigma11, sigma22, sigma12.
//!      Implementation: vertical IIR pass → transpose → vertical IIR pass.
//!      Output is in transposed orientation (saves a final transpose);
//!      compute_error_maps + reduction don't care about orientation.
//!   5. Transpose ref/dis (raw, unblurred) so error_maps can read them
//!      alongside the already-transposed mu1/mu2/sigma planes.
//!   6. compute_error_maps → ssim, artifact, detail_loss (per channel).
//!   7. Reduce each → (Σ, Σ⁴) per (scale, channel, error-map).
//!
//! Final score: weighted dot product of all 108 sub-stats with the
//! published SSIMULACRA2 weights, then the standard sigmoid-like remap.
//! Mirrors `ssimulacra2::Msssim::score` line-for-line.

use cubecl::prelude::*;

use crate::kernels::{blur, downscale, error_maps, reduction, srgb, transpose, xyb};
use crate::{Error, GpuSsim2Result, NUM_SCALES, Result};

/// Per-scale buffer set. Each plane is `width × height` f32, planar
/// (one buffer per channel of a 3-channel image).
struct Scale {
    width: u32,
    height: u32,
    n: usize,

    /// Linear RGB at this scale (planar). Kept across calls so
    /// `set_reference` can rebuild the pyramid without re-uploading.
    ref_lin: [cubecl::server::Handle; 3],
    dis_lin: [cubecl::server::Handle; 3],

    /// Positive XYB after linear→XYB conversion.
    ref_xyb: [cubecl::server::Handle; 3],
    dis_xyb: [cubecl::server::Handle; 3],

    /// Pointwise products (re-used as scratch each call).
    sigma11_in: [cubecl::server::Handle; 3], // ref·ref
    sigma22_in: [cubecl::server::Handle; 3], // dis·dis
    sigma12_in: [cubecl::server::Handle; 3], // ref·dis

    /// First-pass (vertical-walk) blur outputs. Same orientation as input.
    sigma11_v: [cubecl::server::Handle; 3],
    sigma22_v: [cubecl::server::Handle; 3],
    sigma12_v: [cubecl::server::Handle; 3],
    mu1_v: [cubecl::server::Handle; 3],
    mu2_v: [cubecl::server::Handle; 3],

    /// Transposed first-pass outputs (`width` becomes height and vice
    /// versa, stored as a `height × width` row-major buffer of size
    /// `n` floats — same total length).
    sigma11_t: [cubecl::server::Handle; 3],
    sigma22_t: [cubecl::server::Handle; 3],
    sigma12_t: [cubecl::server::Handle; 3],
    mu1_t: [cubecl::server::Handle; 3],
    mu2_t: [cubecl::server::Handle; 3],

    /// Second-pass (vertical-walk on transposed) outputs. Orientation
    /// is "transposed" — for a `width × height` source these now live
    /// in `height × width` row-major.
    sigma11_full: [cubecl::server::Handle; 3],
    sigma22_full: [cubecl::server::Handle; 3],
    sigma12_full: [cubecl::server::Handle; 3],
    mu1_full: [cubecl::server::Handle; 3],
    mu2_full: [cubecl::server::Handle; 3],

    /// Raw XYB transposed (input to compute_error_maps' `source`/`distorted`).
    ref_xyb_t: [cubecl::server::Handle; 3],
    dis_xyb_t: [cubecl::server::Handle; 3],

    /// Error maps (in transposed orientation).
    ssim: [cubecl::server::Handle; 3],
    artifact: [cubecl::server::Handle; 3],
    detail: [cubecl::server::Handle; 3],
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
            ref_xyb: alloc_3(client, n),
            dis_xyb: alloc_3(client, n),
            sigma11_in: alloc_3(client, n),
            sigma22_in: alloc_3(client, n),
            sigma12_in: alloc_3(client, n),
            sigma11_v: alloc_3(client, n),
            sigma22_v: alloc_3(client, n),
            sigma12_v: alloc_3(client, n),
            mu1_v: alloc_3(client, n),
            mu2_v: alloc_3(client, n),
            sigma11_t: alloc_3(client, n),
            sigma22_t: alloc_3(client, n),
            sigma12_t: alloc_3(client, n),
            mu1_t: alloc_3(client, n),
            mu2_t: alloc_3(client, n),
            sigma11_full: alloc_3(client, n),
            sigma22_full: alloc_3(client, n),
            sigma12_full: alloc_3(client, n),
            mu1_full: alloc_3(client, n),
            mu2_full: alloc_3(client, n),
            ref_xyb_t: alloc_3(client, n),
            dis_xyb_t: alloc_3(client, n),
            ssim: alloc_3(client, n),
            artifact: alloc_3(client, n),
            detail: alloc_3(client, n),
        }
    }
}

/// Per-instance allocations + per-call orchestration of the full
/// SSIMULACRA2 pipeline. Construct once for a given resolution; reuse
/// across many image pairs of that resolution.
pub struct Ssim2<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    /// `n_pixels` at scale 0.
    n: usize,

    /// sRGB u8 staging — re-uploaded per call.
    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    /// Per-scale buffer sets.
    scales: Vec<Scale>,

    /// Per-thread partials scratch — `NUM_SLOTS * PARTIALS_PER_REDUCTION`
    /// floats. Used in stage 1 of the two-stage reduction. Never read
    /// by the host; only the much smaller `sums` buffer crosses the
    /// device→host boundary.
    partials: cubecl::server::Handle,
    /// Final (Σ, Σ⁴) per slot — `NUM_SLOTS * 2` floats. Populated by
    /// the finalizer kernel and read once per `compute()`.
    /// Slot index: `(scale * 3 + channel) * 3 + map_type`
    ///   - map_type: 0=ssim, 1=artifact, 2=detail
    sums: cubecl::server::Handle,

    has_cached_reference: bool,
}

const NUM_SLOTS: usize = NUM_SCALES * 3 * 3; // 54
const PARTIALS_LEN: usize = NUM_SLOTS * reduction::PARTIALS_PER_REDUCTION;
const SUMS_LEN: usize = NUM_SLOTS * 2;

impl<R: Runtime> Ssim2<R> {
    /// Allocate every per-instance buffer for the given image size.
    /// Returns `Err(InvalidImageSize)` for images smaller than 8×8.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let s = Ssim2::<WgpuRuntime>::new(client, 1024, 768)?;
    /// assert_eq!(s.dimensions(), (1024, 768));
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize);
        }
        let n = (width as usize) * (height as usize);

        // Pyramid dimensions: 6 levels, each ceil(prev/2). Stop early
        // if a level would shrink below 8×8 — same as the CPU crate.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut w = width;
        let mut h = height;
        for _ in 0..NUM_SCALES {
            if w < 8 || h < 8 {
                break;
            }
            dims.push((w, h));
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        let scales = dims
            .iter()
            .map(|&(w, h)| Scale::new(&client, w, h))
            .collect::<Vec<_>>();

        let n_bytes = n * 3;
        // sRGB bytes uploaded as u32 because wgpu's WGSL backend has
        // no `u8` storage type (Array<u8> reads zero on Metal).
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

    /// Number of active pyramid scales (≤ NUM_SCALES; smaller for
    /// images that shrink below 8×8 before reaching the 6th level).
    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// `(width, height, n_pixels)` at scale `s`.
    pub fn scale_dims(&self, s: usize) -> (u32, u32, usize) {
        let sc = &self.scales[s];
        (sc.width, sc.height, sc.n)
    }

    /// Cached reference handles needed by `Ssim2Batch`. After
    /// `set_reference`, these hold:
    /// - `ref_xyb_t[ch]`: transposed raw reference XYB (`source` input
    ///   to `error_maps`).
    /// - `mu1_full[ch]`: fully-blurred reference XYB (transposed
    ///   orientation, `mu1` input to `error_maps`).
    /// - `sigma11_full[ch]`: fully-blurred ref·ref (`sigma11` input).
    /// - `ref_xyb[ch]`: raw reference XYB (used by `Ssim2Batch` for
    ///   the broadcast `sigma12 = ref_xyb · dis_xyb_batched` mul).
    pub(crate) fn cached_ref_xyb_t(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].ref_xyb_t
    }
    pub(crate) fn cached_mu1_full(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].mu1_full
    }
    pub(crate) fn cached_sigma11_full(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].sigma11_full
    }
    pub(crate) fn cached_ref_xyb(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].ref_xyb
    }
    pub(crate) fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Score one image pair, both sRGB packed RGB u8 of length
    /// `width × height × 3`.
    ///
    /// Returns `Err(DimensionMismatch)` if either buffer's length
    /// doesn't match the configured image size.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
    /// let r = vec![0_u8; 256 * 256 * 3];
    /// let d = vec![0_u8; 256 * 256 * 3];
    /// let score = s.compute(&r, &d)?.score;
    /// assert!((score - 100.0).abs() < 0.1); // identical → ~100
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<GpuSsim2Result> {
        self.check_dims(ref_srgb)?;
        self.check_dims(dist_srgb)?;

        // Upload + sRGB → linear for both sides into scale-0 buffers.
        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.upload_and_srgb_to_linear(false, dist_srgb);

        // Build linear pyramid.
        self.build_linear_pyramid(true);
        self.build_linear_pyramid(false);

        // Per-scale processing — populates per-thread partials.
        for s in 0..self.scales.len() {
            self.process_scale(s, true, true);
        }
        // Stage-2 finalizer folds partials → small (slot, sum, p4) buffer.
        self.run_finalizer();

        Ok(GpuSsim2Result {
            score: self.read_and_aggregate(),
        })
    }

    /// Cache reference-side state for many comparisons against a fixed
    /// reference. Subsequent `compute_with_reference` calls skip the
    /// reference-side pyramid + XYB + ref²-blur work.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
    /// let r = vec![0_u8; 256 * 256 * 3];
    /// s.set_reference(&r)?;
    /// assert!(s.has_cached_reference());
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_dims(ref_srgb)?;
        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.build_linear_pyramid(true);
        for s in 0..self.scales.len() {
            self.run_xyb(s, true);
            self.run_self_products(s, true);
            self.run_blur_pair(s, true);
            // Pre-transpose the raw reference XYB so subsequent
            // compute_with_reference / Ssim2Batch::compute_batch calls
            // can read it directly without re-transposing.
            self.run_transpose_raw_xyb_pair(s, true, false);
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
    /// `Err(NoCachedReference)` if `set_reference` hasn't been called,
    /// `Err(DimensionMismatch)` if the buffer's length doesn't match
    /// the configured image size.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::{Error, Ssim2};
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
    /// // Without set_reference first:
    /// let d = vec![0_u8; 256 * 256 * 3];
    /// assert!(matches!(
    ///     s.compute_with_reference(&d),
    ///     Err(Error::NoCachedReference)
    /// ));
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<GpuSsim2Result> {
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        self.check_dims(dist_srgb)?;

        self.upload_and_srgb_to_linear(false, dist_srgb);
        self.build_linear_pyramid(false);

        for s in 0..self.scales.len() {
            self.run_xyb(s, false);
            self.run_self_products(s, false); // sigma22
            self.run_cross_product(s); // sigma12
            self.run_blur_dis_only(s);
            // ref_xyb_t was cached by set_reference; only transpose dis.
            self.run_transpose_raw_xyb_pair(s, false, true);
            self.run_error_maps(s);
            self.run_reductions(s);
        }
        self.run_finalizer();

        Ok(GpuSsim2Result {
            score: self.read_and_aggregate(),
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
    fn blur_cube_count(width: u32) -> CubeCount {
        let cubes = (width + blur::BLOCK_WIDTH - 1) / blur::BLOCK_WIDTH;
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn blur_cube_dim() -> CubeDim {
        CubeDim::new_1d(blur::BLOCK_WIDTH)
    }

    fn upload_and_srgb_to_linear(&mut self, is_a: bool, srgb: &[u8]) {
        let n_bytes = self.n * 3;
        // Widen each sRGB byte to a u32 so wgpu/Metal can read it
        // natively (WGSL has no u8 storage type).
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

    fn run_xyb(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let (lin, xyb_buf) = if is_a {
            (&s.ref_lin, &s.ref_xyb)
        } else {
            (&s.dis_lin, &s.dis_xyb)
        };
        unsafe {
            xyb::linear_to_xyb_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), s.n),
                ArrayArg::from_raw_parts(lin[1].clone(), s.n),
                ArrayArg::from_raw_parts(lin[2].clone(), s.n),
                ArrayArg::from_raw_parts(xyb_buf[0].clone(), s.n),
                ArrayArg::from_raw_parts(xyb_buf[1].clone(), s.n),
                ArrayArg::from_raw_parts(xyb_buf[2].clone(), s.n),
            );
        }
    }

    /// Pointwise product `a · b → out` for one scale × all 3 channels.
    fn pointwise_mul(
        &self,
        scale: usize,
        a: &[cubecl::server::Handle; 3],
        b: &[cubecl::server::Handle; 3],
        out: &[cubecl::server::Handle; 3],
    ) {
        let n = self.scales[scale].n;
        for ch in 0..3 {
            unsafe {
                pointwise_mul_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(a[ch].clone(), n),
                    ArrayArg::from_raw_parts(b[ch].clone(), n),
                    ArrayArg::from_raw_parts(out[ch].clone(), n),
                );
            }
        }
    }

    /// Compute self-products: `sigma11 = ref²` (if is_a) or
    /// `sigma22 = dis²` (else).
    fn run_self_products(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        if is_a {
            self.pointwise_mul(scale, &s.ref_xyb, &s.ref_xyb, &s.sigma11_in);
        } else {
            self.pointwise_mul(scale, &s.dis_xyb, &s.dis_xyb, &s.sigma22_in);
        }
    }

    fn run_cross_product(&self, scale: usize) {
        let s = &self.scales[scale];
        self.pointwise_mul(scale, &s.ref_xyb, &s.dis_xyb, &s.sigma12_in);
    }

    /// One-plane two-pass blur: `src → v_pass → tv → tv → full` where
    /// `tv` is the transpose of the first pass output, and `full`
    /// receives the second vertical-walk result (in transposed
    /// orientation). Caller supplies all 3 same-channel buffers.
    fn blur_plane_two_pass(
        &self,
        width: u32,
        height: u32,
        n: usize,
        src: &cubecl::server::Handle,
        v_buf: &cubecl::server::Handle,
        t_buf: &cubecl::server::Handle,
        full: &cubecl::server::Handle,
    ) {
        unsafe {
            // 1. v-pass on src (walks columns of width × height) → v_buf.
            blur::blur_pass_kernel::launch_unchecked::<R>(
                &self.client,
                Self::blur_cube_count(width),
                Self::blur_cube_dim(),
                ArrayArg::from_raw_parts(src.clone(), n),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                width,
                height,
            );
            // 2. transpose v_buf → t_buf (now height × width).
            transpose::transpose_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                ArrayArg::from_raw_parts(t_buf.clone(), n),
                width,
                height,
            );
            // 3. v-pass on t_buf (walks columns of height × width) → full.
            //    Note: the transposed buffer's "width" is the original height.
            blur::blur_pass_kernel::launch_unchecked::<R>(
                &self.client,
                Self::blur_cube_count(height),
                Self::blur_cube_dim(),
                ArrayArg::from_raw_parts(t_buf.clone(), n),
                ArrayArg::from_raw_parts(full.clone(), n),
                height,
                width,
            );
        }
    }

    /// Pointwise transpose for the `ref_xyb` / `dis_xyb` planes (raw,
    /// unblurred — used as `source` / `distorted` inputs to compute_error_maps).
    fn run_transpose_raw_xyb_pair(&self, scale: usize, do_ref: bool, do_dis: bool) {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        if do_ref {
            for ch in 0..3 {
                unsafe {
                    transpose::transpose_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(s.ref_xyb[ch].clone(), n),
                        ArrayArg::from_raw_parts(s.ref_xyb_t[ch].clone(), n),
                        w,
                        h,
                    );
                }
            }
        }
        if do_dis {
            for ch in 0..3 {
                unsafe {
                    transpose::transpose_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(s.dis_xyb[ch].clone(), n),
                        ArrayArg::from_raw_parts(s.dis_xyb_t[ch].clone(), n),
                        w,
                        h,
                    );
                }
            }
        }
    }

    /// Reference-only blur pass (sigma11 + mu1). Used by `set_reference`
    /// — populates `sigma11_full` and `mu1_full`.
    fn run_blur_pair(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        if is_a {
            for ch in 0..3 {
                self.blur_plane_two_pass(
                    w, h, n,
                    &s.sigma11_in[ch],
                    &s.sigma11_v[ch],
                    &s.sigma11_t[ch],
                    &s.sigma11_full[ch],
                );
                self.blur_plane_two_pass(
                    w, h, n,
                    &s.ref_xyb[ch],
                    &s.mu1_v[ch],
                    &s.mu1_t[ch],
                    &s.mu1_full[ch],
                );
            }
        } else {
            for ch in 0..3 {
                self.blur_plane_two_pass(
                    w, h, n,
                    &s.sigma22_in[ch],
                    &s.sigma22_v[ch],
                    &s.sigma22_t[ch],
                    &s.sigma22_full[ch],
                );
                self.blur_plane_two_pass(
                    w, h, n,
                    &s.dis_xyb[ch],
                    &s.mu2_v[ch],
                    &s.mu2_t[ch],
                    &s.mu2_full[ch],
                );
            }
        }
    }

    /// Distorted-only blur pass: sigma22, mu2, sigma12. Used by
    /// `compute_with_reference` (assumes sigma11_full + mu1_full are
    /// cached).
    fn run_blur_dis_only(&self, scale: usize) {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        for ch in 0..3 {
            self.blur_plane_two_pass(
                w, h, n,
                &s.sigma22_in[ch],
                &s.sigma22_v[ch],
                &s.sigma22_t[ch],
                &s.sigma22_full[ch],
            );
            self.blur_plane_two_pass(
                w, h, n,
                &s.dis_xyb[ch],
                &s.mu2_v[ch],
                &s.mu2_t[ch],
                &s.mu2_full[ch],
            );
            self.blur_plane_two_pass(
                w, h, n,
                &s.sigma12_in[ch],
                &s.sigma12_v[ch],
                &s.sigma12_t[ch],
                &s.sigma12_full[ch],
            );
        }
    }

    /// Run all 5 blurs (ref+dis paths) for one scale.
    fn run_blur_full(&self, scale: usize) {
        self.run_blur_pair(scale, true);
        self.run_blur_pair(scale, false);
        let s = &self.scales[scale];
        for ch in 0..3 {
            self.blur_plane_two_pass(
                s.width, s.height, s.n,
                &s.sigma12_in[ch],
                &s.sigma12_v[ch],
                &s.sigma12_t[ch],
                &s.sigma12_full[ch],
            );
        }
    }

    fn run_error_maps(&self, scale: usize) {
        let s = &self.scales[scale];
        for ch in 0..3 {
            unsafe {
                error_maps::error_maps_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.ref_xyb_t[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.dis_xyb_t[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.mu1_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.mu2_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma11_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma22_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma12_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.ssim[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.artifact[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.detail[ch].clone(), s.n),
                );
            }
        }
    }

    fn run_reductions(&self, scale: usize) {
        let s = &self.scales[scale];
        for ch in 0..3 {
            let plane_handles = [&s.ssim[ch], &s.artifact[ch], &s.detail[ch]];
            for map_type in 0..3 {
                // Slot encoding: (scale * 3 + ch) * 3 + map_type ∈ [0, 54).
                let slot = ((scale * 3 + ch) * 3 + map_type) as u32;
                reduction::launch_sum_p4::<R>(
                    &self.client,
                    plane_handles[map_type].clone(),
                    s.n,
                    self.partials.clone(),
                    PARTIALS_LEN,
                    slot,
                );
            }
        }
    }

    /// Stage-2 finalizer: fold per-thread partials into a small
    /// `(slot, sum, p4)` buffer the host reads back. One launch per
    /// `compute()` instead of 16× the readback the partials buffer would
    /// require if the host folded directly.
    fn run_finalizer(&self) {
        reduction::launch_finalize::<R>(
            &self.client,
            self.partials.clone(),
            PARTIALS_LEN,
            self.sums.clone(),
            SUMS_LEN,
            NUM_SLOTS as u32,
        );
    }

    /// Per-scale processing for `compute()`: XYB, products, blurs,
    /// error maps, reduction. Called for every pyramid scale.
    fn process_scale(&self, scale: usize, _do_ref: bool, _do_dis: bool) {
        let s = &self.scales[scale];
        // 1. linear → XYB for both sides.
        self.run_xyb(scale, true);
        self.run_xyb(scale, false);
        // 2. Pointwise products: sigma11 = ref²; sigma22 = dis²; sigma12 = ref·dis.
        self.pointwise_mul(scale, &s.ref_xyb, &s.ref_xyb, &s.sigma11_in);
        self.pointwise_mul(scale, &s.dis_xyb, &s.dis_xyb, &s.sigma22_in);
        self.pointwise_mul(scale, &s.ref_xyb, &s.dis_xyb, &s.sigma12_in);
        // 3. Blur all 5: sigma11/22/12 + ref_xyb (mu1) + dis_xyb (mu2).
        self.run_blur_full(scale);
        // 4. Transpose raw XYB so error_maps reads them in the same
        //    orientation as the (transposed) blurred buffers.
        self.run_transpose_raw_xyb_pair(scale, true, true);
        // 5. Per-pixel error maps.
        self.run_error_maps(scale);
        // 6. Reduce to (Σ, Σ⁴) per (channel × map type).
        self.run_reductions(scale);
    }

    /// Read the sums buffer back to host and compute the final SSIMULACRA2
    /// score. Mirrors `ssimulacra2::Msssim::score` exactly (same WEIGHT
    /// table, same sigmoid).
    fn read_and_aggregate(&mut self) -> f64 {
        let bytes = self
            .client
            .read_one(self.sums.clone())
            .expect("read sums buffer");
        let raw = f32::from_bytes(&bytes);
        debug_assert_eq!(raw.len(), SUMS_LEN);

        // Reset both buffers for the next call. In fast mode the
        // partials buffer is the atomic-add target so accumulators
        // need zeroing; in portable mode each thread writes its slot
        // unconditionally so partials reset is technically optional
        // but cheap.
        self.partials = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        self.sums = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; SUMS_LEN]));

        // Layout post-finalizer: `raw[slot * 2]` = Σ, `raw[slot * 2 + 1]` = Σ⁴.
        // Total length = NUM_SLOTS * 2 = 108 floats. The 4096 per-thread
        // partials per slot were already folded on-device by the
        // finalizer kernel — much less device→host bandwidth than reading
        // the full partials buffer.
        let mut avg_ssim = vec![[0.0_f64; 6]; NUM_SCALES]; // [scale][c*2 + n]
        let mut avg_edgediff = vec![[0.0_f64; 12]; NUM_SCALES]; // [scale][c*4 + n]

        let fold_slot = |slot: usize| -> (f64, f64) {
            (raw[slot * 2] as f64, raw[slot * 2 + 1] as f64)
        };

        for scale in 0..self.scales.len() {
            let n_pix = self.scales[scale].n as f64;
            let one_per_pixels = 1.0 / n_pix;
            for ch in 0..3 {
                let s_slot = (scale * 3 + ch) * 3; // ssim
                let a_slot = s_slot + 1;           // artifact
                let d_slot = s_slot + 2;           // detail
                let (s_sum, s_p4) = fold_slot(s_slot);
                let (a_sum, a_p4) = fold_slot(a_slot);
                let (d_sum, d_p4) = fold_slot(d_slot);

                avg_ssim[scale][ch * 2] = one_per_pixels * s_sum;
                avg_ssim[scale][ch * 2 + 1] = (one_per_pixels * s_p4).sqrt().sqrt();

                avg_edgediff[scale][ch * 4] = one_per_pixels * a_sum;
                avg_edgediff[scale][ch * 4 + 1] = (one_per_pixels * a_p4).sqrt().sqrt();
                avg_edgediff[scale][ch * 4 + 2] = one_per_pixels * d_sum;
                avg_edgediff[scale][ch * 4 + 3] = (one_per_pixels * d_p4).sqrt().sqrt();
            }
        }

        score_from_stats(&avg_ssim, &avg_edgediff, self.scales.len())
    }
}

/// Pointwise multiply kernel `out = a · b` (single plane). Exposed
/// to the batched pipeline because the unbatched kernel works fine on
/// flat batched arrays — `Ssim2Batch::process_scale_batched` uses it
/// for `sigma22 = dis · dis`.
#[cube(launch_unchecked)]
pub fn pointwise_mul_kernel(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= out.len() {
        terminate!();
    }
    out[idx] = a[idx] * b[idx];
}

/// Final score: weighted dot-product of 108 stats then sigmoid remap.
/// Verbatim from `ssimulacra2::Msssim::score` — for fewer-than-6 actual
/// scales, missing entries are treated as 0 (matches the CPU which
/// `break`s the per-scale loop early but still iterates all 6 weight
/// rows; slots beyond `n_scales` stay at their `0.0` default).
pub(crate) fn score_from_stats(avg_ssim: &[[f64; 6]], avg_edgediff: &[[f64; 12]], n_scales: usize) -> f64 {
    const WEIGHT: [f64; 108] = [
        0.0,
        0.000_737_660_670_740_658_6,
        0.0,
        0.0,
        0.000_779_348_168_286_730_9,
        0.0,
        0.0,
        0.000_437_115_573_010_737_9,
        0.0,
        1.104_172_642_665_734_6,
        0.000_662_848_341_292_71,
        0.000_152_316_327_837_187_52,
        0.0,
        0.001_640_643_745_659_975_4,
        0.0,
        1.842_245_552_053_929_8,
        11.441_172_603_757_666,
        0.0,
        0.000_798_910_943_601_516_3,
        0.000_176_816_438_078_653,
        0.0,
        1.878_759_497_954_638_7,
        10.949_069_906_051_42,
        0.0,
        0.000_728_934_699_150_807_2,
        0.967_793_708_062_683_3,
        0.0,
        0.000_140_034_242_854_358_84,
        0.998_176_697_785_496_7,
        0.000_319_497_559_344_350_53,
        0.000_455_099_211_379_206_3,
        0.0,
        0.0,
        0.001_364_876_616_324_339_8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.466_890_328_078_848,
        0.0,
        17.445_833_984_131_262,
        0.000_623_560_163_404_146_6,
        0.0,
        0.0,
        6.683_678_146_179_332,
        0.000_377_244_079_796_112_96,
        1.027_889_937_768_264,
        225.205_153_008_492_74,
        0.0,
        0.0,
        19.213_238_186_143_016,
        0.001_140_152_458_661_836_1,
        0.001_237_755_635_509_985,
        176.393_175_984_506_94,
        0.0,
        0.0,
        24.433_009_998_704_76,
        0.285_208_026_121_177_57,
        0.000_448_543_692_383_340_8,
        0.0,
        0.0,
        0.0,
        34.779_063_444_837_72,
        44.835_625_328_877_896,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_868_055_657_329_169_8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_531_319_187_435_874_7,
        0.0,
        0.000_165_338_141_613_791_12,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_417_917_180_325_133_6,
        0.001_729_082_823_472_283_3,
        0.0,
        0.002_082_700_584_663_643_7,
        0.0,
        0.0,
        8.826_982_764_996_862,
        23.192_433_439_989_26,
        0.0,
        95.108_049_881_108_6,
        0.986_397_803_440_068_2,
        0.983_438_279_246_535_3,
        0.001_228_640_504_827_849_3,
        171.266_725_589_730_7,
        0.980_785_887_243_537_9,
        0.0,
        0.0,
        0.0,
        0.000_513_006_458_899_067_9,
        0.0,
        0.000_108_540_578_584_115_37,
    ];

    let mut ssim = 0.0_f64;
    let mut i = 0_usize;
    for c in 0..3 {
        for scale_idx in 0..NUM_SCALES {
            for n in 0..2 {
                let avg_s = if scale_idx < n_scales {
                    avg_ssim[scale_idx][c * 2 + n]
                } else {
                    0.0
                };
                let avg_a = if scale_idx < n_scales {
                    avg_edgediff[scale_idx][c * 4 + n]
                } else {
                    0.0
                };
                let avg_d = if scale_idx < n_scales {
                    avg_edgediff[scale_idx][c * 4 + n + 2]
                } else {
                    0.0
                };
                ssim = WEIGHT[i].mul_add(avg_s.abs(), ssim);
                i += 1;
                ssim = WEIGHT[i].mul_add(avg_a.abs(), ssim);
                i += 1;
                ssim = WEIGHT[i].mul_add(avg_d.abs(), ssim);
                i += 1;
            }
        }
    }

    ssim *= 0.956_238_261_683_484_4_f64;
    ssim = (6.248_496_625_763_138e-5 * ssim * ssim).mul_add(
        ssim,
        2.326_765_642_916_932_f64.mul_add(ssim, -0.020_884_521_182_843_837 * ssim * ssim),
    );

    if ssim > 0.0 {
        ssim = ssim
            .powf(0.627_633_646_783_138_7)
            .mul_add(-10.0, 100.0)
    } else {
        ssim = 100.0;
    }

    ssim
}
