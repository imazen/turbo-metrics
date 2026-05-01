//! Butteraugli pipeline orchestration.
//!
//! The full butteraugli algorithm wired together as kernel launches over
//! pre-allocated CubeCL buffers. Single-resolution flavor only — the
//! multi-resolution supersample-add and the reference cache are
//! follow-up work.
//!
//! Constants and orchestration follow the CPU `butteraugli` v0.9.2
//! crate's `compute_psycho_diff_malta` and `compute_mask_from_hf_uhf`
//! stages — see comments next to each step for the CPU function it
//! mirrors.

use cubecl::prelude::*;

use crate::GpuButteraugliResult;
use crate::kernels::{blur, colors, diffmap, frequency, malta, masking, reduction};

/// Default intensity multiplier — value of one display nit relative to
/// linear-light input scale. CPU butteraugli passes `80.0` for the
/// standard 80-nit display directly to `opsin_dynamics_image`. Linear
/// inputs already live on [0, 1] (after sRGB transfer); they get scaled
/// to [0, 80] inside opsin, *not* divided by 255 again.
pub const DEFAULT_INTENSITY_MULTIPLIER: f32 = 80.0;

// ═══ frequency separation ═══
const SIGMA_LF: f32 = 7.155_933_4;
/// Sigma for opsin's sensitivity-input blur. CPU butteraugli's
/// `opsin_dynamics_image` always uses this 5-tap blur (with mirrored
/// boundaries, kernel radius 2) — *not* SIGMA_LF. Mismatching this
/// causes the per-pixel sensitivity to be smoothed over a 16-pixel
/// radius instead of a 2-pixel radius, which drops the perturbed
/// pixel's apparent contrast by ~12 % on tiny perturbations and
/// roughly doubles the diffmap when one bright pixel is involved.
const SIGMA_OPSIN: f32 = 1.2;
const SIGMA_HF: f32 = 3.224_899_0;
const SIGMA_UHF: f32 = 1.564_163_3;
const REMOVE_MF_RANGE: f32 = 0.29;
const ADD_MF_RANGE: f32 = 0.1;
const REMOVE_HF_RANGE: f32 = 1.5;
const REMOVE_UHF_RANGE: f32 = 0.04;
const SUPPRESS_XY: f32 = 46.0;

// ═══ asymmetry ═══
const HF_ASYMMETRY: f32 = 1.0;

// ═══ Malta band parameters (libjxl/butteraugli) ═══
const W_MF_MALTA: f64 = 37.081_987_039_9;
const NORM1_MF: f64 = 130_262_059.556;
const W_MF_MALTA_X: f64 = 8_246.753_213_53;
const NORM1_MF_X: f64 = 1_009_002.705_82;
const W_HF_MALTA: f64 = 18.723_741_438_7;
const NORM1_HF: f64 = 4_498_534.452_32;
const W_HF_MALTA_X: f64 = 6_923.994_761_09;
const NORM1_HF_X: f64 = 8_051.158_332_47;
const W_UHF_MALTA: f64 = 1.100_390_325_55;
const NORM1_UHF: f64 = 71.780_027_516_9;
const W_UHF_MALTA_X: f64 = 173.5;
const NORM1_UHF_X: f64 = 5.0;

// ═══ frequency-band weights (l2_diff and DC contribution) ═══
const WMUL: [f64; 9] = [
    400.0,
    1.508_157_031_18,
    0.0,
    2_150.0,
    10.619_543_323_9,
    16.217_604_315_2,
    29.235_379_799_4,
    0.844_626_970_982,
    0.703_646_627_719,
];

// ═══ mask blur radius ═══
const MASK_RADIUS: f32 = 2.7;

/// Compute Malta `(norm2_0gt1, norm2_0lt1, norm1_f32)` host-side at
/// f64 precision. Mirrors the f64 prelude in CPU `malta_diff_map_impl`.
fn malta_norm(w_0gt1: f64, w_0lt1: f64, norm1: f64, use_lf: bool) -> (f32, f32, f32) {
    const K_WEIGHT0: f64 = 0.5;
    const K_WEIGHT1: f64 = 0.33;
    const LEN: f64 = 3.75;
    let mulli = if use_lf {
        0.611_612_573_796
    } else {
        0.399_058_176_37
    };
    let w_pre0gt1 = mulli * (K_WEIGHT0 * w_0gt1).sqrt() / (LEN * 2.0 + 1.0);
    let w_pre0lt1 = mulli * (K_WEIGHT1 * w_0lt1).sqrt() / (LEN * 2.0 + 1.0);
    let norm2_0gt1 = (w_pre0gt1 * norm1) as f32;
    let norm2_0lt1 = (w_pre0lt1 * norm1) as f32;
    (norm2_0gt1, norm2_0lt1, norm1 as f32)
}

/// Per-instance allocations + per-call orchestration of the full
/// butteraugli kernel pipeline. Construct once, reuse for many
/// comparisons at the same resolution.
pub struct Butteraugli<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    n: usize,

    // sRGB u8 staging
    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    // Planar linear RGB / XYB after opsin (×2 images × 3 channels = 6 buffers)
    lin_a: [cubecl::server::Handle; 3],
    lin_b: [cubecl::server::Handle; 3],

    // Blurred linear RGB (for opsin dynamics adaptation)
    blur_a: [cubecl::server::Handle; 3],
    blur_b: [cubecl::server::Handle; 3],

    // Frequency bands per channel: [UHF, HF, MF, LF] × [X, Y, B] × 2 images
    freq_a: [[cubecl::server::Handle; 3]; 4],
    freq_b: [[cubecl::server::Handle; 3]; 4],

    // Per-pixel block diff accumulators [X, Y, B]
    block_diff_dc: [cubecl::server::Handle; 3],
    block_diff_ac: [cubecl::server::Handle; 3],

    // Mask plane + scratch for the blurred mask of image B
    mask: cubecl::server::Handle,
    mask_scratch: cubecl::server::Handle,

    // Final diffmap
    diffmap_buf: cubecl::server::Handle,

    // Generic temp planes
    temp1: cubecl::server::Handle,
    temp2: cubecl::server::Handle,
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

impl<R: Runtime> Butteraugli<R> {
    /// Allocate all per-instance buffers for `width × height` images.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        let n_bytes = n * 3;
        let src_u8_a = client.create_from_slice(&vec![0_u8; n_bytes]);
        let src_u8_b = client.create_from_slice(&vec![0_u8; n_bytes]);

        let lin_a = alloc_3(&client, n);
        let lin_b = alloc_3(&client, n);
        let blur_a = alloc_3(&client, n);
        let blur_b = alloc_3(&client, n);

        let freq_a = [
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
        ];
        let freq_b = [
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
        ];

        let block_diff_dc = alloc_3(&client, n);
        let block_diff_ac = alloc_3(&client, n);

        let mask = alloc_plane(&client, n);
        let mask_scratch = alloc_plane(&client, n);
        let diffmap_buf = alloc_plane(&client, n);
        let temp1 = alloc_plane(&client, n);
        let temp2 = alloc_plane(&client, n);

        Self {
            client,
            width,
            height,
            n,
            src_u8_a,
            src_u8_b,
            lin_a,
            lin_b,
            blur_a,
            blur_b,
            freq_a,
            freq_b,
            block_diff_dc,
            block_diff_ac,
            mask,
            mask_scratch,
            diffmap_buf,
            temp1,
            temp2,
        }
    }

    /// Compute the butteraugli `(score, pnorm_3)` for one image pair.
    /// Both images are sRGB u8 packed RGB (`width × height × 3` bytes).
    ///
    /// Single-resolution only — the multi-resolution supersample-add is a
    /// separate follow-up. Empirically the multi-res contribution is
    /// 5-15 % of the score on natural images.
    pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> GpuButteraugliResult {
        let n_bytes = self.n * 3;
        assert_eq!(ref_srgb.len(), n_bytes);
        assert_eq!(dist_srgb.len(), n_bytes);

        // Re-upload pixel data (caller holds CPU-side copies).
        self.src_u8_a = self.client.create_from_slice(ref_srgb);
        self.src_u8_b = self.client.create_from_slice(dist_srgb);

        // ── 1. sRGB u8 → planar linear RGB ──
        unsafe {
            self.launch_srgb_to_linear(true);
            self.launch_srgb_to_linear(false);
        }

        // ── 2. Blur linear RGB at sigma=SIGMA_OPSIN (1.2) for opsin
        //       sensitivity input. CPU's `opsin_dynamics_image` uses a
        //       5-tap blur at sigma=1.2; mirroring this is essential to
        //       avoid double-smoothing the sensitivity input. ──
        for ch in 0..3 {
            self.blur_plane(
                &self.lin_a[ch].clone(),
                &self.blur_a[ch].clone(),
                SIGMA_OPSIN,
            );
            self.blur_plane(
                &self.lin_b[ch].clone(),
                &self.blur_b[ch].clone(),
                SIGMA_OPSIN,
            );
        }

        // ── 3. Opsin dynamics: linear-RGB + blurred → planar XYB ──
        unsafe {
            self.launch_opsin(true);
            self.launch_opsin(false);
        }

        // ── 4. Frequency separation for each image ──
        self.separate_frequencies(true);
        self.separate_frequencies(false);

        // ── 5. Compute psycho diff (Malta + L2_asym + L2 + mask_to_error)
        //       into block_diff_ac[0..2] ──
        self.compute_psycho_diff();

        // ── 6. DC (LF) diffs into block_diff_dc[0..2] ──
        self.compute_dc_diff();

        // ── 7. Mask: combine_channels + diff_precompute + blur(σ=2.7) +
        //       fuzzy_erosion → self.mask;
        //       accumulate mask_to_error contribution into block_diff_ac[1] ──
        self.compute_mask_pipeline();

        // ── 8. compute_diffmap: mask + DC + AC → diffmap ──
        unsafe {
            self.launch_compute_diffmap();
        }

        // ── 9. Reduce diffmap to (score, pnorm_3) ──
        reduction::reduce::<R>(&self.client, self.diffmap_buf.clone(), self.n)
    }

    // ───────────────────────────── helpers ─────────────────────────────

    fn cube_count_1d(&self) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = ((self.n as u32) + TPB - 1) / TPB;
        CubeCount::Static(cubes, 1, 1)
    }

    fn cube_dim_1d(&self) -> CubeDim {
        CubeDim::new_1d(256)
    }

    fn cube_count_2d(&self) -> CubeCount {
        let bx = (self.width + 15) / 16;
        let by = (self.height + 15) / 16;
        CubeCount::Static(bx, by, 1)
    }

    fn cube_dim_2d(&self) -> CubeDim {
        CubeDim::new_2d(16, 16)
    }

    unsafe fn launch_srgb_to_linear(&self, is_a: bool) {
        let n_bytes = self.n * 3;
        let (src, lin) = if is_a {
            (&self.src_u8_a, &self.lin_a)
        } else {
            (&self.src_u8_b, &self.lin_b)
        };
        unsafe {
            colors::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), n_bytes),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
            );
        }
    }

    unsafe fn launch_opsin(&self, is_a: bool) {
        let (lin, bl) = if is_a {
            (&self.lin_a, &self.blur_a)
        } else {
            (&self.lin_b, &self.blur_b)
        };
        unsafe {
            colors::opsin_dynamics_planar_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
                ArrayArg::from_raw_parts(bl[0].clone(), self.n),
                ArrayArg::from_raw_parts(bl[1].clone(), self.n),
                ArrayArg::from_raw_parts(bl[2].clone(), self.n),
                DEFAULT_INTENSITY_MULTIPLIER,
            );
        }
    }

    /// Helper: H+V Gaussian blur with given sigma. Two kernel launches.
    /// `temp1` is reused as the H→V intermediate.
    fn blur_plane(&self, src: &cubecl::server::Handle, dst: &cubecl::server::Handle, sigma: f32) {
        unsafe {
            blur::horizontal_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
            blur::vertical_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
        }
    }

    /// H+V blur with a caller-supplied scratch (so we can blur into
    /// `temp1` without overwriting it mid-pass).
    fn blur_plane_via(
        &self,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        scratch: &cubecl::server::Handle,
        sigma: f32,
    ) {
        unsafe {
            blur::horizontal_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(scratch.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
            blur::vertical_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
        }
    }

    fn copy_plane(&self, src: &cubecl::server::Handle, dst: &cubecl::server::Handle) {
        unsafe {
            frequency::copy_plane_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
            );
        }
    }

    fn zero_plane(&self, dst: &cubecl::server::Handle) {
        unsafe {
            frequency::zero_plane_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
            );
        }
    }

    fn subtract_arrays(
        &self,
        src1: &cubecl::server::Handle,
        src2: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
    ) {
        unsafe {
            frequency::subtract_arrays_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src1.clone(), self.n),
                ArrayArg::from_raw_parts(src2.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
            );
        }
    }

    /// Frequency separation (LF/MF, MF/HF, HF/UHF) for one of the two
    /// image sides. Mirrors CPU `psycho::separate_frequencies` — see that
    /// function for the algorithm.
    fn separate_frequencies(&mut self, is_a: bool) {
        // Borrow split: take cloned handles up front so we can mutate
        // freq_*[1][0]/[1][1] in-place via copy at the UHF step.
        let lin = if is_a {
            self.lin_a.clone()
        } else {
            self.lin_b.clone()
        };
        let freq = if is_a { &self.freq_a } else { &self.freq_b };

        // ── Step 1: LF (low-pass) and MF = XYB − LF ──
        for ch in 0..3 {
            // Blur into freq[3][ch] using temp1 as the H→V scratch.
            self.blur_plane_via(&lin[ch], &freq[3][ch], &self.temp1, SIGMA_LF);
            // MF = XYB − LF
            self.subtract_arrays(&lin[ch], &freq[3][ch], &freq[2][ch]);
        }
        // xyb_low_freq_to_vals on LF — CPU `xyb_low_freq_to_vals`.
        unsafe {
            frequency::xyb_low_freq_to_vals_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[3][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[3][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[3][2].clone(), self.n),
            );
        }

        // ── Step 2: MF/HF separation ──
        // X (ch=0): blur(MF_X) into temp1; split: HF_X = orig - blur,
        //           MF_X = remove_range(blur, REMOVE_MF_RANGE)
        self.blur_plane_via(
            &freq[2][0],
            &self.temp1.clone(),
            &self.temp2.clone(),
            SIGMA_HF,
        );
        unsafe {
            frequency::split_band_remove_inplace_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[2][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                REMOVE_MF_RANGE,
            );
        }
        // Y (ch=1): blur(MF_Y); HF_Y = orig - blur, MF_Y = amplify_range(blur, ADD_MF_RANGE)
        self.blur_plane_via(
            &freq[2][1],
            &self.temp1.clone(),
            &self.temp2.clone(),
            SIGMA_HF,
        );
        unsafe {
            frequency::split_band_amplify_inplace_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[2][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ADD_MF_RANGE,
            );
        }
        // B (ch=2): blur(MF_B) → temp1; copy temp1 → MF_B (no HF for B)
        self.blur_plane_via(
            &freq[2][2],
            &self.temp1.clone(),
            &self.temp2.clone(),
            SIGMA_HF,
        );
        self.copy_plane(&self.temp1.clone(), &freq[2][2]);

        // suppress_x_by_y(HF_y → HF_x)
        unsafe {
            frequency::suppress_x_by_y_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                SUPPRESS_XY,
            );
        }

        // ── Step 3: HF/UHF separation ──
        // X (ch=0): blur(HF_X) → temp1; split → UHF_X (freq[0][0]),
        //           final HF_X (temp2); copy temp2 → freq[1][0].
        self.blur_plane_via(
            &freq[1][0],
            &self.temp1.clone(),
            &self.mask_scratch.clone(),
            SIGMA_UHF,
        );
        unsafe {
            frequency::split_uhf_hf_x_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                REMOVE_UHF_RANGE,
                REMOVE_HF_RANGE,
            );
        }
        self.copy_plane(&self.temp2.clone(), &freq[1][0]);

        // Y (ch=1): same shape, Y kernel with maximum_clamp + amplify_range.
        self.blur_plane_via(
            &freq[1][1],
            &self.temp1.clone(),
            &self.mask_scratch.clone(),
            SIGMA_UHF,
        );
        unsafe {
            frequency::split_uhf_hf_y_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
            );
        }
        self.copy_plane(&self.temp2.clone(), &freq[1][1]);
    }

    /// Compute the AC half of the diffmap: 6 Malta diffs + 2 L2-asym
    /// (HF X/Y) + 3 L2 (MF X/Y/B). Mirrors CPU `compute_psycho_diff_malta`.
    fn compute_psycho_diff(&self) {
        // hf_asymmetry default = 1.0 → all multiplicative factors collapse.
        let asym = HF_ASYMMETRY as f64;
        let sqrt_asym = asym.sqrt();

        // Index conventions: freq[k] where k ∈ {0=UHF, 1=HF, 2=MF, 3=LF};
        //                    freq[k][0]=X, freq[k][1]=Y, freq[k][2]=B.

        // UHF Y (use_lf=false, 9-tap = "_hf" kernel)
        let (g, l, n1) = malta_norm(W_UHF_MALTA * asym, W_UHF_MALTA / asym, NORM1_UHF, false);
        self.zero_plane(&self.block_diff_ac[1]);
        self.malta_hf(
            &self.freq_a[0][1],
            &self.freq_b[0][1],
            &self.block_diff_ac[1],
            g,
            l,
            n1,
        );

        // UHF X
        let (g, l, n1) = malta_norm(
            W_UHF_MALTA_X * asym,
            W_UHF_MALTA_X / asym,
            NORM1_UHF_X,
            false,
        );
        self.zero_plane(&self.block_diff_ac[0]);
        self.malta_hf(
            &self.freq_a[0][0],
            &self.freq_b[0][0],
            &self.block_diff_ac[0],
            g,
            l,
            n1,
        );

        // HF Y (use_lf=true, 5-tap = "_lf" kernel)
        let (g, l, n1) = malta_norm(
            W_HF_MALTA * sqrt_asym,
            W_HF_MALTA / sqrt_asym,
            NORM1_HF,
            true,
        );
        self.malta_lf(
            &self.freq_a[1][1],
            &self.freq_b[1][1],
            &self.block_diff_ac[1],
            g,
            l,
            n1,
        );

        // HF X
        let (g, l, n1) = malta_norm(
            W_HF_MALTA_X * sqrt_asym,
            W_HF_MALTA_X / sqrt_asym,
            NORM1_HF_X,
            true,
        );
        self.malta_lf(
            &self.freq_a[1][0],
            &self.freq_b[1][0],
            &self.block_diff_ac[0],
            g,
            l,
            n1,
        );

        // MF Y (symmetric, use_lf=true)
        let (g, l, n1) = malta_norm(W_MF_MALTA, W_MF_MALTA, NORM1_MF, true);
        self.malta_lf(
            &self.freq_a[2][1],
            &self.freq_b[2][1],
            &self.block_diff_ac[1],
            g,
            l,
            n1,
        );

        // MF X
        let (g, l, n1) = malta_norm(W_MF_MALTA_X, W_MF_MALTA_X, NORM1_MF_X, true);
        self.malta_lf(
            &self.freq_a[2][0],
            &self.freq_b[2][0],
            &self.block_diff_ac[0],
            g,
            l,
            n1,
        );

        // L2_asym on HF X (WMUL[0]) and HF Y (WMUL[1])
        self.l2_diff_asym(
            &self.freq_a[1][0],
            &self.freq_b[1][0],
            &self.block_diff_ac[0],
            (WMUL[0] as f32) * HF_ASYMMETRY,
            (WMUL[0] as f32) / HF_ASYMMETRY,
        );
        self.l2_diff_asym(
            &self.freq_a[1][1],
            &self.freq_b[1][1],
            &self.block_diff_ac[1],
            (WMUL[1] as f32) * HF_ASYMMETRY,
            (WMUL[1] as f32) / HF_ASYMMETRY,
        );
        // WMUL[2] = 0.0, skip HF B.

        // L2 on MF X (WMUL[3]) and MF Y (WMUL[4]) — accumulate.
        self.l2_diff(
            &self.freq_a[2][0],
            &self.freq_b[2][0],
            &self.block_diff_ac[0],
            WMUL[3] as f32,
        );
        self.l2_diff(
            &self.freq_a[2][1],
            &self.freq_b[2][1],
            &self.block_diff_ac[1],
            WMUL[4] as f32,
        );

        // L2 on MF B (WMUL[5]) — write-only (block_diff_ac[2] hasn't been touched yet).
        self.l2_diff_write(
            &self.freq_a[2][2],
            &self.freq_b[2][2],
            &self.block_diff_ac[2],
            WMUL[5] as f32,
        );
    }

    /// DC contributions: per-channel `WMUL[6+ch] · (LF_a[ch] − LF_b[ch])²`
    /// written into `block_diff_dc[ch]`. CPU folds this into
    /// `combine_channels_to_diffmap_fused`; we do it as a separate pass.
    fn compute_dc_diff(&self) {
        for ch in 0..3 {
            self.l2_diff_write(
                &self.freq_a[3][ch],
                &self.freq_b[3][ch],
                &self.block_diff_dc[ch],
                WMUL[6 + ch] as f32,
            );
        }
    }

    /// CPU `compute_mask_from_hf_uhf`: combine UHF+HF → diff_precompute →
    /// blur σ=2.7 → fuzzy_erosion. Also accumulates mask-to-error for Y.
    fn compute_mask_pipeline(&self) {
        // Image A: combine(UHF_a, HF_a) → temp1, diff_precompute(temp1) → mask_scratch
        unsafe {
            masking::combine_channels_for_masking_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.freq_a[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
            );
            masking::diff_precompute_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
            );
        }
        // Blur mask_scratch (image A combined+precompute) at σ=2.7, write into temp1.
        // (Use temp2 as the H→V scratch so temp1 stays clear for use afterwards.)
        self.blur_plane_via(
            &self.mask_scratch.clone(),
            &self.temp1.clone(),
            &self.temp2.clone(),
            MASK_RADIUS,
        );
        // temp1 now holds blurred_a. Run fuzzy_erosion(blurred_a) → mask.
        unsafe {
            masking::fuzzy_erosion_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask.clone(), self.n),
                self.width,
                self.height,
            );
        }

        // Image B: combine + diff_precompute + blur into mask_scratch.
        unsafe {
            masking::combine_channels_for_masking_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.freq_b[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
            );
            masking::diff_precompute_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
            );
        }
        // Blur image B's combined+precompute (temp2) into mask_scratch.
        // Use diffmap_buf as the H-pass scratch — it isn't written until
        // `launch_compute_diffmap` later in `compute()`.
        self.blur_plane_via(
            &self.temp2.clone(),
            &self.mask_scratch.clone(),
            &self.diffmap_buf.clone(),
            MASK_RADIUS,
        );

        // Mask-to-error contribution into block_diff_ac[1] (Y only):
        //   block_diff_ac[1] += MASK_TO_ERROR_MUL · (blurred_a − blurred_b)²
        unsafe {
            masking::mask_to_error_mul_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[1].clone(), self.n),
            );
        }
    }

    unsafe fn launch_compute_diffmap(&self) {
        unsafe {
            diffmap::compute_diffmap_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.mask.clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[0].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[1].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[2].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[0].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[1].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[2].clone(), self.n),
                ArrayArg::from_raw_parts(self.diffmap_buf.clone(), self.n),
            );
        }
    }

    fn malta_hf(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        norm2_0gt1: f32,
        norm2_0lt1: f32,
        norm1: f32,
    ) {
        unsafe {
            malta::malta_diff_map_hf_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_2d(),
                self.cube_dim_2d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                self.width,
                self.height,
                norm2_0gt1,
                norm2_0lt1,
                norm1,
            );
        }
    }

    fn malta_lf(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        norm2_0gt1: f32,
        norm2_0lt1: f32,
        norm1: f32,
    ) {
        unsafe {
            malta::malta_diff_map_lf_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_2d(),
                self.cube_dim_2d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                self.width,
                self.height,
                norm2_0gt1,
                norm2_0lt1,
                norm1,
            );
        }
    }

    fn l2_diff(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        weight: f32,
    ) {
        unsafe {
            diffmap::l2_diff_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                weight,
            );
        }
    }

    fn l2_diff_write(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        weight: f32,
    ) {
        unsafe {
            diffmap::l2_diff_write_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                weight,
            );
        }
    }

    fn l2_diff_asym(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        w_gt: f32,
        w_lt: f32,
    ) {
        unsafe {
            diffmap::l2_asym_diff_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                w_gt,
                w_lt,
            );
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Read the current diffmap back to host memory.
    pub fn copy_diffmap(&self) -> Vec<f32> {
        self.read_plane(&self.diffmap_buf)
    }

    fn read_plane(&self, h: &cubecl::server::Handle) -> Vec<f32> {
        let bytes = self.client.read_one(h.clone()).expect("read_one plane");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Debug: read the AC accumulator for one channel. Available after
    /// [`compute`].
    pub fn debug_block_diff_ac(&self, ch: usize) -> Vec<f32> {
        self.read_plane(&self.block_diff_ac[ch])
    }

    /// Debug: read the DC accumulator for one channel.
    pub fn debug_block_diff_dc(&self, ch: usize) -> Vec<f32> {
        self.read_plane(&self.block_diff_dc[ch])
    }

    /// Debug: read the fuzzy-erosion mask plane.
    pub fn debug_mask(&self) -> Vec<f32> {
        self.read_plane(&self.mask)
    }

    /// Debug: read one of the LF (low-frequency, vals-space) planes for
    /// one of the two image sides.
    pub fn debug_lf(&self, is_a: bool, ch: usize) -> Vec<f32> {
        let f = if is_a { &self.freq_a } else { &self.freq_b };
        self.read_plane(&f[3][ch])
    }

    /// Debug: read the per-channel HF / UHF / MF / LF plane (k ∈ 0..=3).
    pub fn debug_freq(&self, is_a: bool, k: usize, ch: usize) -> Vec<f32> {
        let f = if is_a { &self.freq_a } else { &self.freq_b };
        self.read_plane(&f[k][ch])
    }
}
