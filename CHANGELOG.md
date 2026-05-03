# Changelog

## zen-metrics-cli

### [0.3.0] - 2026-05-03

#### Added
- New `sweep` subcommand and `sweep` cargo feature. Drives a codec
  (`zenwebp` / `zenavif` / `zenjxl`) across a Cartesian grid of
  `(image, q, knob_tuple)` cells, encodes each cell via the codec's
  public + `__expert` APIs, decodes back, and scores against the source
  with one or more selected metrics. Emits a Pareto TSV with one row
  per cell. The driver lives entirely outside any codec source tree —
  codecs stay "dumb" (no in-codec picker glue, no `.bin` shipped
  alongside the encoder). Replaces the per-codec example sweep
  harnesses scattered across coefficient and the encoder repos.
- New `tempfile`, `zencodec`, and `almost-enough` optional dependencies,
  pulled in by the `sweep` feature.
- Three integration tests covering sweep on each supported codec at
  64×64 (`sweep_zenwebp_emits_pareto_rows`, `sweep_zenavif_emits_pareto_rows`,
  `sweep_zenjxl_emits_pareto_rows`).
- Six unit tests under `sweep::grid::tests` for q-grid and knob-grid
  parsing and the canonical-JSON encoding rule that the TSV column
  uses.

#### Notes
- `zenjpeg 0.8.3` (the published version) does not yet expose
  `__expert`; it lands in 0.8.4. The sweep driver does not include
  zenjpeg in this release. It can be added once 0.8.4 publishes (or
  via a path override).
- `zenjxl 0.2.1`'s `JxlEncoderConfig` does not expose
  `with_internal_params` for the lossy path — the lossy
  `LossyInternalParams` are reachable via `jxl-encoder` directly but
  not via the wrapper. For now the sweep only varies zenjxl's public
  knobs (`effort`, `lossless`, `noise`, optional `distance`).

### [0.2.0] - 2026-05-03

#### Changed
- **BREAKING:** `butteraugli` (formerly `butteraugli-cpu`) and `butteraugli-gpu`
  now report the libjxl-style **3-norm** aggregation (`pnorm_3`) instead of
  the max-norm score. Matches `butteraugli_main --pnorm` and the Cloudinary
  CID22 paper. Both backends produce comparable scalars on identical inputs.
  (8c4d02f)
- **BREAKING:** Renamed CPU metric identifiers — drop the `-cpu` suffix.
  `ssim2-cpu` → `ssim2`, `butteraugli-cpu` → `butteraugli`. The `-gpu`
  suffix on `ssim2-gpu` and `butteraugli-gpu` is unchanged.

#### Added
- New `dssim` (CPU, via the canonical `dssim-core` v3.4 crate) and
  `dssim-gpu` (multi-vendor GPU via the workspace `dssim-gpu` crate)
  metrics. DSSIM is a distance metric — `0` means identical, larger
  values mean more distortion. The 0.2.0 metric set is exactly seven
  entries: `ssim2`, `ssim2-gpu`, `butteraugli`, `butteraugli-gpu`,
  `dssim`, `dssim-gpu`, `zensim`. The CPU-side `-cpu` suffix is dropped
  consistently with the rename above (no `dssim-cpu` alias).
- Cross-backend agreement test
  (`butteraugli_cpu_and_gpu_agree_on_3norm`) gated on
  `cpu-metrics + gpu-butteraugli` features. Verifies CPU and GPU
  butteraugli 3-norm scores agree within 5e-2 absolute tolerance on the
  64×64 noisy fixture.
- Tests for the new metrics: `score_dssim_identical_is_zero`,
  `score_dssim_noisy_higher_than_identical`,
  `score_dssim_gpu_identical_is_zero`.

### [0.1.1] - 2026-05-01

#### Fixed
- Release workflow: pin `-p zen-metrics-cli` on cargo build (30a751a)

### [0.1.0] - 2026-05-01

Initial release of the `zen-metrics` CLI.
