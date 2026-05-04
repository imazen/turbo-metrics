# Changelog

## zen-metrics-cli

### [0.5.0] - 2026-05-04

#### Changed (BREAKING)
- `MetricKind` no longer has the `ButteraugliMax` / `ButteraugliMaxGpu`
  variants, and the corresponding `--metric butteraugli-max` /
  `--metric butteraugli-max-gpu` CLI values are removed. The `butteraugli`
  and `butteraugli-gpu` metrics now emit BOTH aggregations from a single
  `compute()` call:
  - `butteraugli` → columns `butteraugli_max` + `butteraugli_pnorm3`
  - `butteraugli-gpu` → columns `butteraugli_max_gpu` + `butteraugli_pnorm3_gpu`
  Sweeps that previously declared both `butteraugli` and `butteraugli-max`
  to capture both aggregations now declare just `butteraugli` and pay
  half the cost.
- `metrics::run_metric` returns `Vec<(&'static str, f64)>` instead of
  `f64` — one `(column_name, value)` pair per emitted column.
- `MetricKind::column_name` (singular) is replaced by
  `MetricKind::column_names` (plural, returns `&'static [&'static str]`)
  so callers can iterate the columns a metric emits.
- `score` subcommand JSON output now nests values under `scores.<column>`
  instead of a top-level `score` field. Single-column metrics still have
  a single key under `scores` (e.g. `scores.zensim`); butteraugli has two.
- `score` subcommand TSV output uses column names (e.g.
  `butteraugli_max\tbutteraugli_pnorm3`) as the header row instead of the
  fixed `metric\tscore` shape.
- `compare` subcommand TSV / JSON / plain output expands butteraugli into
  its two columns, mirroring the `score` change.

#### Notes
- The `score_butteraugli` column that pre-0.5.0 sweep TSVs carried no
  longer exists. New sweeps emit `score_butteraugli_max` and
  `score_butteraugli_pnorm3` (or the `_gpu`-suffixed pair). Loaders that
  hardcoded `score_butteraugli` need to switch to one of the new
  columns — `score_butteraugli_pnorm3` matches the old number bit-for-bit.

### [0.4.0] - 2026-05-04

#### Added
- `--feature-output <path.parquet>` on the `sweep` subcommand. When set,
  every cell that runs the `zensim` metric also persists its 300-feature
  extended vector to the parquet file at `path`. Joins back to the existing
  TSV by `(image_path, codec, q, knob_tuple_json)`. The parquet schema is
  `image_path:utf8, codec:utf8, q:uint32, knob_tuple_json:utf8,
  zensim_score:float32, feat_0..feat_299:float32`. Compressed with zstd-3.
  Buffered into 256-row Arrow batches before flushing to keep memory
  bounded for million-cell sweeps.
- New `parquet`, `arrow-array`, `arrow-schema` optional dependencies,
  pulled in by the `sweep` cargo feature so non-sweep builds don't ship
  the Arrow stack.
- Bump `zensim` minimum to `0.2.8` for the new
  `Zensim::compute_extended_features()` API used by the parquet writer.

#### Notes
- The numeric `score_zensim` column in the TSV is unchanged — the extra
  72 masked features have zero weight in the trained profile, so adding
  them changes neither the weighted distance nor the score.
- The sweep onstart script (`scripts/sweep/onstart_v3.sh`) now uploads
  per-chunk `features-${chunk_id}.parquet` files to R2 alongside the
  existing TSV. Finalize / consolidate steps need to concatenate the
  per-chunk parquets if they want a single sweep-wide file.

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
