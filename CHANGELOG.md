# Changelog

## zen-metrics-cli

### [0.4.1] - 2026-05-04

#### Added
- New `features-backfill` subcommand. Re-derives the per-cell zensim
  300-feature parquet sidecar from a sweep TSV that was produced by
  `zen-metrics-cli` 0.3.0 (which lacked `--feature-output`). Reads the
  TSV row-by-row, re-encodes each `(image, q, knob_tuple_json)` cell
  through the same codec dispatch the `sweep` subcommand uses, runs
  `zensim::compute_extended_features()`, and writes a parquet matching
  the 0.4.0 schema. The TSV is opened read-only — it is never modified.
  Two modes:
  - **Local**: `--input-tsv <chunk.tsv> --output-parquet <chunk.parquet>
    --corpus-root <dir>` — single-chunk, used by tests and ad-hoc reruns.
  - **R2**: `--run-id <id> --codec <codec> --corpus-root <dir>` — walks
    every chunk TSV under `s3://zentrain/<run-id>/<codec>/`, skips chunks
    whose feature parquet is already uploaded, and uploads new parquets
    via a staging key + server-side rename for atomicity.
  Idempotent at both levels: pre-existing parquets are skipped, and a
  partial run only touches the staging key, never the final destination.
  Image paths in the TSV (which were written by the sweep worker against
  a flattened staging dir, e.g. `dir__sub__file.png`) are resolved against
  `--corpus-root` via an unflatten heuristic with a basename-walk
  fallback.
- 8 unit tests + 5 integration tests covering local mode, idempotence,
  unflattened path resolution, and CLI argument validation.

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
