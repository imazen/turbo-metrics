#!/usr/bin/env bash
# Post-sweep finalization. Pulls all per-chunk Pareto TSVs from R2,
# concatenates per codec, writes a summary manifest, and pushes the
# manifest back to R2.

set -euo pipefail

SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-2026-05-03}"
WORK="${WORK:-/tmp/sweep-finalize}"
mkdir -p "$WORK"

set -a
# shellcheck disable=SC1091
source "$HOME/.config/cloudflare/r2-credentials"
set +a
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"

S3() { aws --endpoint-url "$R2_ENDPOINT" "$@"; }

for codec in zenwebp zenavif zenjxl; do
    echo "[finalize] pulling $codec chunks"
    mkdir -p "$WORK/$codec"
    S3 s3 sync "s3://zentrain/${SWEEP_RUN_ID}/${codec}/" "$WORK/$codec/" --no-progress

    OUT="$WORK/${codec}_pareto_${SWEEP_RUN_ID}.tsv"
    rm -f "$OUT"
    # Concatenate: keep header from first file only.
    HEADER_DONE=0
    for f in "$WORK/$codec"/*.tsv; do
        [[ -f "$f" ]] || continue
        if [[ "$HEADER_DONE" == 0 ]]; then
            head -1 "$f" > "$OUT"
            HEADER_DONE=1
        fi
        tail -n +2 "$f" >> "$OUT"
    done
    rows=$(($(wc -l < "$OUT") - 1))
    echo "[finalize] $codec → $OUT (${rows} rows)"
    S3 s3 cp "$OUT" "s3://zentrain/${SWEEP_RUN_ID}/${codec}_pareto_concat.tsv" --quiet
done

# Manifest with per-codec row counts and timestamps.
WORK="$WORK" SWEEP_RUN_ID="$SWEEP_RUN_ID" python3 - <<'EOF' > "$WORK/manifest.json"
import os, json, glob, datetime
work = os.environ["WORK"]
run = os.environ["SWEEP_RUN_ID"]
out = {"run_id": run, "generated_at": datetime.datetime.utcnow().isoformat() + "Z", "codecs": {}}
for codec in ("zenwebp", "zenavif", "zenjxl"):
    chunks = sorted(glob.glob(os.path.join(work, codec, "*.tsv")))
    total_rows = 0
    for f in chunks:
        with open(f) as h:
            total_rows += sum(1 for _ in h) - 1
    out["codecs"][codec] = {
        "chunks": len(chunks),
        "rows": total_rows,
        "concat_path": f"s3://zentrain/{run}/{codec}_pareto_concat.tsv",
    }
print(json.dumps(out, indent=2))
EOF

cat "$WORK/manifest.json"
S3 s3 cp "$WORK/manifest.json" "s3://zentrain/${SWEEP_RUN_ID}/_manifest.json" --quiet
echo "[finalize] manifest at s3://zentrain/${SWEEP_RUN_ID}/_manifest.json"
