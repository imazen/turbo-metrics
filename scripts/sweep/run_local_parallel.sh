#!/usr/bin/env bash
# Parallel local sweep driver. Forks N worker processes that each pull
# a chunk off a shared work queue, run zen-metrics sweep on it, and
# upload the result to R2. Designed for the 7950X workstation where
# the GPU box was slower + flakier than running locally.
#
# Each zen-metrics process is single-threaded inside (the metrics use
# rayon internally but for typical 1-2MP CID22 images the parallel
# region is too small to saturate cores). Running 8 chunks in parallel
# scales nearly linearly until DRAM bandwidth saturates.

set -euo pipefail

BIN="${BIN:-$HOME/work/turbo-metrics/target/release/zen-metrics}"
CHUNK_FILE="${CHUNK_FILE:-/tmp/chunks.jsonl}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-2026-05-03}"
SOURCES_ROOT="${SOURCES_ROOT:-$HOME/work/zentrain-corpus/mlp-tune-fast}"
WORK="${WORK:-/tmp/sweep-local}"
PARALLEL="${PARALLEL:-8}"
mkdir -p "$WORK/out"

set -a
# shellcheck disable=SC1091
source "$HOME/.config/cloudflare/r2-credentials"
set +a
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"

# One-line worker. Reads a JSONL chunk on stdin and processes it.
process_chunk() {
    local line="$1"
    local codec chunk_id q_grid knob_grid metrics_args
    codec=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["codec"])')
    chunk_id=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["chunk_id"])')
    q_grid=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["q_grid"])')
    knob_grid=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["knob_grid"])')
    metrics_args=$(printf '%s' "$line" | python3 -c 'import sys, json; m=json.loads(sys.stdin.read())["metrics"]; print(" ".join(f"--metric {x}" for x in m))')

    # Skip if already on R2 (idempotent restart).
    if aws --endpoint-url "$R2_ENDPOINT" s3 ls \
        "s3://zentrain/${SWEEP_RUN_ID}/${codec}/${chunk_id}.tsv" \
        >/dev/null 2>&1
    then
        echo "[skip] $chunk_id already on R2"
        return 0
    fi

    local stage="$WORK/stage-${chunk_id}"
    rm -rf "$stage"; mkdir -p "$stage"
    SOURCES_ROOT="$SOURCES_ROOT" STAGE="$stage" python3 -c '
import sys, json, os
spec = json.loads(sys.stdin.read())
src_root = os.environ["SOURCES_ROOT"]
stage = os.environ["STAGE"]
for relpath in spec["images"]:
    src = os.path.join(src_root, relpath)
    flat = relpath.replace(os.sep, "__")
    dst = os.path.join(stage, flat)
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass
' <<<"$line"

    local out_tsv="$WORK/out/${chunk_id}.tsv"
    local start
    start=$(date +%s)
    # shellcheck disable=SC2086
    "$BIN" sweep \
        --codec "$codec" \
        --sources "$stage" \
        --q-grid "$q_grid" \
        --knob-grid "$knob_grid" \
        $metrics_args \
        --output "$out_tsv" \
        2>"$WORK/out/${chunk_id}.err" \
        || { echo "[fail] $chunk_id (see $WORK/out/${chunk_id}.err)"; rm -rf "$stage"; return 1; }
    local elapsed=$(( $(date +%s) - start ))

    aws --endpoint-url "$R2_ENDPOINT" s3 cp "$out_tsv" \
        "s3://zentrain/${SWEEP_RUN_ID}/${codec}/${chunk_id}.tsv" --quiet
    rm -rf "$stage"
    local rows
    rows=$(($(wc -l < "$out_tsv") - 1))
    echo "[done] $chunk_id ${elapsed}s ${rows}rows"
}
export -f process_chunk
export BIN R2_ENDPOINT AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
export SWEEP_RUN_ID SOURCES_ROOT WORK

# Feed chunks through xargs -P for parallel execution.
< "$CHUNK_FILE" xargs -d '\n' -I {} -P "$PARALLEL" \
    bash -c 'process_chunk "$@"' _ {}
