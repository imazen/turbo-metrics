#!/usr/bin/env bash
# Local-first sweep driver. Iterates the same JSONL chunks the vastai
# worker would pull and runs each through the locally-built zen-metrics
# binary. Pushes Pareto TSVs to R2 as it goes so a partial run is still
# durable.
#
# Why a local-first option exists: vast.ai SSH provisioning has been
# unreliable in this run (instances stuck in `loading` for several
# minutes; ssh keys not propagating cleanly to the proxy). Local
# encoding with zen-metrics on a strong box (Ryzen 7950X here) is
# 2-3x faster than the cheap RTX 3060 boxes in the available offers
# anyway, since the 4 selected metrics are all CPU-bound.

set -euo pipefail

BIN="${BIN:-$HOME/work/turbo-metrics/target/release/zen-metrics}"
CHUNK_FILE="${CHUNK_FILE:-/tmp/chunks.jsonl}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-2026-05-03}"
SOURCES_ROOT="${SOURCES_ROOT:-$HOME/work/zentrain-corpus/mlp-tune-fast}"
WORK="${WORK:-/tmp/sweep-local}"
mkdir -p "$WORK/chunks" "$WORK/out"

set -a
# shellcheck disable=SC1091
source "$HOME/.config/cloudflare/r2-credentials"
set +a
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"

S3() { aws --endpoint-url "$R2_ENDPOINT" "$@"; }

log() { printf '[local-sweep %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Skip chunks that already have a result on R2 (lets us interrupt and
# resume without re-running anything).
chunk_exists_on_r2() {
    local codec="$1" chunk_id="$2"
    S3 s3 ls "s3://zentrain/${SWEEP_RUN_ID}/${codec}/${chunk_id}.tsv" >/dev/null 2>&1
}

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    codec=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["codec"])')
    chunk_id=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["chunk_id"])')
    q_grid=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["q_grid"])')
    knob_grid=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["knob_grid"])')
    metrics_args=$(printf '%s' "$line" | python3 -c 'import sys, json; m=json.loads(sys.stdin.read())["metrics"]; print(" ".join(f"--metric {x}" for x in m))')

    if chunk_exists_on_r2 "$codec" "$chunk_id"; then
        log "skip ${chunk_id}: already on R2"
        continue
    fi

    STAGE="$WORK/chunks/${chunk_id}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    printf '%s' "$line" | python3 -c '
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
'

    OUT_TSV="$WORK/out/${chunk_id}.tsv"
    log "running chunk ${chunk_id} (codec=${codec}) — $(ls "$STAGE" | wc -l) imgs"
    # shellcheck disable=SC2086
    "$BIN" sweep \
        --codec "$codec" \
        --sources "$STAGE" \
        --q-grid "$q_grid" \
        --knob-grid "$knob_grid" \
        $metrics_args \
        --output "$OUT_TSV" \
        2>&1 | tail -1

    log "uploading $OUT_TSV → R2"
    S3 s3 cp "$OUT_TSV" "s3://zentrain/${SWEEP_RUN_ID}/${codec}/${chunk_id}.tsv" --quiet
done < "$CHUNK_FILE"

log "all chunks done"
