#!/usr/bin/env bash
# Worker bootstrap for the zen-metrics sweep on vast.ai.
#
# Designed to run inside a fresh GPU instance (Ubuntu 22.04 / 24.04 with
# CUDA-capable GPU). The script:
#   1. Installs system + rust toolchain dependencies (idempotent — if a
#      step is already done it is skipped).
#   2. Clones imazen/turbo-metrics at the requested ref.
#   3. Builds zen-metrics with `--features sweep,gpu,gpu-wgpu`.
#   4. Pulls source images from R2 into a local cache.
#   5. Iterates a jobspec list (one chunk per (codec, image-subset)),
#      runs the sweep, pushes Pareto TSV chunks back to R2.
#   6. Refreshes a heartbeat file in R2 so the dispatcher can tell the
#      worker is alive.
#
# Configuration via environment variables:
#   SWEEP_REF              Git ref to check out (default: zen-metrics-v0.3.0).
#   R2_ENDPOINT            CloudFlare R2 endpoint URL.
#   R2_ACCESS_KEY_ID
#   R2_SECRET_ACCESS_KEY   (above three are required)
#   SWEEP_RUN_ID           Run identifier (default: sweep-2026-05-03).
#   SWEEP_CHUNK_FILE       Local path to a JSONL file with one chunk per
#                          line. If unset, the worker pulls
#                          s3://coefficient/jobs/<run>/chunks.jsonl.
#   WORKER_ID              Optional human-readable id used in heartbeats
#                          (default: hostname).
#
# Each JSONL chunk is a single object:
#   {"codec": "zenwebp", "q_grid": "5,10,...,95",
#    "knob_grid": "{}", "metrics": ["zensim","ssim2"],
#    "images": ["cid22-train/foo.png", ...],
#    "chunk_id": "zenwebp-001"}
#
# Output for chunk `<id>` lands at:
#   s3://zentrain/<run>/<codec>/<chunk_id>.tsv

set -euo pipefail

SWEEP_REF="${SWEEP_REF:-zen-metrics-v0.3.0}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-2026-05-03}"
WORKER_ID="${WORKER_ID:-$(hostname)}"
WORKDIR="${WORKDIR:-/workspace/sweep}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

log() { printf '[sweep-worker %s %s] %s\n' "$(date -u +%H:%M:%S)" "$WORKER_ID" "$*"; }

# ── Step 1: system deps ────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1; then
    log "installing system deps"
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev cmake nasm git curl ca-certificates \
        clang libclang-dev libvulkan1 vulkan-tools mesa-vulkan-drivers awscli
    log "installing rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    export PATH="$HOME/.cargo/bin:$PATH"
fi
export PATH="$HOME/.cargo/bin:$PATH"

# ── Step 2: source checkout ───────────────────────────────────────────
if [[ ! -d turbo-metrics ]]; then
    log "cloning turbo-metrics @ $SWEEP_REF"
    git clone --depth 1 --branch "$SWEEP_REF" \
        https://github.com/imazen/turbo-metrics.git turbo-metrics
fi

# ── Step 3: build ─────────────────────────────────────────────────────
BIN="$WORKDIR/turbo-metrics/target/release/zen-metrics"
if [[ ! -x "$BIN" ]]; then
    log "building zen-metrics with sweep + gpu-wgpu"
    cd turbo-metrics
    cargo build --release -p zen-metrics-cli \
        --features "sweep,gpu,gpu-wgpu" 2>&1 | tail -30
    cd "$WORKDIR"
fi
"$BIN" --version

# ── R2 setup ──────────────────────────────────────────────────────────
: "${R2_ENDPOINT:?R2_ENDPOINT must be set}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID must be set}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY must be set}"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
S3() { aws --endpoint-url "$R2_ENDPOINT" "$@"; }

heartbeat() {
    local note="$1"
    local stamp
    stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"worker_id":"%s","stamp":"%s","note":"%s"}\n' \
        "$WORKER_ID" "$stamp" "$note" \
        > /tmp/heartbeat.json
    S3 s3 cp /tmp/heartbeat.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.json" \
        --quiet || true
}

heartbeat "starting"

# ── Step 4: source mirror ─────────────────────────────────────────────
SOURCES_DIR="$WORKDIR/sources"
if [[ ! -d "$SOURCES_DIR" ]]; then
    log "syncing sources from R2"
    mkdir -p "$SOURCES_DIR"
    S3 s3 sync "s3://zentrain/${SWEEP_RUN_ID}/sources/" "$SOURCES_DIR/" --no-progress
fi
SRC_COUNT=$(find "$SOURCES_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
log "have $SRC_COUNT source images"

# ── Step 5: chunk loop ────────────────────────────────────────────────
CHUNK_FILE="${SWEEP_CHUNK_FILE:-/tmp/chunks.jsonl}"
if [[ ! -f "$CHUNK_FILE" ]]; then
    log "fetching chunks list from R2"
    S3 s3 cp "s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl" "$CHUNK_FILE"
fi

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    codec=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["codec"])')
    chunk_id=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["chunk_id"])')
    q_grid=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["q_grid"])')
    knob_grid=$(printf '%s' "$line" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["knob_grid"])')
    metrics_args=$(printf '%s' "$line" | python3 -c 'import sys, json; m=json.loads(sys.stdin.read())["metrics"]; print(" ".join(f"--metric {x}" for x in m))')
    images=$(printf '%s' "$line" | python3 -c 'import sys, json; print("\n".join(json.loads(sys.stdin.read())["images"]))')

    # Skip if already done.
    OUT_KEY="s3://zentrain/${SWEEP_RUN_ID}/${codec}/${chunk_id}.tsv"
    if S3 s3 ls "$OUT_KEY" >/dev/null 2>&1; then
        log "skip ${chunk_id}: already done"
        continue
    fi

    # Stage just the chunk's images into a flat working dir.
    STAGE="$WORKDIR/stage-${chunk_id}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    while IFS= read -r relpath; do
        [[ -z "$relpath" ]] && continue
        # Replace path separators so multi-source-dir layouts collapse to
        # flat names (preserving uniqueness).
        flat="${relpath//\//__}"
        ln -sf "$SOURCES_DIR/$relpath" "$STAGE/$flat" || true
    done <<<"$images"

    OUT_TSV="$WORKDIR/out-${chunk_id}.tsv"
    log "running chunk ${chunk_id} (codec=${codec})"
    heartbeat "chunk ${chunk_id} ${codec}"

    # shellcheck disable=SC2086
    "$BIN" sweep \
        --codec "$codec" \
        --sources "$STAGE" \
        --q-grid "$q_grid" \
        --knob-grid "$knob_grid" \
        $metrics_args \
        --gpu-runtime wgpu \
        --output "$OUT_TSV" \
        2>&1 | tail -3

    log "uploading $OUT_TSV → $OUT_KEY"
    S3 s3 cp "$OUT_TSV" "$OUT_KEY" --quiet
    rm -rf "$STAGE"
    heartbeat "done ${chunk_id}"
done < "$CHUNK_FILE"

heartbeat "exhausted-chunks"
log "all chunks done"
