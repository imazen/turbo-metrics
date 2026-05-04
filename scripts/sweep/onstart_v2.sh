#!/usr/bin/env bash
# v2 onstart for vast.ai zen-metrics sweep workers.
#
# Improvements over vastai_zen_metrics_sweep.sh:
#   1. Pulls the pre-built x86_64 binary from a tagged GitHub release
#      instead of building from source. Saves 10-20 min per box.
#   2. Atomic R2-based chunk claim (PUT-If-None-Match) so multiple
#      workers cannot redundantly process the same chunk.
#   3. Multi-chunk parallelism inside the box via xargs -P (defaults to
#      max(2, nproc/4) — leaves cores for the codec's internal rayon).
#   4. Periodic stats reporter (cpu, mem, gpu_util, gpu_mem, rows_done)
#      pushed to R2 every 60s for cheap centralised monitoring.
#   5. Final-state sentinel file on completion + on graceful exit.
#
# Required environment:
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID                e.g. sweep-v04-2026-05-04
#   SWEEP_BIN_VERSION           e.g. zen-metrics-v0.3.0  (defaults below)
# Optional:
#   WORKER_ID                   default: hostname
#   WORKER_PARALLEL             default: max(2, nproc/4)
#   SWEEP_BIN_OVERRIDE          full URL to download instead of release
#   SWEEP_GPU_RUNTIME           wgpu | cuda | hip | cpu | auto (default)
#   STATS_INTERVAL_SEC          default 60
#
# Sentinel files (R2):
#   coefficient/heartbeats/<run>/<worker>.json     periodic
#   coefficient/heartbeats/<run>/<worker>.done     final OK
#   coefficient/heartbeats/<run>/<worker>.failed   final non-OK
#   coefficient/heartbeats/<run>/stats/<worker>.tsv  rolling stats

set -euo pipefail

SWEEP_BIN_VERSION="${SWEEP_BIN_VERSION:-zen-metrics-v0.3.0}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-2026-05-04}"
WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
WORKDIR="${WORKDIR:-/workspace/sweep}"
PARALLEL="${WORKER_PARALLEL:-}"
GPU_RUNTIME="${SWEEP_GPU_RUNTIME:-auto}"
STATS_INTERVAL_SEC="${STATS_INTERVAL_SEC:-60}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ -z "$PARALLEL" ]]; then
    nc=$(nproc 2>/dev/null || echo 8)
    PARALLEL=$(( nc > 8 ? nc / 4 : 2 ))
fi

log() { printf '[onstart-v2 %s %s] %s\n' "$(date -u +%H:%M:%S)" "$WORKER_ID" "$*"; }

trap 'final_state failed "$LINENO" "$BASH_COMMAND"' ERR

final_state() {
    local kind="${1:-done}"; local lineno="${2:-}"; local cmd="${3:-}"
    local stamp; stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"worker":"%s","stamp":"%s","kind":"%s","line":"%s","cmd":"%s"}\n' \
        "$WORKER_ID" "$stamp" "$kind" "$lineno" "$cmd" \
        > /tmp/final.json
    S3 s3 cp /tmp/final.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.${kind}" \
        --quiet 2>/dev/null || true
    log "FINAL: $kind"
    exit 0
}

# ── Step 1: minimal system deps (no rust toolchain) ─────────────────
if ! command -v aws >/dev/null 2>&1; then
    log "installing awscli + vulkan runtime"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        awscli ca-certificates curl jq libvulkan1 mesa-vulkan-drivers \
        2>&1 | tail -5
fi

# ── R2 setup ────────────────────────────────────────────────────────
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID must be set}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID must be set}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY must be set}"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=auto
S3() { aws --endpoint-url "$R2_ENDPOINT" "$@"; }

# ── Step 2: pre-built binary ────────────────────────────────────────
BIN="$WORKDIR/zen-metrics"
if [[ ! -x "$BIN" ]]; then
    arch=$(uname -m)
    case "$arch" in
        x86_64)  pkg="zen-metrics-${SWEEP_BIN_VERSION#zen-metrics-v}-linux-x86_64.tar.gz" ;;
        aarch64) pkg="zen-metrics-${SWEEP_BIN_VERSION#zen-metrics-v}-linux-aarch64.tar.gz" ;;
        *) log "FATAL: unsupported arch $arch"; final_state failed ;;
    esac
    URL="${SWEEP_BIN_OVERRIDE:-https://github.com/imazen/turbo-metrics/releases/download/${SWEEP_BIN_VERSION}/${pkg}}"
    log "downloading $URL"
    curl -fsSL "$URL" -o /tmp/zm.tgz
    tar xzf /tmp/zm.tgz -C "$WORKDIR" --strip-components=1 \
        "$(tar tzf /tmp/zm.tgz | grep -E '/zen-metrics$' | head -1)"
    chmod +x "$BIN"
fi
"$BIN" --version

# ── Step 3: heartbeat ──────────────────────────────────────────────
heartbeat() {
    local note="$1"
    local stamp; stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"worker":"%s","stamp":"%s","note":"%s","parallel":%d,"runtime":"%s"}\n' \
        "$WORKER_ID" "$stamp" "$note" "$PARALLEL" "$GPU_RUNTIME" \
        > /tmp/heartbeat.json
    S3 s3 cp /tmp/heartbeat.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.json" \
        --quiet 2>/dev/null || true
}
heartbeat "starting"

# ── Step 4: stats reporter (background) ────────────────────────────
stats_loop() {
    local started; started=$(date +%s)
    local stats_file=/tmp/stats.tsv
    printf 'ts_utc\tload1\tcpu_pct\tmem_used_mib\tmem_total_mib\tgpu_util\tgpu_mem_mib\tgpu_temp\trows_done\twall_min\n' > "$stats_file"
    while true; do
        local now ts load1 cpu_pct mem_used mem_total gpu_util gpu_mem gpu_temp rows wall
        now=$(date +%s)
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        load1=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
        cpu_pct=$(top -bn1 -d 0.5 2>/dev/null | awk '/Cpu/ { gsub(/,/, ""); print 100 - $8 }' | head -1 | awk '{printf "%.1f", $1}')
        [[ -z "$cpu_pct" ]] && cpu_pct=0
        mem_used=$(free -m 2>/dev/null | awk '/^Mem:/ {print $3}')
        mem_total=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2}')
        if command -v nvidia-smi >/dev/null 2>&1; then
            read -r gpu_util gpu_mem gpu_temp < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F, '{gsub(/ /,""); print $1, $2, $3}')
        fi
        gpu_util="${gpu_util:-0}"; gpu_mem="${gpu_mem:-0}"; gpu_temp="${gpu_temp:-0}"
        rows=$(cat /tmp/rows_done 2>/dev/null || echo 0)
        wall=$(awk -v s=$started -v n=$now 'BEGIN{printf "%.1f", (n-s)/60.0}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$ts" "$load1" "$cpu_pct" "$mem_used" "$mem_total" "$gpu_util" "$gpu_mem" "$gpu_temp" "$rows" "$wall" \
            >> "$stats_file"
        S3 s3 cp "$stats_file" \
            "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/stats/${WORKER_ID}.tsv" \
            --quiet 2>/dev/null || true
        sleep "$STATS_INTERVAL_SEC"
    done
}
stats_loop > /tmp/stats.log 2>&1 &
STATS_PID=$!
log "stats_loop pid=$STATS_PID interval=${STATS_INTERVAL_SEC}s"

# ── Step 5: source mirror ──────────────────────────────────────────
SOURCES_DIR="$WORKDIR/sources"
if [[ ! -d "$SOURCES_DIR" ]] || [[ -z "$(find "$SOURCES_DIR" -type f -print -quit 2>/dev/null)" ]]; then
    log "syncing sources from R2"
    mkdir -p "$SOURCES_DIR"
    S3 s3 sync "s3://zentrain/${SWEEP_RUN_ID}/sources/" "$SOURCES_DIR/" --no-progress 2>&1 | tail -3
fi
SRC_COUNT=$(find "$SOURCES_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
log "have $SRC_COUNT source images, parallelism=$PARALLEL"

# ── Step 6: chunks list ────────────────────────────────────────────
CHUNK_FILE="${SWEEP_CHUNK_FILE:-/tmp/chunks.jsonl}"
if [[ ! -f "$CHUNK_FILE" ]]; then
    log "fetching chunks list from R2"
    S3 s3 cp "s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl" "$CHUNK_FILE"
fi
TOTAL_CHUNKS=$(wc -l < "$CHUNK_FILE")
log "loaded $TOTAL_CHUNKS chunks"

# ── Step 7: chunk processor (atomic claim + run + upload) ──────────
echo 0 > /tmp/rows_done
process_chunk() {
    local line="$1"
    local codec chunk_id q_grid knob_grid metrics_args images
    codec=$(printf '%s' "$line" | jq -r '.codec')
    chunk_id=$(printf '%s' "$line" | jq -r '.chunk_id')
    q_grid=$(printf '%s' "$line" | jq -r '.q_grid')
    knob_grid=$(printf '%s' "$line" | jq -r '.knob_grid')
    metrics_args=$(printf '%s' "$line" | jq -r '.metrics | map("--metric " + .) | join(" ")')

    local OUT_KEY="s3://zentrain/${SWEEP_RUN_ID}/${codec}/${chunk_id}.tsv"
    local CLAIM_KEY="coefficient/claims/${SWEEP_RUN_ID}/${chunk_id}.claim"

    # Cheap pre-check.
    if S3 s3 ls "$OUT_KEY" >/dev/null 2>&1; then
        echo "[skip] $chunk_id already done"
        return 0
    fi

    # Atomic claim via R2 PUT-If-None-Match. Only one worker can succeed.
    local claim_body="/tmp/claim-${chunk_id}.txt"
    printf '{"worker":"%s","stamp":"%s","ttl_sec":1800}\n' \
        "$WORKER_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$claim_body"
    if ! S3 s3api put-object \
            --bucket coefficient --key "${CLAIM_KEY}" \
            --body "$claim_body" \
            --if-none-match '*' >/dev/null 2>&1; then
        # Claim exists. Check age — overwrite if stale (>30 min).
        local claim_age
        claim_age=$(S3 s3api head-object --bucket coefficient --key "${CLAIM_KEY}" \
            --query LastModified --output text 2>/dev/null || echo "")
        if [[ -n "$claim_age" ]]; then
            local age_sec
            age_sec=$(date -d "$claim_age" +%s 2>/dev/null || echo 0)
            local now_sec; now_sec=$(date +%s)
            if (( now_sec - age_sec > 1800 )); then
                # Stale: force overwrite (no If-None-Match).
                S3 s3api put-object \
                    --bucket coefficient --key "${CLAIM_KEY}" \
                    --body "$claim_body" >/dev/null 2>&1 || true
                echo "[stale-claim] $chunk_id taken over"
            else
                echo "[claimed-elsewhere] $chunk_id"
                return 0
            fi
        else
            echo "[claimed-elsewhere] $chunk_id"
            return 0
        fi
    fi

    # Re-check (small window where another worker uploaded after our pre-check).
    if S3 s3 ls "$OUT_KEY" >/dev/null 2>&1; then
        echo "[skip-after-claim] $chunk_id"
        return 0
    fi

    local STAGE="$WORKDIR/stage-${chunk_id}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    printf '%s' "$line" | jq -r '.images[]' | while IFS= read -r relpath; do
        [[ -z "$relpath" ]] && continue
        flat="${relpath//\//__}"
        ln -sf "$SOURCES_DIR/$relpath" "$STAGE/$flat" 2>/dev/null || true
    done

    local OUT_TSV="$WORKDIR/out-${chunk_id}.tsv"
    local start_t; start_t=$(date +%s)
    # shellcheck disable=SC2086
    if "$BIN" sweep \
        --codec "$codec" \
        --sources "$STAGE" \
        --q-grid "$q_grid" \
        --knob-grid "$knob_grid" \
        $metrics_args \
        --gpu-runtime "$GPU_RUNTIME" \
        --output "$OUT_TSV" \
        > "/tmp/sweep-${chunk_id}.log" 2>&1
    then
        local elapsed=$(( $(date +%s) - start_t ))
        local rows
        rows=$(($(wc -l < "$OUT_TSV") - 1))
        S3 s3 cp "$OUT_TSV" "$OUT_KEY" --quiet
        # Update rows_done atomically.
        ( flock -x 200; echo $(( $(cat /tmp/rows_done) + rows )) > /tmp/rows_done ) 200>/tmp/rows_done.lock
        echo "[done] $chunk_id ${elapsed}s ${rows}rows"
    else
        echo "[fail] $chunk_id (see /tmp/sweep-${chunk_id}.log)"
        S3 s3 cp "/tmp/sweep-${chunk_id}.log" \
            "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/errors/${chunk_id}.log" \
            --quiet 2>/dev/null || true
    fi
    rm -rf "$STAGE" "$OUT_TSV"
}
export -f process_chunk
export BIN R2_ENDPOINT AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION
export SWEEP_RUN_ID WORKDIR SOURCES_DIR WORKER_ID GPU_RUNTIME

# Shuffle for diversity, then xargs -P for parallelism within the box.
SEED=$(echo -n "$WORKER_ID" | md5sum | cut -c1-16)
shuf < "$CHUNK_FILE" \
    --random-source=<(yes "$SEED" | tr -d '\n' | head -c 32768) 2>/dev/null \
    > "$WORKDIR/chunks_shuf.jsonl" || cp "$CHUNK_FILE" "$WORKDIR/chunks_shuf.jsonl"

heartbeat "running"
log "starting xargs -P $PARALLEL"
< "$WORKDIR/chunks_shuf.jsonl" xargs -d '\n' -I {} -P "$PARALLEL" \
    bash -c 'process_chunk "$@"' _ {}

# ── Final ──────────────────────────────────────────────────────────
heartbeat "exhausted-chunks"
log "all assigned chunks done; rows=$(cat /tmp/rows_done 2>/dev/null || echo 0)"
final_state done
