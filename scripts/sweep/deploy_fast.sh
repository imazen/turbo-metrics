#!/usr/bin/env bash
# Fan out N vast.ai workers using the pre-baked GHCR image.
#
# Compared to the legacy `vastai_zen_metrics_sweep.sh` flow, this skips
# the in-instance source build entirely and starts processing chunks
# within ~30s of `vastai create instance` returning.
#
# Usage:
#   N_BOXES=8 SWEEP_RUN_ID=sweep-v04-2026-05-04 bash deploy_fast.sh
#
# Env:
#   N_BOXES                 number of workers to launch (default 8)
#   SWEEP_RUN_ID            run id, e.g. sweep-v04-2026-05-04 (required)
#   IMAGE_TAG               GHCR tag (default: 0.3.0)
#   MAX_DPH                 max $/hr per box (default 0.07)
#   MIN_CORES               min CPU cores per box (default 12)
#   MIN_RAM_MB              min RAM MB (default 16000)
#   MIN_DISK_GB             min disk GB (default 40)
#   GPU_RUNTIME             auto | wgpu | cuda | hip | cpu (default auto)
#   DRY_RUN                 set to 1 to print plan only

set -euo pipefail

N_BOXES="${N_BOXES:-8}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-}"
IMAGE_TAG="${IMAGE_TAG:-0.3.0}"
MAX_DPH="${MAX_DPH:-0.07}"
MIN_CORES="${MIN_CORES:-12}"
MIN_RAM_MB="${MIN_RAM_MB:-16000}"
MIN_DISK_GB="${MIN_DISK_GB:-40}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
DRY_RUN="${DRY_RUN:-0}"

[[ -z "$SWEEP_RUN_ID" ]] && {
    echo "SWEEP_RUN_ID must be set (e.g. sweep-v04-2026-05-04)" >&2
    exit 64
}

set -a
# shellcheck disable=SC1091
source "$HOME/.config/cloudflare/r2-credentials"
set +a

# Sanity-check that the image is reachable. Fail fast rather than launch
# 8 boxes that all fail to pull.
IMAGE="ghcr.io/imazen/zen-metrics-sweep:${IMAGE_TAG}"
GHCR_AUTH=""
if [[ -n "${GHCR_TOKEN:-}" ]]; then
    # Authenticated check (use this when the package is private).
    if ! curl -fsSL "https://ghcr.io/v2/imazen/zen-metrics-sweep/manifests/${IMAGE_TAG}" \
        -u "${GHCR_USER:-imazen}:${GHCR_TOKEN}" \
        -o /dev/null 2>/dev/null
    then
        echo "WARNING: cannot verify $IMAGE on GHCR with provided token." >&2
    else
        GHCR_AUTH="-e GHCR_TOKEN=${GHCR_TOKEN} -e GHCR_USER=${GHCR_USER:-imazen}"
    fi
else
    # Anonymous check — works only when the GHCR package is public.
    if ! curl -fsSL "https://ghcr.io/v2/imazen/zen-metrics-sweep/manifests/${IMAGE_TAG}" \
        -o /dev/null 2>/dev/null
    then
        echo "WARNING: $IMAGE is private (or unreachable). Set GHCR_TOKEN to a PAT with read:packages, OR make the package public via:" >&2
        echo "  https://github.com/orgs/imazen/packages/container/zen-metrics-sweep/settings (Change visibility -> Public)" >&2
    fi
fi

QUERY="rentable=true verified=true reliability>0.95 cuda_max_good>=11.0 num_gpus=1 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_MB} disk_space>${MIN_DISK_GB} inet_down>=200 inet_up>=50"

echo "[deploy] querying offers: $QUERY"
OFFERS_JSON=$(vastai search offers "$QUERY" --order 'dph_total' --raw)

OFFER_IDS=$(echo "$OFFERS_JSON" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
offers = d if isinstance(d, list) else d.get('offers', [])
seen = set()
picked = []
for o in offers:
    mid = o.get('machine_id')
    if mid in seen: continue
    seen.add(mid)
    picked.append(str(o['id']))
    if len(picked) >= int('$N_BOXES'): break
print('\n'.join(picked))
")

n=$(echo "$OFFER_IDS" | wc -w)
echo "[deploy] picked $n distinct offers (need $N_BOXES)"
[[ "$DRY_RUN" == "1" ]] && { echo "$OFFER_IDS"; exit 0; }
[[ "$n" -lt 2 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

INSTANCE_FILE="/tmp/v04_fast_instances.txt"
> "$INSTANCE_FILE"

i=0
for offer_id in $OFFER_IDS; do
    i=$((i+1))
    WORKER_ID="${SWEEP_RUN_ID}-fast-${i}"
    LABEL="zen-fast-${SWEEP_RUN_ID#sweep-}-${i}"
    ENV_STR="-e R2_ACCOUNT_ID=$R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY -e SWEEP_RUN_ID=$SWEEP_RUN_ID -e WORKER_ID=$WORKER_ID -e SWEEP_GPU_RUNTIME=$GPU_RUNTIME"

    echo "[deploy] launching $i/$n worker=$WORKER_ID offer=$offer_id"
    OUT=$(vastai create instance "$offer_id" \
        --image "$IMAGE" \
        --disk "$MIN_DISK_GB" \
        --label "$LABEL" \
        --env "$ENV_STR" \
        --raw 2>&1) || { echo "  failed: $OUT"; continue; }

    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    if [[ -z "$ID" ]]; then
        echo "  could not parse instance id: $OUT"
        continue
    fi
    echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
    echo "  -> instance $ID"
done

echo
echo "[deploy] launched $(wc -l < "$INSTANCE_FILE") instances:"
cat "$INSTANCE_FILE"
echo
echo "[deploy] watch progress with:"
echo "  watch -n 30 'vastai show instances-v1 --raw | python3 -c \"import json,sys;d=json.load(sys.stdin);print(len(d if isinstance(d,list) else d.get(\\\"instances\\\",[])))\"'"
echo "  aws --endpoint-url \"https://\${R2_ACCOUNT_ID}.r2.cloudflarestorage.com\" s3 ls s3://zentrain/$SWEEP_RUN_ID/zenavif/ | wc -l"
echo "  aws --endpoint-url \"https://\${R2_ACCOUNT_ID}.r2.cloudflarestorage.com\" s3 ls s3://zentrain/$SWEEP_RUN_ID/zenjxl/ | wc -l"
