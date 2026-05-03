#!/usr/bin/env bash
# Dispatch the zen-metrics sweep to a vast.ai instance.
#
# Assumes:
#   * `vastai` CLI is configured.
#   * `~/.config/cloudflare/r2-credentials` exists and exports
#     R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ACCOUNT_ID.
#   * The instance contract id is passed as `$1`. SSH is reachable.
#
# Pushes the bootstrap script + R2 credentials to the instance and
# launches the sweep in tmux so an SSH disconnect doesn't kill it.

set -euo pipefail

INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" ]]; then
    echo "usage: $0 <instance_id>" >&2
    exit 64
fi

set -a
# shellcheck disable=SC1091
source "${HOME}/.config/cloudflare/r2-credentials"
set +a
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Pull SSH host/port from vastai metadata.
read -r SSH_HOST SSH_PORT < <(
    vastai show instances-v1 --raw \
        | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
if isinstance(data, dict) and 'instances' in data:
    data = data['instances']
for x in data:
    if str(x['id']) == '$INSTANCE_ID':
        print(x.get('ssh_host',''), x.get('ssh_port',''))
        break
"
)
if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    echo "could not find SSH host/port for instance $INSTANCE_ID" >&2
    exit 65
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$SSH_PORT")
SSH="ssh ${SSH_OPTS[*]} root@$SSH_HOST"

# Stage env file.
ENV_FILE=$(mktemp)
cat > "$ENV_FILE" <<EOF
export R2_ENDPOINT='$R2_ENDPOINT'
export R2_ACCESS_KEY_ID='$R2_ACCESS_KEY_ID'
export R2_SECRET_ACCESS_KEY='$R2_SECRET_ACCESS_KEY'
export SWEEP_REF='${SWEEP_REF:-zen-metrics-v0.3.0}'
export SWEEP_RUN_ID='${SWEEP_RUN_ID:-sweep-2026-05-03}'
export WORKER_ID='vastai-${INSTANCE_ID}'
EOF

scp "${SSH_OPTS[@]}" "$ENV_FILE" "root@${SSH_HOST}:/root/sweep.env"
scp "${SSH_OPTS[@]}" \
    "$(dirname "$0")/vastai_zen_metrics_sweep.sh" \
    "root@${SSH_HOST}:/root/vastai_zen_metrics_sweep.sh"
$SSH "chmod +x /root/vastai_zen_metrics_sweep.sh"

# Launch under tmux so the sweep survives an ssh disconnect.
$SSH "tmux new-session -d -s sweep 'set -a; source /root/sweep.env; set +a; bash /root/vastai_zen_metrics_sweep.sh 2>&1 | tee /root/sweep.log'"
echo "launched sweep on $SSH_HOST:$SSH_PORT under tmux session 'sweep'"
echo "tail with: ssh ${SSH_OPTS[*]} root@$SSH_HOST 'tail -f /root/sweep.log'"

rm -f "$ENV_FILE"
