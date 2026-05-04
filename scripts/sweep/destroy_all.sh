#!/usr/bin/env bash
# Destroy all vast.ai instances matching a label prefix.
# Usage: bash destroy_all.sh zen-sweep-v04
#        bash destroy_all.sh zen-fast-v04
set -euo pipefail
PREFIX="${1:-zen-sweep-v04}"
IDS=$(vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
ins = d if isinstance(d, list) else d.get('instances', [])
for i in ins:
    if i.get('label','').startswith('$PREFIX'):
        print(i['id'])
")
if [[ -z "$IDS" ]]; then
    echo "No instances matching prefix=$PREFIX"
    exit 0
fi
echo "Will destroy:"
echo "$IDS"
[[ "${YES:-0}" == "1" ]] || { read -rp "Confirm destroy [y/N]? " ans; [[ "$ans" == "y" ]] || exit 1; }
echo "$IDS" | xargs -n1 vastai destroy instance --raw 2>&1 | tail -20
