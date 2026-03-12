#!/bin/bash
# =============================================================================
# Shared config + helpers for all mini experiment scripts.
# Source with: source "$(dirname "$0")/_common.sh"
# =============================================================================

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

PROVIDER="modal"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TAX="./eval/finance/data/xbrl_taxonomy.json"

MAX_SAMPLES_ONLINE=75
MAX_SAMPLES_OFFLINE=150
NUM_EPOCHS=2
MAX_ROUNDS=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Returns 0 if the run directory contains a completed final_results.json
is_complete() {
    local save="$1"
    local f
    f=$(find "$save" -maxdepth 2 -name 'final_results.json' 2>/dev/null | head -1)
    [ -z "$f" ] && return 1
    python - "$f" <<'PYEOF' 2>/dev/null
import json, sys
with open(sys.argv[1]) as fh:
    d = json.load(fh)
def has_acc(obj):
    if isinstance(obj, dict):
        if 'accuracy' in obj and isinstance(obj['accuracy'], (int, float)):
            return True
        return any(has_acc(v) for v in obj.values())
    return False
sys.exit(0 if has_acc(d) else 1)
PYEOF
}

# Returns the partial run directory (has progress.json but no final_results.json)
find_partial_run() {
    local save="$1"
    local dir
    while IFS= read -r pf; do
        dir=$(dirname "$pf")
        if [ ! -f "$dir/final_results.json" ]; then
            echo "$dir"
            return
        fi
    done < <(find "$save" -maxdepth 2 -name 'progress.json' 2>/dev/null)
}

# Move to project root regardless of where the script is called from
cd /c/Users/Window/Desktop/gsam-rsh
mkdir -p results/mini
