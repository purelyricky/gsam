#!/bin/bash
# =============================================================================
# GSAM experiment runner — minimum set to fill all thesis tables
#
# Mapping to paper tables
# ─────────────────────────────────────────────────────────────────────────────
#  Table 1  (main results)    : GSAM FiNER/Formula online + offline   [4 runs]
#  Table 2  (ablation)        : 5 GSAM ablations on FiNER online      [5 runs]
#                               + no-multi-epoch on FiNER offline      [1 run]
#  Table 3  (FiNER-Transfer)  : ACE + GSAM transfer experiments       [2 runs]
#  Table 4  (retrieval/RFR)   : ACE FiNER online                      [1 run]
#  Tables 5/6 (latency/cost)  : Timing collected during FiNER online runs
#  App. graph-quality         : Covered by gsam_finer/formula_offline
#  App. transfer-detail       : Covered by transfer experiments
#
#  NOT run (use published ACE paper numbers):
#    ACE main FiNER/Formula offline & online — already in \citet{ace2025}
#    warmup+online — appendix rows dropped from Tab full-results
#
#  Total: 13 experiment runs  (vs 23 in the full script)
# =============================================================================

set -e
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

PROVIDER="modal"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TAX="./eval/finance/data/xbrl_taxonomy.json"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# Helpers
# =============================================================================

# is_complete SAVE_DIR
#   Returns 0 if final_results.json exists and contains a valid accuracy field.
is_complete() {
    local save="$1"
    local f
    f=$(find "$save" -maxdepth 2 -name 'final_results.json' 2>/dev/null | head -1)
    [ -z "$f" ] && return 1
    python3 - "$f" <<'PYEOF' 2>/dev/null
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

# find_partial_run SAVE_DIR
#   Prints the path of the most recently modified subdirectory that has
#   progress.json but NOT final_results.json (i.e. an interrupted run).
find_partial_run() {
    local save="$1"
    local best="" best_time=0
    while IFS= read -r pf; do
        local dir; dir=$(dirname "$pf")
        if [ ! -f "$dir/final_results.json" ]; then
            local t
            t=$(stat -c '%Y' "$pf" 2>/dev/null || stat -f '%m' "$pf" 2>/dev/null || echo 0)
            if [ "$t" -gt "$best_time" ]; then
                best_time="$t"; best="$dir"
            fi
        fi
    done < <(find "$save" -maxdepth 2 -name 'progress.json' 2>/dev/null)
    echo "$best"
}

run_ace() {
    local name="$1"; shift
    local save="results/$name"
    mkdir -p "$save"
    if is_complete "$save"; then
        log "SKIP $name — already complete"; return 0
    fi
    local resume_args=""
    local partial; partial=$(find_partial_run "$save")
    if [ -n "$partial" ]; then
        log "RESUME $name from: $partial"
        resume_args="--resume_path $partial"
    else
        log "START $name (fresh)"
    fi
    python -m eval.finance.run \
        --api_provider $PROVIDER \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model "$MODEL" \
        --save_path "$save" \
        $resume_args \
        "$@" 2>&1 | tee "results/${name}.log"
    log "DONE $name"
}

run_gsam() {
    local name="$1"; shift
    local save="results/$name"
    mkdir -p "$save"
    if is_complete "$save"; then
        log "SKIP $name — already complete"; return 0
    fi
    local resume_args=""
    local partial; partial=$(find_partial_run "$save")
    if [ -n "$partial" ]; then
        log "RESUME $name from: $partial"
        resume_args="--resume_path $partial"
    else
        log "START $name (fresh)"
    fi
    python -m eval.finance.run_gsam \
        --api_provider $PROVIDER \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model "$MODEL" \
        --taxonomy_path "$TAX" \
        --save_path "$save" \
        $resume_args \
        "$@" 2>&1 | tee "results/${name}.log"
    log "DONE $name"
}

cd /c/Users/Window/Desktop/gsam-rsh
mkdir -p results

# =============================================================================
# STEP 1 — GSAM main experiments  (Table 1: offline GT + online GT rows)
# Run GSAM only; ACE numbers come from the published ace2025 paper.
# =============================================================================
log "=== STEP 1: GSAM MAIN EXPERIMENTS ==="

# --- FiNER ---
run_gsam gsam_finer_online \
    --task_name finer \
    --mode online

run_gsam gsam_finer_offline \
    --task_name finer \
    --mode offline \
    --num_epochs 5

# --- Formula ---
run_gsam gsam_formula_online \
    --task_name formula \
    --mode online

run_gsam gsam_formula_offline \
    --task_name formula \
    --mode offline \
    --num_epochs 5

# =============================================================================
# STEP 2 — ACE FiNER online  (Tables 4, 5, 6: ACE comparison metrics)
# Retrieval precision, repeated failure rate, latency, token cost all need an
# ACE FiNER online run to provide the ACE column values.
# =============================================================================
log "=== STEP 2: ACE FINER ONLINE (comparison baseline) ==="

run_ace ace_finer_online \
    --task_name finer \
    --mode online

# =============================================================================
# STEP 3 — Ablation study  (Table 2: FiNER benchmark, online mode)
# All 5 structural ablations run online on FiNER.
# The multi-epoch ablation is offline/1-epoch (graph refinement only applies
# across epochs, so online mode is not meaningful for that variant).
# ACE flat-baseline row in the table uses the ace_finer_online result above.
# =============================================================================
log "=== STEP 3: ABLATIONS (FiNER online) ==="

# Remove ontology layer
run_gsam ablation_no_ontology \
    --task_name finer \
    --mode online \
    --no_ontology

# Remove failure cascade modeling
run_gsam ablation_no_cascades \
    --task_name finer \
    --mode online \
    --no_failure_cascades

# Embedding-only retrieval (no graph BFS — tests graph structure value)
run_gsam ablation_embedding_only \
    --task_name finer \
    --mode online \
    --embedding_only_retrieval

# Untyped edges (all edges become generic 'related_to')
run_gsam ablation_untyped_edges \
    --task_name finer \
    --mode online \
    --untyped_edges

# No multi-epoch graph refinement — offline 1-epoch vs 5-epoch full system
run_gsam ablation_no_multiepoch \
    --task_name finer \
    --mode offline \
    --num_epochs 1 \
    --no_multi_epoch_refinement

# =============================================================================
# STEP 4 — FiNER-Transfer benchmark  (Table 3)
# Build the transfer dataset (idempotent), then evaluate both ACE and GSAM.
# =============================================================================
log "=== STEP 4: FINER-TRANSFER BENCHMARK ==="

TRANSFER_DIR="./eval/finance/data/finer_transfer"

if [ ! -f "$TRANSFER_DIR/concept_pairs.json" ]; then
    log "Building FiNER-Transfer dataset..."
    python -m eval.finance.finer_transfer \
        --taxonomy_path "$TAX" \
        --finer_data_path ./eval/finance/data/finer_train_batched_1000_samples.jsonl \
        --output_dir "$TRANSFER_DIR" 2>&1 | tee results/finer_transfer_build.log
    log "DONE building FiNER-Transfer"
else
    log "SKIP FiNER-Transfer build — concept_pairs.json already exists"
fi

# --- GSAM transfer ---
if [ ! -f "results/transfer_gsam/aggregate_metrics.json" ]; then
    log "START transfer experiments (GSAM)"
    python - <<'PYEOF' 2>&1 | tee results/transfer_gsam.log
import os, json, sys
sys.path.insert(0, '.')
from eval.finance.finer_transfer import (
    load_taxonomy, build_transfer_splits,
    evaluate_transfer, compute_aggregate_transfer_metrics,
)
from eval.finance.data_processor import DataProcessor
from gsam import GSAM

pairs = json.load(open("eval/finance/data/finer_transfer/concept_pairs.json"))
processor = DataProcessor(task_name="finer")
finer_data = []
with open("./eval/finance/data/finer_train_batched_1000_samples.jsonl") as f:
    for line in f:
        if line.strip():
            finer_data.append(json.loads(line))
finer_data = processor.process_task_data(finer_data)
experiments = build_transfer_splits(finer_data, pairs)

gsam = GSAM(
    api_provider="modal",
    generator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    reflector_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    curator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    taxonomy_path="./eval/finance/data/xbrl_taxonomy.json",
)
config = {'max_num_rounds': 3, 'curator_frequency': 1, 'playbook_token_budget': 80000}
os.makedirs("results/transfer_gsam", exist_ok=True)

results = []
for experiment in experiments:
    result = evaluate_transfer(
        method_name="gsam", experiment=experiment, system=gsam,
        data_processor=processor, config=config, save_path="results/transfer_gsam",
    )
    results.append(result)

agg = compute_aggregate_transfer_metrics(results)
print(f"Near-transfer rate:     {agg['near_transfer_rate']:.2%}")
print(f"Far-transfer rate:      {agg['far_transfer_rate']:.2%}")
print(f"Negative transfer rate: {agg['negative_transfer_rate']:.2%}")
json.dump(agg, open("results/transfer_gsam/aggregate_metrics.json", "w"), indent=2)
PYEOF
    log "DONE transfer experiments (GSAM)"
else
    log "SKIP transfer GSAM — aggregate_metrics.json already exists"
fi

# --- ACE transfer (needed for Table 3 ACE row) ---
if [ ! -f "results/transfer_ace/aggregate_metrics.json" ]; then
    log "START transfer experiments (ACE)"
    python - <<'PYEOF' 2>&1 | tee results/transfer_ace.log
import os, json, sys
sys.path.insert(0, '.')
from eval.finance.finer_transfer import (
    build_transfer_splits, evaluate_transfer, compute_aggregate_transfer_metrics,
)
from eval.finance.data_processor import DataProcessor
from ace import ACE
from utils import initialize_clients

pairs = json.load(open("eval/finance/data/finer_transfer/concept_pairs.json"))
processor = DataProcessor(task_name="finer")
finer_data = []
with open("./eval/finance/data/finer_train_batched_1000_samples.jsonl") as f:
    for line in f:
        if line.strip():
            finer_data.append(json.loads(line))
finer_data = processor.process_task_data(finer_data)
experiments = build_transfer_splits(finer_data, pairs)

ace = ACE(
    api_provider="modal",
    generator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    reflector_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    curator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)
config = {'max_num_rounds': 3, 'curator_frequency': 1, 'playbook_token_budget': 80000}
os.makedirs("results/transfer_ace", exist_ok=True)

results = []
for experiment in experiments:
    result = evaluate_transfer(
        method_name="ace", experiment=experiment, system=ace,
        data_processor=processor, config=config, save_path="results/transfer_ace",
    )
    results.append(result)

agg = compute_aggregate_transfer_metrics(results)
print(f"Near-transfer rate:     {agg['near_transfer_rate']:.2%}")
print(f"Far-transfer rate:      {agg['far_transfer_rate']:.2%}")
print(f"Negative transfer rate: {agg['negative_transfer_rate']:.2%}")
json.dump(agg, open("results/transfer_ace/aggregate_metrics.json", "w"), indent=2)
PYEOF
    log "DONE transfer experiments (ACE)"
else
    log "SKIP transfer ACE — aggregate_metrics.json already exists"
fi

# =============================================================================
# STEP 5 — Collect results and print all paper tables
# =============================================================================
log "=== STEP 5: COLLECTING ALL RESULTS ==="
python analyze_results.py

log "ALL EXPERIMENTS COMPLETE"
log "Results in: results/"
log "Re-run 'python analyze_results.py' at any time to reprint paper tables."