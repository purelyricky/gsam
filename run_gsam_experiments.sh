#!/bin/bash
# =============================================================================
# GSAM experiment runner — reproduces clean_results/ directory structure
#
# Directory layout produced:
#   clean_results/ace/          ACE baselines  (finer+formula × online+offline)
#   clean_results/gsam/         Full GSAM      (finer+formula × online+offline)
#   clean_results/ablations/    5 GSAM ablations × finer+formula (all online)
#   clean_results/finer_transfer/  Transfer benchmark
#
# Mapping to paper tables
# ─────────────────────────────────────────────────────────────────────────────
#  Table 1  (main results)   : ACE + GSAM FiNER/Formula online + offline
#  Table 2  (ablation)       : 5 GSAM ablations × FiNER + Formula online
#  Table 3  (FiNER-Transfer) : ACE + GSAM transfer experiments
#  Table 4  (retrieval/RFR)  : ACE FiNER online baseline
#  Tables 5/6 (latency/cost) : Timing collected from FiNER online runs
#  App. graph-quality        : Covered by gsam_finer/formula_offline
# =============================================================================

set -e
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

PROVIDER="modal"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
TAX="./eval/finance/data/xbrl_taxonomy.json"
SAVE_ROOT="clean_results"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# Helpers
# =============================================================================

# is_complete SAVE_DIR
#   Returns 0 if a final_results.json with a valid accuracy field exists.
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
#   Prints the most recently modified subdir that has progress.json but NOT
#   final_results.json (i.e. an interrupted run).
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
    local save="${SAVE_ROOT}/ace/$name"
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
        "$@" 2>&1 | tee "${SAVE_ROOT}/ace/${name}.log"
    log "DONE $name"
}

run_gsam() {
    local name="$1"; shift
    local save="${SAVE_ROOT}/gsam/$name"
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
        "$@" 2>&1 | tee "${SAVE_ROOT}/gsam/${name}.log"
    log "DONE $name"
}

# run_ablation ABLATION_NAME TASK MODE [extra flags...]
#   Saves inside clean_results/ablations/ABLATION_NAME/.
#   Completion is checked per task/mode by matching the timestamped subdir name
#   (gsam_run_*_{task}_{mode}/final_results.json), so both finer and formula
#   runs can coexist in the same parent directory without false-skipping.
run_ablation() {
    local ablation_name="$1" task="$2" mode="$3"
    shift 3
    local save="${SAVE_ROOT}/ablations/${ablation_name}"
    mkdir -p "$save"

    # Check if a completed run for this task/mode already exists
    if find "$save" -maxdepth 2 -name 'final_results.json' 2>/dev/null \
           | grep -q "${task}_${mode}"; then
        log "SKIP ablation/${ablation_name} (${task}/${mode}) — already complete"
        return 0
    fi

    # Resume any interrupted run for this task/mode
    local resume_args=""
    local partial
    partial=$(find "$save" -maxdepth 2 -name 'progress.json' 2>/dev/null \
        | while IFS= read -r pf; do
              dir=$(dirname "$pf")
              echo "$dir" | grep -q "${task}_${mode}" \
                  && [ ! -f "$dir/final_results.json" ] \
                  && echo "$dir"
          done | head -1)
    if [ -n "$partial" ]; then
        log "RESUME ablation/${ablation_name} (${task}/${mode}) from: $partial"
        resume_args="--resume_path $partial"
    else
        log "START ablation/${ablation_name} (${task}/${mode}) (fresh)"
    fi

    python -m eval.finance.run_gsam \
        --api_provider $PROVIDER \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model "$MODEL" \
        --taxonomy_path "$TAX" \
        --save_path "$save" \
        --task_name "$task" \
        --mode "$mode" \
        $resume_args \
        "$@" 2>&1 | tee "${SAVE_ROOT}/ablations/${ablation_name}_${task}_${mode}.log"
    log "DONE ablation/${ablation_name} (${task}/${mode})"
}

cd /c/Users/Window/Desktop/gsam-rsh
mkdir -p "${SAVE_ROOT}/ace" "${SAVE_ROOT}/gsam" "${SAVE_ROOT}/ablations" "${SAVE_ROOT}/finer_transfer"

# =============================================================================
# STEP 1 — ACE baselines  (all 4 conditions for Table 1)
# =============================================================================
log "=== STEP 1: ACE BASELINES ==="

run_ace ace_finer_online \
    --task_name finer \
    --mode online \
    --max_samples 300

run_ace ace_formula_online \
    --task_name formula \
    --mode online \
    --max_samples 300

run_ace ace_finer_offline \
    --task_name finer \
    --mode offline \
    --num_epochs 5

run_ace ace_formula_offline \
    --task_name formula \
    --mode offline \
    --num_epochs 5

# =============================================================================
# STEP 2 — GSAM main experiments  (all 4 conditions for Table 1)
# =============================================================================
log "=== STEP 2: GSAM MAIN EXPERIMENTS ==="

run_gsam gsam_finer_online \
    --task_name finer \
    --mode online \
    --max_samples 300

run_gsam gsam_finer_offline \
    --task_name finer \
    --mode offline \
    --num_epochs 5

run_gsam gsam_formula_online \
    --task_name formula \
    --mode online \
    --max_samples 300

run_gsam gsam_formula_offline \
    --task_name formula \
    --mode offline \
    --num_epochs 5

# =============================================================================
# STEP 3 — Ablation study  (Table 2: all 5 ablations × FiNER + Formula, online)
# Each ablation saves two timestamped run dirs inside its own parent directory.
# =============================================================================
log "=== STEP 3: ABLATIONS (FiNER + Formula, online) ==="

# --- no_ontology ---
run_ablation gsam_no_ontology finer online \
    --max_samples 300 \
    --no_ontology

run_ablation gsam_no_ontology formula online \
    --max_samples 300 \
    --no_ontology

# --- no_cascades ---
run_ablation gsam_no_cascades finer online \
    --max_samples 300 \
    --no_failure_cascades

run_ablation gsam_no_cascades formula online \
    --max_samples 300 \
    --no_failure_cascades

# --- embedding_only ---
run_ablation gsam_embedding_only finer online \
    --max_samples 300 \
    --embedding_only_retrieval

run_ablation gsam_embedding_only formula online \
    --max_samples 300 \
    --embedding_only_retrieval

# --- untyped_edges ---
run_ablation gsam_untyped_edges finer online \
    --max_samples 300 \
    --untyped_edges

run_ablation gsam_untyped_edges formula online \
    --max_samples 300 \
    --untyped_edges

# --- no_multi_epoch  (online; flag documents the ablation for comparison) ---
run_ablation gsam_no_multi_epoch finer online \
    --max_samples 300 \
    --no_multi_epoch_refinement

run_ablation gsam_no_multi_epoch formula online \
    --max_samples 300 \
    --no_multi_epoch_refinement

# =============================================================================
# STEP 4 — FiNER-Transfer benchmark  (Table 3)
# =============================================================================
log "=== STEP 4: FINER-TRANSFER BENCHMARK ==="

TRANSFER_DIR="./eval/finance/data/finer_transfer"

if [ ! -f "$TRANSFER_DIR/concept_pairs.json" ]; then
    log "Building FiNER-Transfer dataset..."
    python -m eval.finance.finer_transfer \
        --taxonomy_path "$TAX" \
        --finer_data_path ./eval/finance/data/finer_train_batched_1000_samples.jsonl \
        --output_dir "$TRANSFER_DIR" 2>&1 | tee "${SAVE_ROOT}/finer_transfer/finer_transfer_build.log"
    log "DONE building FiNER-Transfer"
else
    log "SKIP FiNER-Transfer build — concept_pairs.json already exists"
fi

# --- GSAM transfer ---
if [ ! -f "${SAVE_ROOT}/finer_transfer/gsam_transfer_results.json" ]; then
    log "START transfer experiments (GSAM)"
    python - <<PYEOF 2>&1 | tee "${SAVE_ROOT}/finer_transfer/finer_transfer.log"
import os, json, sys
sys.path.insert(0, '.')
from eval.finance.finer_transfer import (
    build_transfer_splits, evaluate_transfer, compute_aggregate_transfer_metrics,
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
    generator_model="${MODEL}",
    reflector_model="${MODEL}",
    curator_model="${MODEL}",
    taxonomy_path="./eval/finance/data/xbrl_taxonomy.json",
)
config = {'max_num_rounds': 3, 'curator_frequency': 1, 'playbook_token_budget': 80000}
os.makedirs("${SAVE_ROOT}/finer_transfer", exist_ok=True)

results = []
for experiment in experiments:
    result = evaluate_transfer(
        method_name="gsam", experiment=experiment, system=gsam,
        data_processor=processor, config=config,
        save_path="${SAVE_ROOT}/finer_transfer",
    )
    results.append(result)

agg = compute_aggregate_transfer_metrics(results)
print(f"Near-transfer rate:     {agg['near_transfer_rate']:.2%}")
print(f"Far-transfer rate:      {agg['far_transfer_rate']:.2%}")
print(f"Negative transfer rate: {agg['negative_transfer_rate']:.2%}")

sibling_results = [r for r in results if r.get('pair_type') == 'sibling']
distant_results = [r for r in results if r.get('pair_type') == 'distant']
output = {
    "sibling_pair_results": sibling_results,
    "distant_pair_results": distant_results,
    "aggregate": agg,
}
json.dump(output, open("${SAVE_ROOT}/finer_transfer/gsam_transfer_results.json", "w"), indent=2)
PYEOF
    log "DONE transfer experiments (GSAM)"
else
    log "SKIP transfer GSAM — gsam_transfer_results.json already exists"
fi

# --- ACE transfer ---
if [ ! -f "${SAVE_ROOT}/finer_transfer/ace_transfer_results.json" ]; then
    log "START transfer experiments (ACE)"
    python - <<PYEOF 2>&1 | tee -a "${SAVE_ROOT}/finer_transfer/finer_transfer.log"
import os, json, sys
sys.path.insert(0, '.')
from eval.finance.finer_transfer import (
    build_transfer_splits, evaluate_transfer, compute_aggregate_transfer_metrics,
)
from eval.finance.data_processor import DataProcessor
from ace import ACE

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
    generator_model="${MODEL}",
    reflector_model="${MODEL}",
    curator_model="${MODEL}",
)
config = {'max_num_rounds': 3, 'curator_frequency': 1, 'playbook_token_budget': 80000}

results = []
for experiment in experiments:
    result = evaluate_transfer(
        method_name="ace", experiment=experiment, system=ace,
        data_processor=processor, config=config,
        save_path="${SAVE_ROOT}/finer_transfer",
    )
    results.append(result)

agg = compute_aggregate_transfer_metrics(results)
print(f"Near-transfer rate:     {agg['near_transfer_rate']:.2%}")
print(f"Far-transfer rate:      {agg['far_transfer_rate']:.2%}")
print(f"Negative transfer rate: {agg['negative_transfer_rate']:.2%}")

sibling_results = [r for r in results if r.get('pair_type') == 'sibling']
distant_results = [r for r in results if r.get('pair_type') == 'distant']
output = {
    "sibling_pair_results": sibling_results,
    "distant_pair_results": distant_results,
    "aggregate": agg,
}
json.dump(output, open("${SAVE_ROOT}/finer_transfer/ace_transfer_results.json", "w"), indent=2)
PYEOF
    log "DONE transfer experiments (ACE)"
else
    log "SKIP transfer ACE — ace_transfer_results.json already exists"
fi

# --- Generate transfer_summary.json once both results are available ---
if [ -f "${SAVE_ROOT}/finer_transfer/gsam_transfer_results.json" ] && \
   [ -f "${SAVE_ROOT}/finer_transfer/ace_transfer_results.json" ] && \
   [ ! -f "${SAVE_ROOT}/finer_transfer/transfer_summary.json" ]; then
    log "Generating transfer_summary.json..."
    python - <<PYEOF
import json

gsam = json.load(open("${SAVE_ROOT}/finer_transfer/gsam_transfer_results.json"))['aggregate']
ace  = json.load(open("${SAVE_ROOT}/finer_transfer/ace_transfer_results.json"))['aggregate']

def safe_ratio(a, b):
    return round(a / b, 3) if b and b != 0 else None

summary = {
    "comparison": {
        "near_transfer_rate": {
            "ACE":  round(ace.get('near_transfer_rate', 0), 3),
            "GSAM": round(gsam.get('near_transfer_rate', 0), 3),
            "improvement": safe_ratio(
                gsam.get('near_transfer_rate', 0),
                ace.get('near_transfer_rate', 1),
            ),
        },
        "far_transfer_rate": {
            "ACE":  round(ace.get('far_transfer_rate', 0), 3),
            "GSAM": round(gsam.get('far_transfer_rate', 0), 3),
        },
        "transfer_precision": {
            "ACE":  round(ace.get('transfer_precision', 0), 3),
            "GSAM": round(gsam.get('transfer_precision', 0), 3),
        },
        "negative_transfer_rate": {
            "ACE":  round(ace.get('negative_transfer_rate', 0), 3),
            "GSAM": round(gsam.get('negative_transfer_rate', 0), 3),
            "reduction": safe_ratio(
                ace.get('negative_transfer_rate', 0) - gsam.get('negative_transfer_rate', 0),
                ace.get('negative_transfer_rate', 1),
            ),
        },
    }
}
json.dump(summary, open("${SAVE_ROOT}/finer_transfer/transfer_summary.json", "w"), indent=2)
print("transfer_summary.json written.")
PYEOF
fi

# =============================================================================
# STEP 5 — Collect results and print all paper tables
# =============================================================================
log "=== STEP 5: COLLECTING ALL RESULTS ==="
python analyze_results.py --results_dir "${SAVE_ROOT}"

log "ALL EXPERIMENTS COMPLETE"
log "Results in: ${SAVE_ROOT}/"
log "Re-run 'python analyze_results.py --results_dir ${SAVE_ROOT}' at any time to reprint paper tables."
