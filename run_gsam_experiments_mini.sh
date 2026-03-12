#!/bin/bash
# =============================================================================
# GSAM complete experiment suite — 7B model, budget-optimised sample counts
#
# Goal: produce ALL numbers needed for every thesis table using the same
#       model and test conditions across every system/ablation.
#
# Model: DeepSeek-R1-Distill-Qwen-7B  (1× A100-40, ~$3/hr on Modal)
# Test samples (online):  75   (was 150 — halved to save cost)
# Train samples (offline): 150  (was 300 — halved to save cost)
# Offline epochs:          2   (was 3  — reduced to save cost)
# Reflection rounds:       1
# Transfer pairs:          8 total (4 sibling + 4 distant, capped to save cost)
#
# Run order — paired ACE vs GSAM for immediate comparison:
#   STEP 1 (online pairs, fastest):
#     ACE Formula online  -> GSAM Formula online   (~12-18 + 15-22 min)
#     ACE FiNER online    -> GSAM FiNER online      (~12-20 + 18-25 min)
#   STEP 2 (online ablations, ~12-20 min each):
#     no_ontology, no_cascades, untyped_edges, embedding_only
#   STEP 3 (offline pairs, medium):
#     ACE Formula offline -> GSAM Formula offline  (~20-30 + 22-35 min)
#     ACE FiNER offline   -> GSAM FiNER offline    (~22-35 + 25-40 min)
#     ablation_no_multiepoch (offline 1 epoch)     (~12-18 min)
#   STEP 4 (transfer, capped at 8 pairs):
#     ACE transfer  -> GSAM transfer               (~15-25 + 18-30 min)
#
# Total: ~4-6 hours, $12-18 on Modal A100-40GB
#
# RESUME: safe to interrupt at any time (Ctrl+C).
#   The system saves a checkpoint after every window (online) or step (offline).
#   Re-running the script auto-detects completed and partial runs:
#     - completed (has final_results.json with accuracy) → skipped
#     - partial   (has progress.json, no final_results)  → resumed from checkpoint
#     - pending   (no directory yet)                     → started fresh
#   A status table is printed at startup so you can see where things stand.
#
# After all runs finish:
#   python analyze_results.py --results_dir results/mini
# =============================================================================

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# =============================================================================
# ARCHIVING STALE RESULTS — never delete, always archive
#
# When a run produced bad results (e.g. due to a bug that has since been
# fixed), NEVER delete the directory — move it to results/mini/archived/
# so it is preserved for debugging and comparison.
#
# Use archive_run() defined below.  It moves the run directory AND its
# .log file into results/mini/archived/<timestamp>/ and prints a one-line
# summary of why it was archived.
#
# To force a specific run to re-execute, call:
#   archive_run <name> "<reason>"
# before the corresponding run_gsam / run_ace call.  The run wrapper will
# then see no final_results.json and start fresh.
#
# NOTE: Runs that were stale at the time this script was updated have
# already been archived manually (or lost before this policy was in place).
# All future stale results must be archived, not deleted.
#
# ALL ACE runs (online + offline, Formula + FiNER) are ALREADY DONE — their
# final_results.json was recovered from the run logs and written manually.
# All run_ace calls are commented out so they are never re-executed.
# =============================================================================

PROVIDER="modal"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TAX="./eval/finance/data/xbrl_taxonomy.json"

MAX_SAMPLES_ONLINE=75
MAX_SAMPLES_OFFLINE=150
NUM_EPOCHS=2
MAX_ROUNDS=1

FAILED_RUNS=()

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# archive_run <name> "<reason>"
#
# Moves results/mini/<name>/ and results/mini/<name>.log into
# results/mini/archived/<timestamp>_<name>/ so the run can be re-executed
# while the old results are preserved for debugging and comparison.
#
# Usage example (before a run whose old results are stale):
#   archive_run gsam_finer_online "Bug10+11 fixed — old result was 4.0%"
# =============================================================================
ARCHIVE_TS=$(date '+%Y%m%d_%H%M%S')

archive_run() {
    local name="$1"
    local reason="${2:-stale}"
    local src_dir="results/mini/${name}"
    local src_log="results/mini/${name}.log"
    local dest="results/mini/archived/${ARCHIVE_TS}_${name}"

    # Nothing to archive if the run never started
    if [ ! -e "$src_dir" ] && [ ! -e "$src_log" ]; then
        log "ARCHIVE $name — nothing to archive (never ran)"
        return 0
    fi

    mkdir -p "results/mini/archived"
    mkdir -p "$dest"

    [ -d "$src_dir" ] && mv "$src_dir" "$dest/run_data"
    [ -f "$src_log" ] && mv "$src_log" "$dest/run.log"

    # Write a one-line reason file so it is obvious later why it was archived
    echo "$reason" > "$dest/ARCHIVED_REASON.txt"
    echo "archived_at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$dest/ARCHIVED_REASON.txt"

    log "ARCHIVED $name -> archived/${ARCHIVE_TS}_${name}/ ($reason)"
}

# =============================================================================
# Resume helpers
# =============================================================================

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
# There is at most one partial run per save_dir at any time.
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

# Print a status table for all 16 runs before starting
print_status() {
    local names=(
        ace_formula_online    gsam_formula_online
        ace_finer_online      gsam_finer_online
        ablation_no_ontology  ablation_no_cascades
        ablation_untyped_edges ablation_embedding_only
        ace_formula_offline   gsam_formula_offline
        ace_finer_offline     gsam_finer_offline
        ablation_no_multiepoch
        transfer_ace          transfer_gsam
    )
    log "--- Run status at startup ---"
    for name in "${names[@]}"; do
        local save="results/mini/$name"
        if is_complete "$save"; then
            log "  [DONE   ] $name"
        elif [ -n "$(find_partial_run "$save")" ]; then
            local partial; partial=$(find_partial_run "$save")
            local prog="$partial/progress.json"
            local cursor=""
            if [ -f "$prog" ]; then
                cursor=$(python - "$prog" <<'PYEOF' 2>/dev/null
import json,sys; p=json.load(open(sys.argv[1]))
mode=p.get('mode','?')
if mode=='online':   print(f"window={p.get('window','?')}")
elif mode=='offline': print(f"epoch={p.get('epoch','?')} step={p.get('step','?')}")
PYEOF
)
            fi
            log "  [PARTIAL] $name  ($cursor)"
        else
            log "  [PENDING] $name"
        fi
    done
    log "----------------------------"
}

# =============================================================================
# Run wrappers
# =============================================================================

run_ace() {
    local name="$1"; shift
    local save="results/mini/$name"
    mkdir -p "$save"

    if is_complete "$save"; then
        log "SKIP $name — already complete"
        return 0
    fi

    local resume_args=""
    local partial; partial=$(find_partial_run "$save")
    if [ -n "$partial" ]; then
        log "RESUME $name from checkpoint: $(basename "$partial")"
        resume_args="--resume_path $partial"
    else
        log "START $name"
    fi

    python -m eval.finance.run \
        --api_provider "$PROVIDER" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model "$MODEL" \
        --max_num_rounds $MAX_ROUNDS \
        --save_path "$save" \
        $resume_args \
        "$@" 2>&1 | tee "results/mini/${name}.log"

    # tee always exits 0 — check the python exit code via PIPESTATUS
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "ERROR in $name (exit $rc) — continuing with next run"
        FAILED_RUNS+=("$name")
        return 1
    fi
    log "DONE $name"
}

run_gsam() {
    local name="$1"; shift
    local save="results/mini/$name"
    mkdir -p "$save"

    if is_complete "$save"; then
        log "SKIP $name — already complete"
        return 0
    fi

    local resume_args=""
    local partial; partial=$(find_partial_run "$save")
    if [ -n "$partial" ]; then
        log "RESUME $name from checkpoint: $(basename "$partial")"
        resume_args="--resume_path $partial"
    else
        log "START $name"
    fi

    python -m eval.finance.run_gsam \
        --api_provider "$PROVIDER" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model "$MODEL" \
        --max_num_rounds $MAX_ROUNDS \
        --taxonomy_path "$TAX" \
        --save_path "$save" \
        $resume_args \
        "$@" 2>&1 | tee "results/mini/${name}.log"

    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "ERROR in $name (exit $rc) — continuing with next run"
        FAILED_RUNS+=("$name")
        return 1
    fi
    log "DONE $name"
}

# =============================================================================
# Setup
# =============================================================================

cd /c/Users/Window/Desktop/gsam-rsh
mkdir -p results/mini
print_status

# =============================================================================
# STEP 1 — Online pairs (fastest, immediate ACE vs GSAM comparison)
# =============================================================================
log "=== STEP 1: ONLINE PAIRS ==="

# --- Formula online pair ---
# SKIP: ACE Formula online already done — final_results.json written from logs
# run_ace ace_formula_online \
#     --task_name formula --mode online --max_samples $MAX_SAMPLES_ONLINE

run_gsam gsam_formula_online \
    --task_name formula --mode online --max_samples $MAX_SAMPLES_ONLINE

# --- FiNER online pair ---
# SKIP: ACE FiNER online already done — final_results.json written from logs
# run_ace ace_finer_online \
#     --task_name finer --mode online --max_samples $MAX_SAMPLES_ONLINE

run_gsam gsam_finer_online \
    --task_name finer --mode online --max_samples $MAX_SAMPLES_ONLINE

# =============================================================================
# STEP 2 — Online ablations (compare directly against gsam_finer_online above)
# =============================================================================
log "=== STEP 2: ONLINE ABLATIONS ==="

# No ontology — tests taxonomic backbone (H1)
run_gsam ablation_no_ontology \
    --task_name finer --mode online --max_samples $MAX_SAMPLES_ONLINE \
    --no_ontology

# No failure cascades — tests anti-pattern propagation (H2)
run_gsam ablation_no_cascades \
    --task_name finer --mode online --max_samples $MAX_SAMPLES_ONLINE \
    --no_failure_cascades

# Untyped edges — tests edge type semantics
run_gsam ablation_untyped_edges \
    --task_name finer --mode online --max_samples $MAX_SAMPLES_ONLINE \
    --untyped_edges

# Embedding-only retrieval — tests graph BFS value (H3)
run_gsam ablation_embedding_only \
    --task_name finer --mode online --max_samples $MAX_SAMPLES_ONLINE \
    --embedding_only_retrieval

# =============================================================================
# STEP 3 — Offline pairs + offline ablation
# =============================================================================
log "=== STEP 3: OFFLINE PAIRS ==="

# --- Formula offline pair ---
# SKIP: ACE Formula offline already done — final_results.json written from logs
# run_ace ace_formula_offline \
#     --task_name formula --mode offline \
#     --num_epochs $NUM_EPOCHS --max_samples $MAX_SAMPLES_OFFLINE

run_gsam gsam_formula_offline \
    --task_name formula --mode offline \
    --num_epochs $NUM_EPOCHS --max_samples $MAX_SAMPLES_OFFLINE

# --- FiNER offline pair ---
# SKIP: ACE FiNER offline already done — final_results.json written from logs
# run_ace ace_finer_offline \
#     --task_name finer --mode offline \
#     --num_epochs $NUM_EPOCHS --max_samples $MAX_SAMPLES_OFFLINE

run_gsam gsam_finer_offline \
    --task_name finer --mode offline \
    --num_epochs $NUM_EPOCHS --max_samples $MAX_SAMPLES_OFFLINE

# No multi-epoch refinement — offline 1 epoch (compare against gsam_finer_offline above)
run_gsam ablation_no_multiepoch \
    --task_name finer --mode offline \
    --num_epochs 1 --max_samples $MAX_SAMPLES_OFFLINE \
    --no_multi_epoch_refinement

# =============================================================================
# STEP 4 — FiNER-Transfer (longest, ~50-90 min each)
# =============================================================================
log "=== STEP 4: FINER-TRANSFER BENCHMARK (all pairs) ==="

TRANSFER_DIR="./eval/finance/data/finer_transfer"

if [ ! -f "$TRANSFER_DIR/concept_pairs.json" ]; then
    log "Building FiNER-Transfer dataset..."
    python -m eval.finance.finer_transfer \
        --taxonomy_path "$TAX" \
        --finer_data_path ./eval/finance/data/finer_train_batched_1000_samples.jsonl \
        --output_dir "$TRANSFER_DIR" 2>&1 | tee results/mini/finer_transfer_build.log
    log "DONE building FiNER-Transfer"
else
    log "SKIP FiNER-Transfer build — concept_pairs.json already exists"
fi

# --- ACE transfer ---
if [ ! -f "results/mini/transfer_ace/aggregate_metrics.json" ]; then
    log "START transfer experiments (ACE)"
    python - <<'PYEOF' 2>&1 | tee results/mini/transfer_ace.log
import os, json, sys
sys.path.insert(0, '.')
from eval.finance.finer_transfer import (
    build_transfer_splits, evaluate_transfer, compute_aggregate_transfer_metrics,
)
from eval.finance.data_processor import DataProcessor
from ace import ACE

pairs_all = json.load(open("eval/finance/data/finer_transfer/concept_pairs.json"))
sibling = [p for p in pairs_all if p["pair_type"] == "sibling"][:4]
distant = [p for p in pairs_all if p["pair_type"] == "distant"][:4]
pairs = sibling + distant
print(f"Running transfer on {len(sibling)} sibling + {len(distant)} distant pairs ({len(pairs)} total, capped for budget)")

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
    generator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    reflector_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    curator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
config = {'max_num_rounds': 1, 'curator_frequency': 1, 'playbook_token_budget': 80000}
os.makedirs("results/mini/transfer_ace", exist_ok=True)

results = []
for i, experiment in enumerate(experiments):
    print(f"  Pair {i+1}/{len(experiments)}: {experiment.get('source_concept')} -> {experiment.get('target_concept')}")
    result = evaluate_transfer(
        method_name="ace", experiment=experiment, system=ace,
        data_processor=processor, config=config, save_path="results/mini/transfer_ace",
    )
    results.append(result)

agg = compute_aggregate_transfer_metrics(results)
print(f"Near-transfer rate:     {agg['near_transfer_rate']:.2%}")
print(f"Far-transfer rate:      {agg['far_transfer_rate']:.2%}")
print(f"Negative transfer rate: {agg['negative_transfer_rate']:.2%}")
json.dump(agg, open("results/mini/transfer_ace/aggregate_metrics.json", "w"), indent=2)
PYEOF
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "ERROR in transfer_ace (exit $rc) — continuing"
        FAILED_RUNS+=("transfer_ace")
    else
        log "DONE transfer experiments (ACE)"
    fi
else
    log "SKIP transfer ACE — aggregate_metrics.json already exists"
fi

# --- GSAM transfer ---
if [ ! -f "results/mini/transfer_gsam/aggregate_metrics.json" ]; then
    log "START transfer experiments (GSAM)"
    python - <<'PYEOF' 2>&1 | tee results/mini/transfer_gsam.log
import os, json, sys
sys.path.insert(0, '.')
from eval.finance.finer_transfer import (
    load_taxonomy, build_transfer_splits,
    evaluate_transfer, compute_aggregate_transfer_metrics,
)
from eval.finance.data_processor import DataProcessor
from gsam import GSAM

pairs_all = json.load(open("eval/finance/data/finer_transfer/concept_pairs.json"))
sibling = [p for p in pairs_all if p["pair_type"] == "sibling"][:4]
distant = [p for p in pairs_all if p["pair_type"] == "distant"][:4]
pairs = sibling + distant
print(f"Running transfer on {len(sibling)} sibling + {len(distant)} distant pairs ({len(pairs)} total, capped for budget)")

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
    generator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    reflector_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    curator_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    taxonomy_path="./eval/finance/data/xbrl_taxonomy.json",
)
config = {'max_num_rounds': 1, 'curator_frequency': 1, 'playbook_token_budget': 80000}
os.makedirs("results/mini/transfer_gsam", exist_ok=True)

results = []
for i, experiment in enumerate(experiments):
    print(f"  Pair {i+1}/{len(experiments)}: {experiment.get('source_concept')} -> {experiment.get('target_concept')}")
    result = evaluate_transfer(
        method_name="gsam", experiment=experiment, system=gsam,
        data_processor=processor, config=config, save_path="results/mini/transfer_gsam",
    )
    results.append(result)

agg = compute_aggregate_transfer_metrics(results)
print(f"Near-transfer rate:     {agg['near_transfer_rate']:.2%}")
print(f"Far-transfer rate:      {agg['far_transfer_rate']:.2%}")
print(f"Negative transfer rate: {agg['negative_transfer_rate']:.2%}")
json.dump(agg, open("results/mini/transfer_gsam/aggregate_metrics.json", "w"), indent=2)
PYEOF
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "ERROR in transfer_gsam (exit $rc) — continuing"
        FAILED_RUNS+=("transfer_gsam")
    else
        log "DONE transfer experiments (GSAM)"
    fi
else
    log "SKIP transfer GSAM — aggregate_metrics.json already exists"
fi

# =============================================================================
# STEP 5 — Collect and print all results
# =============================================================================
log "=== STEP 5: COLLECTING RESULTS ==="
python analyze_results.py --results_dir results/mini

# Final status report
echo ""
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log "FINISHED WITH ERRORS — ${#FAILED_RUNS[@]} run(s) failed:"
    for r in "${FAILED_RUNS[@]}"; do
        log "  FAILED: $r  (log: results/mini/${r}.log)"
    done
    log "Fix the issue and re-run — completed runs will be skipped automatically."
else
    log "ALL EXPERIMENTS COMPLETE — no errors"
fi

log ""
log "Each thesis table is populated by:"
log "  Table 1 (main)     : results/mini/{gsam,ace}_{finer,formula}_{online,offline}/"
log "  Table 2 (ablation) : results/mini/ablation_*/ vs gsam_finer_online + ace_finer_online"
log "  Table 3 (transfer) : results/mini/transfer_{gsam,ace}/aggregate_metrics.json"
log "  Table 4 (ret/RFR)  : results/mini/{gsam,ace}_finer_online/ retrieval_logs.jsonl"
log "  Tables 5/6 (latency): results/mini/{gsam,ace}_finer_online/ final_results.json"
log ""
log "Re-run analysis at any time: python analyze_results.py --results_dir results/mini"
