#!/bin/bash
source "$(dirname "$0")/_common.sh"

TRANSFER_DIR="./eval/finance/data/finer_transfer"

# Build the FiNER-Transfer dataset if not already done
if [ ! -f "$TRANSFER_DIR/concept_pairs.json" ]; then
    log "Building FiNER-Transfer dataset..."
    python -m eval.finance.finer_transfer \
        --taxonomy_path "$TAX" \
        --finer_data_path ./eval/finance/data/finer_train_batched_1000_samples.jsonl \
        --output_dir "$TRANSFER_DIR" 2>&1 | tee results/mini/finer_transfer_build.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR building FiNER-Transfer dataset — aborting"
        exit 1
    fi
    log "DONE building FiNER-Transfer dataset"
else
    log "SKIP FiNER-Transfer build — concept_pairs.json already exists"
fi

if [ -f "results/mini/transfer_gsam/aggregate_metrics.json" ]; then
    log "SKIP transfer_gsam — aggregate_metrics.json already exists"
    exit 0
fi

log "START transfer_gsam"
mkdir -p results/mini/transfer_gsam

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
    log "ERROR in transfer_gsam (exit $rc)"
    exit 1
fi
log "DONE transfer_gsam"
