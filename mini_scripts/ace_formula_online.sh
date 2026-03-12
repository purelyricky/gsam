#!/bin/bash
source "$(dirname "$0")/_common.sh"

NAME="ace_formula_online"
SAVE="results/mini/$NAME"
mkdir -p "$SAVE"

if is_complete "$SAVE"; then
    log "SKIP $NAME — already complete"
    exit 0
fi

resume_args=""
partial=$(find_partial_run "$SAVE")
if [ -n "$partial" ]; then
    log "RESUME $NAME from checkpoint: $(basename "$partial")"
    resume_args="--resume_path $partial"
else
    log "START $NAME"
fi

python -m eval.finance.run \
    --api_provider "$PROVIDER" \
    --generator_model "$MODEL" \
    --reflector_model "$MODEL" \
    --curator_model "$MODEL" \
    --max_num_rounds $MAX_ROUNDS \
    --save_path "$SAVE" \
    --task_name formula \
    --mode online \
    --max_samples $MAX_SAMPLES_ONLINE \
    $resume_args 2>&1 | tee "results/mini/${NAME}.log"

rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
    log "ERROR in $NAME (exit $rc)"
    exit 1
fi
log "DONE $NAME"
