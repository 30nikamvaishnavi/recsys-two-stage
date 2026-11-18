#!/usr/bin/env bash
set -euo pipefail

MODE="${RECSYS_MODE:-sample}"
PYTHON="${PYTHON:-python}"
mkdir -p logs

run_retrieval() {
    local tag="$1"
    shift
    "$PYTHON" -m recsys.retrieval.train --mode "$MODE" --tag "$tag" "$@" \
        >"logs/retrieval-${MODE}-${tag}.log" 2>&1
}

for seed in 0 1 2; do
    run_retrieval "logq-on-seed-${seed}" \
        --seed "$seed" --set retrieval.logq_correction=true
    run_retrieval "logq-off-seed-${seed}" \
        --seed "$seed" --set retrieval.logq_correction=false
done

run_retrieval "temperature-0.1" --seed 0 --set retrieval.temperature=0.1
run_retrieval "temperature-0.02" --seed 0 --set retrieval.temperature=0.02
run_retrieval "embedding-dim-128" --seed 0 --set retrieval.embedding_dim=128
run_retrieval "learning-rate-0.001" --seed 0 --set retrieval.learning_rate=0.001
run_retrieval "user-id-dropout-0.5" --seed 0 --set retrieval.user_id_dropout=0.5

"$PYTHON" scripts/aggregate_retrieval.py --mode "$MODE" \
    >"logs/retrieval-${MODE}-aggregate.log" 2>&1
