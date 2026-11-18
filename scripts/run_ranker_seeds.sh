#!/usr/bin/env bash
# Train the rankers with seeds 0,1,2 (isolated outputs), then aggregate mean +/- std.
# Run from anywhere:  bash scripts/run_ranker_seeds.sh      (about 1-2 minutes per seed on the sample)
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${RECSYS_MODE:-sample}"
for seed in 0 1 2; do
  echo "=== ranker seed ${seed} (mode ${mode}) ==="
  python -m recsys.ranking.train --mode "${mode}" --seed "${seed}" --tag "s${seed}"
done
python -m scripts.aggregate_ranking --mode "${mode}"
