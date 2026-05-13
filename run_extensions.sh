#!/usr/bin/env bash
# Run the second-wave extension experiments. Assumes joint head checkpoints
# already exist in results/ (from the first sweep) and that cache_features.py
# has been run with --with-stl10.
set -euo pipefail
cd "$(dirname "$0")"

METHODS=(pseudo_oracle pseudo_natural pseudo_oracle_natural)
SEEDS=(0 1 2)

for m in "${METHODS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo "=== $m seed=$s ==="
    python train.py --config "configs/${m}.yaml" --seed "$s"
  done
done

echo
echo "=== aggregate (all methods) ==="
python evaluate.py --results results/

echo
echo "=== mechanism check ==="
python mechanism_check.py

echo
echo "=== plots ==="
python plots.py
