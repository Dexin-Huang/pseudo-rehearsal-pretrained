#!/usr/bin/env bash
# Run all six methods × three seeds. Assumes cache_features.py has been run.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p results

METHODS=(joint sequential ewc replay lwf pseudo)
SEEDS=(0 1 2)

for m in "${METHODS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo "=== $m seed=$s ==="
    python train.py --config "configs/${m}.yaml" --seed "$s"
  done
done

echo
echo "=== aggregate ==="
python evaluate.py --results results/

# Diagnostics use the joint-seed-0 head as a clean classifier (SPEC §3.7).
if [[ -f results/joint_seed0_head.pt ]]; then
  echo
  echo "=== diagnostics ==="
  python diagnostics.py --joint-ckpt results/joint_seed0_head.pt --out results/diagnostics.json
fi
