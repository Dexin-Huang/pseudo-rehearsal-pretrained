#!/usr/bin/env bash
# Alpha sweep on KD methods. Tests whether higher alpha rescues pseudo-rehearsal.
# Assumes joint head checkpoints already exist in results/.
set -euo pipefail
cd "$(dirname "$0")"

METHODS=(pseudo pseudo_oracle lwf)
ALPHAS=(0.5 1.0 5.0 10.0 50.0)
SEEDS=(0 1 2)

for m in "${METHODS[@]}"; do
  for a in "${ALPHAS[@]}"; do
    for s in "${SEEDS[@]}"; do
      tag="alpha${a}"
      echo "=== $m alpha=$a seed=$s ==="
      python train.py --config "configs/${m}.yaml" --seed "$s" --alpha "$a" --tag "$tag"
    done
  done
done

echo
echo "=== alpha sweep done ==="
