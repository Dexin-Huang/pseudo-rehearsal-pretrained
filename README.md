# Revisiting Pseudo-Rehearsal with Pretrained Backbones

Companion code for the COMS4995W Applied Deep Learning paper.

See [SPEC.md](SPEC.md) for the full design.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
# 1. Cache ViT-S features for CIFAR-100 and a random-pixel pool (one-time)
python cache_features.py

# 2. Run all methods × seeds (writes results/*.json)
bash run_all.sh

# 3. Aggregate and print final table
python evaluate.py --results results/
```

Caches land in `cache/`, results in `results/`. Both gitignored.

## Layout

See SPEC.md §4.4.
