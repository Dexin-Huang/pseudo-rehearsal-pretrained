"""Diagnostic probes for the random-pixel-feature hypothesis (SPEC §3.7).

Runs after a joint training run (which produces a clean 100-way head).
Computes:
  1. Real vs random feature L2 norm (mean ± std)
  2. NN cosine distance from random-feature samples to real-feature samples
  3. Predictive entropy of the joint head on real vs random features
  4. Top-class histogram on random features (collapse vs uniform vs informative)

Output: CSV row + a stdout summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load_cifar100_cached, load_random_pool

RESULTS_DIR = Path(__file__).parent / "results"


def load_joint_head(joint_ckpt: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a joint head saved as a torch checkpoint dict {'w': (100,384), 'b': (100,)}."""
    state = torch.load(joint_ckpt, map_location="cpu")
    return state["w"], state["b"]


def feature_norms(x: np.ndarray) -> tuple[float, float]:
    n = np.linalg.norm(x.astype(np.float32), axis=1)
    return float(n.mean()), float(n.std())


def nn_cosine(query: np.ndarray, gallery: np.ndarray, sample: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(query), size=min(sample, len(query)), replace=False)
    g_idx = rng.choice(len(gallery), size=min(sample, len(gallery)), replace=False)
    q = query[q_idx].astype(np.float32)
    g = gallery[g_idx].astype(np.float32)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    sims = q @ g.T  # cosine similarity
    nn_sim = sims.max(axis=1)
    return {
        "mean_nn_cosine": float(nn_sim.mean()),
        "p25_nn_cosine": float(np.quantile(nn_sim, 0.25)),
        "p75_nn_cosine": float(np.quantile(nn_sim, 0.75)),
    }


def entropy(features: np.ndarray, w: torch.Tensor, b: torch.Tensor, batch: int = 2048) -> float:
    feats = torch.from_numpy(features.astype(np.float32))
    entropies = []
    for i in range(0, len(feats), batch):
        chunk = feats[i : i + batch]
        logits = F.linear(chunk, w, b)
        p = F.softmax(logits, dim=-1)
        ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)
        entropies.append(ent.numpy())
    return float(np.concatenate(entropies).mean())


def top_class_distribution(
    features: np.ndarray, w: torch.Tensor, b: torch.Tensor, n_classes: int = 100, batch: int = 2048
) -> dict:
    feats = torch.from_numpy(features.astype(np.float32))
    preds = []
    for i in range(0, len(feats), batch):
        chunk = feats[i : i + batch]
        logits = F.linear(chunk, w, b)
        preds.append(logits.argmax(dim=-1).numpy())
    preds = np.concatenate(preds)
    counts = np.bincount(preds, minlength=n_classes)
    return {
        "top_class": int(counts.argmax()),
        "top_class_frac": float(counts.max() / counts.sum()),
        "n_classes_predicted": int((counts > 0).sum()),
        "entropy_over_class_distribution": float(
            -(
                (counts / counts.sum())
                * np.log((counts / counts.sum()).clip(min=1e-12))
            ).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-ckpt",
        type=Path,
        required=True,
        help="Path to a joint head checkpoint (state dict with 'w', 'b').",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "diagnostics.json")
    parser.add_argument("--pool-size", type=int, default=100_000)
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--nn-sample", type=int, default=5000)
    args = parser.parse_args()

    train_x, _train_y, _, _ = load_cifar100_cached()
    random_x = load_random_pool(args.pool_size, args.pool_seed)
    w, b = load_joint_head(args.joint_ckpt)

    print(f"real features: {train_x.shape}, random features: {random_x.shape}")

    real_norm = feature_norms(train_x)
    rand_norm = feature_norms(random_x)
    nn = nn_cosine(random_x, train_x, sample=args.nn_sample)
    real_ent = entropy(train_x, w, b)
    rand_ent = entropy(random_x, w, b)
    top_dist = top_class_distribution(random_x, w, b)

    report = {
        "real_feature_norm_mean": real_norm[0],
        "real_feature_norm_std": real_norm[1],
        "random_feature_norm_mean": rand_norm[0],
        "random_feature_norm_std": rand_norm[1],
        **nn,
        "teacher_entropy_real": real_ent,
        "teacher_entropy_random": rand_ent,
        "uniform_max_entropy": float(np.log(100)),
        **top_dist,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
