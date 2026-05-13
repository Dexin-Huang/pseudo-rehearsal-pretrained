"""Mechanism check: predict the dominant pseudo-class from a geometric argument.

Hypothesis: random pixels through a frozen ViT collapse to a tight region of
feature space. The mean random-pool feature mu_random, multiplied by the
joint-trained head's weights, gives a prediction for which class the teacher
will assign most random pixels.

  predicted_class = argmax_c (W_c @ mu_random + b_c)

This script verifies against the empirically-dominant class from diagnostics.json.
Also computes the histogram of teacher predictions on the random pool directly
(not on the mean), and the entropy of that histogram.

Run on the pod after cache_features.py has produced random_pixel_features_*.npy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data import load_random_pool

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def main() -> None:
    state = torch.load(RESULTS / "joint_seed0_head.pt", map_location="cpu")
    w = state["w"].numpy().astype(np.float32)  # (100, 384)
    b = state["b"].numpy().astype(np.float32)  # (100,)

    pool = load_random_pool().astype(np.float32)  # (N, 384)
    mu = pool.mean(axis=0)                        # (384,)
    sigma = pool.std(axis=0)

    # Prediction from the mean random feature
    mean_logits = w @ mu + b
    predicted_from_mean = int(mean_logits.argmax())
    top3_from_mean = np.argsort(mean_logits)[-3:][::-1].tolist()

    # Empirical histogram across the full random pool
    all_logits = pool @ w.T + b                   # (N, 100)
    all_preds = all_logits.argmax(axis=1)
    hist = np.bincount(all_preds, minlength=100)
    top_class_emp = int(hist.argmax())
    top_frac = float(hist.max() / hist.sum())
    n_classes_predicted = int((hist > 0).sum())

    # Empirical match
    with open(RESULTS / "diagnostics.json") as f:
        diag = json.load(f)
    diag_top = int(diag["top_class"])

    report = {
        "predicted_from_mean_feature": predicted_from_mean,
        "predicted_top3_from_mean": top3_from_mean,
        "empirical_top_class": top_class_emp,
        "empirical_top_class_fraction": top_frac,
        "n_classes_predicted_on_random": n_classes_predicted,
        "diagnostics_top_class": diag_top,
        "geometric_prediction_matches_empirical": (predicted_from_mean == top_class_emp),
        "geometric_prediction_matches_diagnostics": (predicted_from_mean == diag_top),
        "random_pool_mean_feature_norm": float(np.linalg.norm(mu)),
        "random_pool_mean_per_dim_std": float(sigma.mean()),
    }

    out = RESULTS / "mechanism_check.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
