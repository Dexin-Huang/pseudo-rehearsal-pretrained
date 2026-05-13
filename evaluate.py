"""Compute ACC / FGT / BWT from an accuracy matrix, and aggregate runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def acc_fgt_bwt(A: np.ndarray) -> tuple[float, float, float]:
    """A: (T, T) matrix where A[k, t] = accuracy on task t after training task k.
    Entries with t > k are NaN (not yet trained).
    """
    T = A.shape[0]
    final = A[T - 1]  # (T,)
    acc = float(np.mean(final))

    fgt_terms, bwt_terms = [], []
    for t in range(T - 1):
        # FGT_t = max_{k>=t} A[k,t] - A[T-1,t]. Include final row in the max
        # (so FGT >= 0 by construction).
        col = A[t:T, t]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        fgt_terms.append(np.max(col) - final[t])
        bwt_terms.append(final[t] - A[t, t])

    fgt = float(np.mean(fgt_terms)) if fgt_terms else 0.0
    bwt = float(np.mean(bwt_terms)) if bwt_terms else 0.0
    return acc, fgt, bwt


def aggregate(results_dir: Path) -> None:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        by_method[r["method"]].append(r)

    print(f"{'method':<14}{'ACC':>10}{'FGT':>10}{'BWT':>10}    seeds")
    print("-" * 60)
    for method, runs in by_method.items():
        accs, fgts, bwts = [], [], []
        for r in runs:
            A = np.array(r["accuracy_matrix"])
            a, f, b = acc_fgt_bwt(A)
            accs.append(a)
            fgts.append(f)
            bwts.append(b)
        accs, fgts, bwts = np.array(accs), np.array(fgts), np.array(bwts)
        print(
            f"{method:<14}"
            f"{accs.mean():>6.3f}±{accs.std():.3f}"
            f"{fgts.mean():>6.3f}±{fgts.std():.3f}"
            f"{bwts.mean():>6.3f}±{bwts.std():.3f}"
            f"    n={len(runs)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.results)


if __name__ == "__main__":
    main()
