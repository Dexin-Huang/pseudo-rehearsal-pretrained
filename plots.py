"""Generate figures for the paper.

Two plots:
1. running_avg_accuracy.png — for each method, mean over seeds of
   (1/(k+1)) sum_{t<=k} A[k,t] as k goes 0..T-1. The standard CL forgetting curve.
2. final_per_task.png — for each method, A[T-1, t] vs t. Shows which task's
   classes survived to the end.

Aggregates the diagnostic numbers and emits a CSV row alongside the figures.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


PLOT_ORDER = [
    "joint",
    "replay",
    "ewc",
    "lwf",
    "pseudo",
    "pseudo_oracle",
    "pseudo_natural",
    "pseudo_oracle_natural",
    "sequential",
]


def load_runs() -> dict[str, list[np.ndarray]]:
    by_method: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in sorted(glob.glob(str(RESULTS / "*.json"))):
        name = Path(path).stem
        if "diagnostics" in name or "mechanism" in name:
            continue
        with open(path) as f:
            r = json.load(f)
        by_method[r["method"]].append(np.array(r["accuracy_matrix"]))
    return by_method


def running_avg(A: np.ndarray) -> np.ndarray:
    T = A.shape[0]
    out = np.zeros(T)
    for k in range(T):
        row = A[k, : k + 1]
        out[k] = np.nanmean(row) if np.isfinite(row).any() else np.nan
    return out


def plot_running_avg(by_method: dict[str, list[np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in PLOT_ORDER:
        if m not in by_method:
            continue
        runs = by_method[m]
        if m == "joint":
            # Joint has only a single populated row; report its final accuracy as a horizontal reference.
            val = np.nanmean([r[-1, :] for r in runs])
            ax.axhline(val, color="black", linestyle="--", label=f"joint = {val:.2f}", linewidth=1)
            continue
        curves = np.array([running_avg(r) for r in runs])
        m_mean = curves.mean(axis=0)
        m_std = curves.std(axis=0)
        x = np.arange(len(m_mean))
        ax.plot(x, m_mean, marker="o", label=m)
        ax.fill_between(x, m_mean - m_std, m_mean + m_std, alpha=0.15)
    ax.set_xlabel("task index (after training task k)")
    ax.set_ylabel("average accuracy on tasks 0..k")
    ax.set_title("Running average accuracy across continual-learning tasks")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "running_avg_accuracy.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def plot_final_per_task(by_method: dict[str, list[np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    T = next(iter(by_method.values()))[0].shape[0]
    for m in PLOT_ORDER:
        if m not in by_method or m == "joint":
            continue
        runs = by_method[m]
        finals = np.array([r[-1] for r in runs])
        mean = finals.mean(axis=0)
        std = finals.std(axis=0)
        x = np.arange(T)
        ax.plot(x, mean, marker="o", label=m)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("task index t")
    ax.set_ylabel("accuracy on task t after all 10 tasks")
    ax.set_title("Final per-task accuracy")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "final_per_task.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    by_method = load_runs()
    if not by_method:
        print("no results found in results/*.json")
        return
    plot_running_avg(by_method)
    plot_final_per_task(by_method)


if __name__ == "__main__":
    main()
