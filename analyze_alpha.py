"""Aggregate alpha-sweep results into a (method, alpha) -> ACC mean+std table.
Also produces a figure: ACC vs alpha for each method.
"""

from __future__ import annotations

import glob
import json
import re
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


ALPHA_RE = re.compile(r"_alpha([0-9.]+)\.json$")


def main() -> None:
    by_key: dict[tuple[str, float], list[float]] = defaultdict(list)

    for path in sorted(glob.glob(str(RESULTS / "*_alpha*.json"))):
        with open(path) as f:
            r = json.load(f)
        m = r.get("method")
        if not m:
            continue
        match = ALPHA_RE.search(Path(path).name)
        if not match:
            continue
        alpha = float(match.group(1))
        A = np.array(r["accuracy_matrix"])
        final_acc = float(np.nanmean(A[-1]))
        by_key[(m, alpha)].append(final_acc)

    if not by_key:
        print("no alpha-sweep results found (results/*_alpha*.json)")
        return

    methods = sorted({k[0] for k in by_key})
    alphas = sorted({k[1] for k in by_key})

    print(f"{'method':<16}" + "".join(f"alpha={a:>5.1f}    " for a in alphas))
    print("-" * (16 + 14 * len(alphas)))
    for m in methods:
        row = f"{m:<16}"
        for a in alphas:
            vals = by_key.get((m, a), [])
            if not vals:
                row += f"{'-':>10}    "
            else:
                arr = np.array(vals)
                row += f"{arr.mean():>5.3f}±{arr.std():.3f}  "
        print(row)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        xs, means, stds = [], [], []
        for a in alphas:
            vals = by_key.get((m, a), [])
            if not vals:
                continue
            xs.append(a)
            arr = np.array(vals)
            means.append(arr.mean())
            stds.append(arr.std())
        xs, means, stds = np.array(xs), np.array(means), np.array(stds)
        ax.errorbar(xs, means, yerr=stds, marker="o", label=m, capsize=3)
    ax.axhline(0.529, color="C0", linestyle="--", linewidth=1, label="replay (0.529)")
    ax.axhline(0.788, color="black", linestyle="--", linewidth=1, label="joint (0.788)")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (KD weight)")
    ax.set_ylabel("final ACC")
    ax.set_title("KD weight sweep on pseudo / pseudo_oracle / lwf")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "alpha_sweep.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
