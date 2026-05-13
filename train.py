"""Generic training loop. Reads a config, runs 10 tasks (or 1 for joint),
populates the A[k][t] accuracy matrix, writes a JSON result.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import methods
from data import (
    CachedFeatureSet,
    N_CLASSES,
    N_TASKS,
    TaskSplit,
    load_cifar100_cached,
)

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class TrainConfig:
    method: str
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    method_kwargs: dict | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_accuracy_matrix(
    method: methods.Method,
    test_x: np.ndarray,
    test_y: np.ndarray,
    split: TaskSplit,
    up_to_task: int,
    device: torch.device,
) -> np.ndarray:
    """Return a (N_TASKS,) row of accuracies on tasks 0..up_to_task, NaN for unseen."""
    seen = split.seen_classes(up_to_task)
    method.eval()
    row = np.full(N_TASKS, np.nan, dtype=np.float64)
    test_x_t = torch.from_numpy(test_x).float().to(device)
    test_y_t = torch.from_numpy(test_y).long().to(device)
    with torch.no_grad():
        preds = method.predict(test_x_t, seen).cpu().numpy()
    for t in range(up_to_task + 1):
        cls_t = split.task_classes[t]
        mask = np.isin(test_y, cls_t)
        if mask.sum() == 0:
            continue
        row[t] = float((preds[mask] == test_y[mask]).mean())
    method.train()
    return row


def train_one_task(
    method: methods.Method,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    labels: np.ndarray,
    seen: np.ndarray,
    current: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    ds = CachedFeatureSet(features, labels, class_ids=current)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    method.train()
    for _ in range(cfg.epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = method.loss(x, y, seen_classes=seen, current_classes=current)
            loss.backward()
            optimizer.step()


def run(cfg: TrainConfig) -> tuple[dict, methods.Method]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(cfg.seed)

    train_x, train_y, test_x, test_y = load_cifar100_cached()
    split = TaskSplit.from_seed(cfg.seed)
    method = methods.build(cfg.method, **(cfg.method_kwargs or {})).to(device)
    optimizer = torch.optim.AdamW(
        method.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    A = np.full((N_TASKS, N_TASKS), np.nan, dtype=np.float64)
    t0 = time.time()

    if cfg.method == "joint":
        # Single pass over all 100 classes, no continual aspect.
        seen = split.permutation  # all 100 classes
        method.begin_task(task_k=N_TASKS - 1, current_classes=seen, seen_classes=seen)
        train_one_task(method, optimizer, train_x, train_y, seen, seen, cfg, device)
        method.end_task(N_TASKS - 1, train_x, train_y, seen, seen)
        row = evaluate_accuracy_matrix(method, test_x, test_y, split, N_TASKS - 1, device)
        A[N_TASKS - 1] = row
    else:
        for k in range(N_TASKS):
            current = split.task_classes[k]
            seen = split.seen_classes(k)
            method.begin_task(task_k=k, current_classes=current, seen_classes=seen)
            train_one_task(method, optimizer, train_x, train_y, seen, current, cfg, device)
            method.end_task(k, train_x, train_y, seen, current)
            A[k] = evaluate_accuracy_matrix(method, test_x, test_y, split, k, device)

    elapsed = time.time() - t0

    result = {
        "method": cfg.method,
        "config": asdict(cfg),
        "accuracy_matrix": A.tolist(),
        "elapsed_sec": elapsed,
    }
    return result, method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None, help="override seed in config")
    parser.add_argument("--tag", type=str, default="", help="extra tag for output filename")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="override method_kwargs.alpha (for KD-method sweeps)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.alpha is not None:
        cfg.method_kwargs = dict(cfg.method_kwargs or {})
        cfg.method_kwargs["alpha"] = args.alpha

    print(f"running {cfg.method} seed={cfg.seed}")
    result, method = run(cfg)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"{cfg.method}_seed{cfg.seed}{suffix}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}  ({result['elapsed_sec']:.1f}s)")

    if cfg.method == "joint":
        ckpt = RESULTS_DIR / f"joint_seed{cfg.seed}{suffix}_head.pt"
        torch.save(
            {"w": method.head.weight.detach().cpu(), "b": method.head.bias.detach().cpu()},
            ckpt,
        )
        print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
