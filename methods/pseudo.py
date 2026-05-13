"""Pseudo-rehearsal (Robins 1995, faithful):
random pixels -> frozen backbone -> old head -> pseudo-soft-targets, distilled onto the student.

Two orthogonal extensions for the 2x2 ablation (SPEC §3.4 + extension):
- `pool_source`: 'random' (uniform-pixel features) or 'stl10' (natural OOD features)
- `oracle_teacher_ckpt`: None (CL-snapshot teacher, faithful) or path to a joint head
  checkpoint (oracle teacher; information leakage but a clean failure-isolation diagnostic)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data import load_random_pool, load_stl10_pool

from ._distill import kd_loss_old_classes
from .base import Method


class Pseudo(Method):
    name = "pseudo"

    def __init__(
        self,
        alpha: float = 1.0,
        pool_source: str = "random",
        pool_size: int = 100_000,
        pool_seed: int = 0,
        oracle_teacher_ckpt: str | None = None,
        temperature: float = 1.0,
        **kw,
    ):
        super().__init__(**kw)
        if pool_source not in {"random", "stl10"}:
            raise ValueError(f"pool_source must be 'random' or 'stl10', got {pool_source!r}")
        self.alpha = alpha
        self.pool_source = pool_source
        self.pool_size = pool_size
        self.pool_seed = pool_seed
        self.oracle_teacher_ckpt = oracle_teacher_ckpt
        self.temperature = temperature

        self.teacher_w: torch.Tensor | None = None
        self.teacher_b: torch.Tensor | None = None
        self.old_classes: np.ndarray = np.array([], dtype=np.int64)
        self._pool: torch.Tensor | None = None
        self._oracle_loaded = False

    def _ensure_pool(self) -> torch.Tensor:
        if self._pool is None:
            if self.pool_source == "random":
                arr = load_random_pool(self.pool_size, self.pool_seed)
            else:
                arr = load_stl10_pool()
            self._pool = torch.from_numpy(arr).float()
        return self._pool

    def _load_oracle_once(self, device: torch.device) -> None:
        if self._oracle_loaded:
            return
        state = torch.load(Path(self.oracle_teacher_ckpt), map_location=device)
        self.teacher_w = state["w"].to(device)
        self.teacher_b = state["b"].to(device)
        self._oracle_loaded = True

    def begin_task(self, task_k, current_classes, seen_classes):
        cur = set(int(c) for c in current_classes)
        self.old_classes = np.array(
            [int(c) for c in seen_classes if int(c) not in cur], dtype=np.int64
        )

        if self.oracle_teacher_ckpt is not None:
            # Static oracle: loaded once on first call; never updated.
            self._load_oracle_once(self.head.weight.device)
        elif task_k == 0:
            # CL teacher, but no prior task -> no teacher yet.
            self.teacher_w = None
            self.teacher_b = None
        else:
            # CL teacher snapshot at start of new task = head as of end of previous task.
            self.teacher_w = self.head.weight.detach().clone()
            self.teacher_b = self.head.bias.detach().clone()

    def _draw_samples(self, n: int, device: torch.device) -> torch.Tensor:
        pool = self._ensure_pool()
        idx = torch.randint(0, len(pool), (n,))
        return pool[idx].to(device, non_blocking=True)

    def loss(self, x, y, seen_classes, current_classes):
        ce = super().loss(x, y, seen_classes, current_classes)
        if self.teacher_w is None or len(self.old_classes) == 0:
            return ce
        samples = self._draw_samples(x.shape[0], x.device)
        kd = kd_loss_old_classes(
            self.head, self.teacher_w, self.teacher_b, samples, self.old_classes, self.temperature
        )
        return ce + self.alpha * kd
