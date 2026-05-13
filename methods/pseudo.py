"""Pseudo-rehearsal (Robins 1995, faithful):
random pixels -> frozen backbone -> old head -> pseudo-soft-targets,
distilled onto the student. Uses a pre-cached random-feature pool to skip
the backbone at training time.
"""

from __future__ import annotations

import numpy as np
import torch

from data import load_random_pool

from ._distill import kd_loss_old_classes
from .base import Method


class Pseudo(Method):
    name = "pseudo"

    def __init__(
        self,
        alpha: float = 1.0,
        pool_size: int = 100_000,
        pool_seed: int = 0,
        temperature: float = 1.0,
        **kw,
    ):
        super().__init__(**kw)
        self.alpha = alpha
        self.pool_size = pool_size
        self.pool_seed = pool_seed
        self.temperature = temperature

        self.teacher_w: torch.Tensor | None = None
        self.teacher_b: torch.Tensor | None = None
        self.old_classes: np.ndarray = np.array([], dtype=np.int64)
        self._pool: torch.Tensor | None = None

    def _ensure_pool(self) -> torch.Tensor:
        if self._pool is None:
            self._pool = torch.from_numpy(load_random_pool(self.pool_size, self.pool_seed)).float()
        return self._pool

    def begin_task(self, task_k, current_classes, seen_classes):
        if task_k == 0:
            self.teacher_w = None
            self.teacher_b = None
            self.old_classes = np.array([], dtype=np.int64)
        else:
            self.teacher_w = self.head.weight.detach().clone()
            self.teacher_b = self.head.bias.detach().clone()
            cur = set(int(c) for c in current_classes)
            self.old_classes = np.array(
                [int(c) for c in seen_classes if int(c) not in cur], dtype=np.int64
            )

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
