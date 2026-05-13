"""LwF (Learning without Forgetting; Li & Hoiem 2017).

Same KD machinery as Pseudo, but distill on CURRENT-TASK real features
instead of random-pixel features. This is the cleanest comparator for
the pretrained-backbone-prior hypothesis (SPEC §3.5).
"""

from __future__ import annotations

import numpy as np
import torch

from ._distill import kd_loss_old_classes
from .base import Method


class LwF(Method):
    name = "lwf"

    def __init__(self, alpha: float = 1.0, temperature: float = 1.0, **kw):
        super().__init__(**kw)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_w: torch.Tensor | None = None
        self.teacher_b: torch.Tensor | None = None
        self.old_classes: np.ndarray = np.array([], dtype=np.int64)

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

    def loss(self, x, y, seen_classes, current_classes):
        ce = super().loss(x, y, seen_classes, current_classes)
        if self.teacher_w is None or len(self.old_classes) == 0:
            return ce
        kd = kd_loss_old_classes(
            self.head, self.teacher_w, self.teacher_b, x, self.old_classes, self.temperature
        )
        return ce + self.alpha * kd
