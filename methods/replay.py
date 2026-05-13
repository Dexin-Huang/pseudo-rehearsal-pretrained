"""Class-balanced experience replay: 5 examples per seen class (500 total at end)."""

from __future__ import annotations

import numpy as np
import torch

from .base import Method


class Replay(Method):
    name = "replay"

    def __init__(self, examples_per_class: int = 5, mix_ratio: float = 0.5, **kw):
        super().__init__(**kw)
        self.epc = examples_per_class
        self.mix = mix_ratio
        self.buf_x: torch.Tensor | None = None  # (M, 384) on CPU
        self.buf_y: torch.Tensor | None = None  # (M,) on CPU

    def loss(self, x, y, seen_classes: np.ndarray, current_classes: np.ndarray):
        # SPEC §3.3: each training batch is `mix_ratio`-from-buffer / (1-mix_ratio)-from-current,
        # keeping total batch size constant. Subsample current and sample buffer to hit the mix.
        if self.buf_x is not None and len(self.buf_x) > 0:
            bs = x.shape[0]
            n_replay = max(1, min(bs - 1, int(round(bs * self.mix))))
            n_current = bs - n_replay
            cur_idx = torch.randperm(bs, device=x.device)[:n_current]
            x_cur = x[cur_idx]
            y_cur = y[cur_idx]
            buf_idx = torch.randint(0, len(self.buf_x), (n_replay,))
            bx = self.buf_x[buf_idx].to(x.device, non_blocking=True)
            by = self.buf_y[buf_idx].to(y.device, non_blocking=True)
            x = torch.cat([x_cur, bx], dim=0)
            y = torch.cat([y_cur, by], dim=0)
        return super().loss(x, y, seen_classes, current_classes)

    def end_task(self, task_k, train_features, train_labels, seen_classes, current_classes):
        rng = np.random.default_rng(seed=10_000 + task_k)
        chunks_x, chunks_y = [], []
        for c in current_classes:
            mask = train_labels == c
            idx = np.where(mask)[0]
            take = rng.choice(idx, size=min(self.epc, len(idx)), replace=False)
            chunks_x.append(train_features[take])
            chunks_y.append(train_labels[take])
        new_x = torch.from_numpy(np.concatenate(chunks_x, axis=0)).float()
        new_y = torch.from_numpy(np.concatenate(chunks_y, axis=0)).long()
        if self.buf_x is None:
            self.buf_x, self.buf_y = new_x, new_y
        else:
            self.buf_x = torch.cat([self.buf_x, new_x], dim=0)
            self.buf_y = torch.cat([self.buf_y, new_y], dim=0)
