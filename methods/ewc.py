"""Online EWC (Kirkpatrick 2017 + Schwarz 2018 online variant).

Accumulates diagonal Fisher across tasks; stores theta* at most recent task end.
Penalty: lam * sum_p F_p * (theta_p - theta*_p)^2.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CachedFeatureSet

from .base import Method, seen_class_mask


class EWC(Method):
    name = "ewc"

    def __init__(self, lam: float = 100.0, fisher_batch_size: int = 128, **kw):
        super().__init__(**kw)
        self.lam = lam
        self.fisher_bs = fisher_batch_size
        self.register_buffer("fisher_w", torch.zeros_like(self.head.weight))
        self.register_buffer("fisher_b", torch.zeros_like(self.head.bias))
        self.register_buffer("star_w", torch.zeros_like(self.head.weight))
        self.register_buffer("star_b", torch.zeros_like(self.head.bias))
        self.has_prior = False

    def loss(self, x, y, seen_classes, current_classes):
        ce = super().loss(x, y, seen_classes, current_classes)
        if not self.has_prior:
            return ce
        pen = (self.fisher_w * (self.head.weight - self.star_w).pow(2)).sum()
        pen = pen + (self.fisher_b * (self.head.bias - self.star_b).pow(2)).sum()
        return ce + self.lam * pen

    @torch.no_grad()
    def _snapshot(self) -> None:
        self.star_w.copy_(self.head.weight.detach())
        self.star_b.copy_(self.head.bias.detach())

    def end_task(self, task_k, train_features, train_labels, seen_classes, current_classes):
        # Empirical Fisher diagonal in closed form for a (masked) linear-CE head:
        #   For example (x_i, y_i): d/dW_{c,j} (-log p(y_i|x_i)) = (p_i[c] - 1{c==y_i}) * x_i[j]
        #   F_W[c, j] = mean_i (p_i[c] - 1{c==y_i})^2 * x_i[j]^2
        #   F_b[c]    = mean_i (p_i[c] - 1{c==y_i})^2
        # This is the correct per-example mean-of-squared-grads, not (mean-grad)^2.
        device = self.head.weight.device
        ds = CachedFeatureSet(train_features, train_labels, class_ids=current_classes)
        loader = DataLoader(ds, batch_size=self.fisher_bs, shuffle=False)

        fw = torch.zeros_like(self.head.weight)
        fb = torch.zeros_like(self.head.bias)
        n = 0
        mask = seen_class_mask(self.n_classes, seen_classes, device)
        with torch.no_grad():
            self.eval()
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = self.head(x) + mask
                p = F.softmax(logits, dim=-1)
                one_hot = F.one_hot(y, num_classes=self.n_classes).float()
                delta = (p - one_hot).pow(2)  # (B, C)
                fw += delta.t() @ x.pow(2)    # (C, D)
                fb += delta.sum(dim=0)        # (C,)
                n += x.shape[0]
            self.train()
        fw /= max(n, 1)
        fb /= max(n, 1)

        # Online EWC: sum across tasks, single theta* anchor (most recent end-of-task).
        self.fisher_w += fw
        self.fisher_b += fb
        self._snapshot()
        self.has_prior = True
