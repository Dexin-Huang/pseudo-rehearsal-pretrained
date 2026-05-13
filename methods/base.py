"""Shared CL method interface.

A Method owns the 384->100 linear head and any auxiliary state (Fisher,
buffer, teacher). The training loop in train.py calls:

    method.begin_task(task_k, current_classes, seen_classes)
    for epoch in range(E):
        for x, y in loader:
            optim.zero_grad()
            loss = method.loss(x, y, seen_classes, current_classes)
            loss.backward()
            optim.step()
    method.end_task(task_k, train_features, train_labels, seen_classes)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM = 384
N_CLASSES = 100


def seen_class_mask(n_classes: int, seen_classes: np.ndarray, device: torch.device) -> torch.Tensor:
    """Additive logit mask: 0 on seen classes, -inf elsewhere. Shape (n_classes,)."""
    mask = torch.full((n_classes,), float("-inf"), device=device)
    mask[torch.from_numpy(seen_classes).to(device).long()] = 0.0
    return mask


class Method(nn.Module):
    """Base class. Subclasses implement loss() and optional hooks."""

    name: str = "method"

    def __init__(self, n_classes: int = N_CLASSES, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.head = nn.Linear(feat_dim, n_classes)
        self.n_classes = n_classes

    def begin_task(
        self,
        task_k: int,
        current_classes: np.ndarray,
        seen_classes: np.ndarray,
    ) -> None:
        pass

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seen_classes: np.ndarray,
        current_classes: np.ndarray,
    ) -> torch.Tensor:
        """Default: CE with seen-class masking. Override in subclasses to add regularization / KD / replay."""
        logits = self.head(x) + seen_class_mask(self.n_classes, seen_classes, x.device)
        return F.cross_entropy(logits, y)

    def end_task(
        self,
        task_k: int,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        seen_classes: np.ndarray,
        current_classes: np.ndarray,
    ) -> None:
        pass

    def predict(self, x: torch.Tensor, seen_classes: np.ndarray) -> torch.Tensor:
        """Argmax over seen classes only. Returns class ids in [0, n_classes)."""
        logits = self.head(x) + seen_class_mask(self.n_classes, seen_classes, x.device)
        return logits.argmax(dim=-1)
