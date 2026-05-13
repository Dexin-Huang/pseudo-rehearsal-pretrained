"""Shared knowledge-distillation helpers for Pseudo and LwF."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def kd_loss_old_classes(
    student_head: torch.nn.Linear,
    teacher_w: torch.Tensor,
    teacher_b: torch.Tensor,
    samples: torch.Tensor,
    old_classes: np.ndarray,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distill teacher onto student on `samples`, restricted to `old_classes` logits.

    Direction: teacher -> student soft-label CE:
        loss = - sum_c q_teacher[c] * log p_student[c]
    """
    if len(old_classes) == 0:
        return samples.new_zeros(())

    idx = torch.from_numpy(old_classes).to(samples.device).long()

    with torch.no_grad():
        t_logits = F.linear(samples, teacher_w, teacher_b)[:, idx] / temperature
        q = F.softmax(t_logits, dim=-1)

    s_logits = student_head(samples)[:, idx] / temperature
    log_p = F.log_softmax(s_logits, dim=-1)
    return -(q * log_p).sum(dim=-1).mean()
