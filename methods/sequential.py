"""Sequential fine-tuning baseline. Uses default Method.loss (seen-class-masked CE)."""

from __future__ import annotations

from .base import Method


class Sequential(Method):
    name = "sequential"
