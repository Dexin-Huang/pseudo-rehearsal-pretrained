"""Joint training upper-bound: all 100 classes at once, no continual aspect.

train.py recognizes joint specially: it skips the per-task loop and feeds the
full CIFAR-100 train set in one pass, with seen_classes = all 100.
"""

from __future__ import annotations

from .base import Method


class Joint(Method):
    name = "joint"
