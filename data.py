"""Cached feature datasets + class-permutation task splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CACHE_DIR = Path(__file__).parent / "cache"
N_CLASSES = 100
N_TASKS = 10
CLASSES_PER_TASK = N_CLASSES // N_TASKS


@dataclass
class TaskSplit:
    """Class permutation + per-task class-id buckets for one seed."""

    permutation: np.ndarray  # (100,) — original_class_id ordered by the permutation
    task_classes: list[np.ndarray]  # len 10, each (10,) — original class ids in task k

    @classmethod
    def from_seed(cls, seed: int) -> "TaskSplit":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N_CLASSES)
        task_classes = [perm[k * CLASSES_PER_TASK : (k + 1) * CLASSES_PER_TASK] for k in range(N_TASKS)]
        return cls(permutation=perm, task_classes=task_classes)

    def seen_classes(self, task_k: int) -> np.ndarray:
        """Original class ids visible by the end of task k (inclusive)."""
        return np.concatenate(self.task_classes[: task_k + 1])


class CachedFeatureSet(Dataset):
    """Wraps a cached (features, labels) pair, optionally filtered to a class subset."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, class_ids: np.ndarray | None = None):
        if class_ids is not None:
            mask = np.isin(labels, class_ids)
            features = features[mask]
            labels = labels[mask]
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_cifar100_cached() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_features, train_labels, test_features, test_labels)."""
    train_x = np.load(CACHE_DIR / "cifar100_train_features.npy")
    train_y = np.load(CACHE_DIR / "cifar100_train_labels.npy")
    test_x = np.load(CACHE_DIR / "cifar100_test_features.npy")
    test_y = np.load(CACHE_DIR / "cifar100_test_labels.npy")
    return train_x, train_y, test_x, test_y


def load_random_pool(n: int = 100_000, seed: int = 0) -> np.ndarray:
    return np.load(CACHE_DIR / f"random_pixel_features_n{n}_seed{seed}.npy")


def load_stl10_pool() -> np.ndarray:
    return np.load(CACHE_DIR / "stl10_unlabeled_features.npy")
