"""Cache ViT-S/16 features for CIFAR-100 + a pool of random-pixel features.

One-time setup. Writes to cache/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

CACHE_DIR = Path(__file__).parent / "cache"
DATA_DIR = Path(__file__).parent / "cache" / "torchvision"

MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_backbone(device: torch.device) -> torch.nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def cache_cifar100(model: torch.nn.Module, device: torch.device, batch_size: int) -> None:
    transform = build_transform()
    for split, train_flag in [("train", True), ("test", False)]:
        out_x = CACHE_DIR / f"cifar100_{split}_features.npy"
        out_y = CACHE_DIR / f"cifar100_{split}_labels.npy"
        if out_x.exists() and out_y.exists():
            print(f"[skip] {out_x.name} already exists")
            continue

        ds = datasets.CIFAR100(root=DATA_DIR, train=train_flag, download=True, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        feats, labels = [], []
        for x, y in tqdm(loader, desc=f"cifar100 {split}"):
            x = x.to(device, non_blocking=True)
            f = model(x)
            feats.append(f.half().cpu())
            labels.append(y)
        feats = torch.cat(feats).numpy()
        labels = torch.cat(labels).numpy().astype(np.int64)
        np.save(out_x, feats)
        np.save(out_y, labels)
        print(f"[save] {out_x.name}: {feats.shape} {feats.dtype}")


@torch.no_grad()
def cache_stl10_unlabeled(model: torch.nn.Module, device: torch.device, batch_size: int) -> None:
    """Cache ViT features on STL-10 unlabeled (100K natural OOD images for CIFAR-100).
    Used as an alternative pseudo-rehearsal input distribution (SPEC extension)."""
    out_path = CACHE_DIR / "stl10_unlabeled_features.npy"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    transform = build_transform()
    ds = datasets.STL10(root=DATA_DIR, split="unlabeled", download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    feats = []
    for x, _y in tqdm(loader, desc="stl10 unlabeled"):
        x = x.to(device, non_blocking=True)
        feats.append(model(x).half().cpu())
    feats = torch.cat(feats).numpy()
    np.save(out_path, feats)
    print(f"[save] {out_path.name}: {feats.shape} {feats.dtype}")


@torch.no_grad()
def cache_random_pool(
    model: torch.nn.Module,
    device: torch.device,
    n: int,
    batch_size: int,
    seed: int,
) -> None:
    out_path = CACHE_DIR / f"random_pixel_features_n{n}_seed{seed}.npy"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    g = torch.Generator(device="cpu").manual_seed(seed)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    feats = []
    n_done = 0
    with tqdm(total=n, desc="random pool") as pbar:
        while n_done < n:
            bs = min(batch_size, n - n_done)
            # Uniform [0,1] pixels iid per channel, then ImageNet-normalize.
            x = torch.rand(bs, 3, IMG_SIZE, IMG_SIZE, generator=g)
            x = (x - mean) / std
            x = x.to(device, non_blocking=True)
            f = model(x)
            feats.append(f.half().cpu())
            n_done += bs
            pbar.update(bs)
    feats = torch.cat(feats).numpy()
    np.save(out_path, feats)
    print(f"[save] {out_path.name}: {feats.shape} {feats.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-pool-size", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--with-stl10", action="store_true", help="also cache STL-10 unlabeled features")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_backbone(device)
    print(f"backbone: {MODEL_NAME}")

    cache_cifar100(model, device, args.batch_size)
    if not args.skip_random:
        cache_random_pool(model, device, args.random_pool_size, args.batch_size, args.random_seed)
    if args.with_stl10:
        cache_stl10_unlabeled(model, device, args.batch_size)

    print("done.")


if __name__ == "__main__":
    main()
