"""Continual-learning methods. Each method subclasses Method (see base.py)."""

from .base import Method
from .ewc import EWC
from .joint import Joint
from .lwf import LwF
from .pseudo import Pseudo
from .replay import Replay
from .sequential import Sequential

REGISTRY: dict[str, type[Method]] = {
    "sequential": Sequential,
    "joint": Joint,
    "replay": Replay,
    "ewc": EWC,
    "pseudo": Pseudo,
    "lwf": LwF,
}


def build(name: str, **cfg) -> Method:
    try:
        cls = REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"unknown method '{name}'. Registered: {sorted(REGISTRY)}") from e
    return cls(**cfg)
