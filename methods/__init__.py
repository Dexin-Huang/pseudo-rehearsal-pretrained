"""Continual-learning methods. Each method subclasses Method (see base.py)."""

from typing import Callable

from .base import Method
from .ewc import EWC
from .joint import Joint
from .lwf import LwF
from .pseudo import Pseudo
from .replay import Replay
from .sequential import Sequential


# Each entry is a callable that returns a Method instance.
# The 2x2 pseudo ablation lives here as factories that fix defaults; users can
# still override via method_kwargs.
REGISTRY: dict[str, Callable[..., Method]] = {
    "sequential": Sequential,
    "joint": Joint,
    "replay": Replay,
    "ewc": EWC,
    "lwf": LwF,
    "pseudo": Pseudo,
    "pseudo_oracle": lambda **kw: Pseudo(
        **{"oracle_teacher_ckpt": "results/joint_seed0_head.pt", "pool_source": "random", **kw}
    ),
    "pseudo_natural": lambda **kw: Pseudo(**{"pool_source": "stl10", **kw}),
    "pseudo_oracle_natural": lambda **kw: Pseudo(
        **{"oracle_teacher_ckpt": "results/joint_seed0_head.pt", "pool_source": "stl10", **kw}
    ),
}


def build(name: str, **cfg) -> Method:
    try:
        factory = REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"unknown method '{name}'. Registered: {sorted(REGISTRY)}") from e
    return factory(**cfg)
