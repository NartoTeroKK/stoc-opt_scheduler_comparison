"""
LR schedulers - get_scheduler() factory + NoneScheduler baseline.
"""
from __future__ import annotations

import inspect
from typing import Any

from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR, CyclicLR, OneCycleLR


class NoneScheduler(_LRScheduler):
    """Baseline scheduler: keeps learning rate constant."""

    def get_lr(self) -> list[float | Tensor]:
        return list(self.base_lrs)


_SCHEDULER_REGISTRY: dict[str, type[Any]] = {
    "none": NoneScheduler,
    "exponential": ExponentialLR,
    "cosine": CosineAnnealingLR,
    "cyclic": CyclicLR,
    "one-cycle": OneCycleLR,
}


def get_scheduler(optimizer: optim.Optimizer, name: str, **kwargs) -> _LRScheduler:
    """
    Create a learning rate scheduler by name.

    Filters kwargs to only include parameters accepted by the scheduler constructor.
    """
    if name not in _SCHEDULER_REGISTRY:
        available = list(_SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler: '{name}'. Available: {available}")

    scheduler_cls = _SCHEDULER_REGISTRY[name]
    valid_params = inspect.signature(scheduler_cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return scheduler_cls(optimizer, **filtered)
