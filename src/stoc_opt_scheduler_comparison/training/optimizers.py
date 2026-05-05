"""
Optimizers - get_optimizer() factory for SGD and Adam.
"""
from __future__ import annotations

import torch.nn as nn
import torch.optim as optim


_OPTIMIZER_REGISTRY: dict[str, type[optim.Optimizer]] = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
}


def get_optimizer(model: nn.Module, name: str, lr: float, **kwargs) -> optim.Optimizer:
    """Create an optimizer by name."""
    if name not in _OPTIMIZER_REGISTRY:
        available = list(_OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer: '{name}'. Available: {available}")

    optimizer_cls = _OPTIMIZER_REGISTRY[name]
    return optimizer_cls(model.parameters(), lr=lr, **kwargs) # type: ignore
