"""
LR schedulers - get_scheduler() factory + NoneScheduler baseline.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
    OneCycleLR,
)


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


def get_dynamic_scheduler_params(
    scheduler_name: str,
    lr: float,
    epochs: int,
    steps_per_epoch: int,
    base_sched_params: Optional[Dict[str, Any]] = None,
    max_lr_factor: float = 10.0,
    exponential_target_drop: float = 10.0,
) -> Dict[str, Any]:
    """
    Calcola e inietta i parametri dinamici e contestuali per gli scheduler di PyTorch.

    Args:
        scheduler_name: Nome dello scheduler (es. "exponential", "one-cycle").
        lr: Learning rate base dell'ottimizzatore.
        epochs: Numero totale di epoche di training.
        steps_per_epoch: Numero di batch (iterazioni) per singola epoca.
        base_sched_params: Eventuali parametri statici caricati dal file YAML.
        max_lr_factor: Fattore di moltiplicazione per calcolare il massimo learning rate.
        exponential_target_drop: Fattore di riduzione per lo scheduler esponenziale.
    Returns:
        Un dizionario contenente tutti i parametri necessari per inizializzare lo scheduler.
    """
    # Usiamo .copy() per evitare modifiche in-place inaspettate al dizionario originale
    sched_params = base_sched_params.copy() if base_sched_params else {}

    # ─── CALCOLO PROGRAMMATICO MAX_LR (10x rispetto al LR base) ───────
    computed_max_lr = lr * max_lr_factor
    # ──────────────────────────────────────────────────────────────────

    if scheduler_name == "exponential":
        target_drop = (
            exponential_target_drop  # Target drop factor (10x reduction of initial lr)
        )
        gamma = (1.0 / target_drop) ** (1.0 / epochs)
        sched_params["gamma"] = gamma

    elif scheduler_name == "cosine":
        sched_params["T_max"] = epochs

    elif scheduler_name == "one-cycle":
        sched_params["max_lr"] = computed_max_lr
        sched_params["epochs"] = epochs
        sched_params["steps_per_epoch"] = steps_per_epoch

    elif scheduler_name == "cyclic":
        sched_params["base_lr"] = lr
        sched_params["max_lr"] = computed_max_lr
        sched_params["step_size_up"] = 2 * steps_per_epoch

    return sched_params
