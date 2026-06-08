"""
Common utilities - Shared patterns across the project.
"""

from __future__ import annotations

import logging
from typing import Callable
from functools import wraps


# ── Logging Configuration ───────────────────────────────────────────────────────


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(f"[{name}] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ── Inference Decorator ──────────────────────────────────────────────────────


def inference_mode(func: Callable) -> Callable:
    """Decorator for evaluation functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        import torch

        with torch.inference_mode():
            return func(*args, **kwargs)

    return wrapper


# ── Logger Instances ───────────────────────────────────────────────────────────────

data_logger = setup_logger("data")
model_logger = setup_logger("model")
train_logger = setup_logger("train")
scheduler_logger = setup_logger("scheduler")
metrics_logger = setup_logger("metrics")
tracking_logger = setup_logger("tracking")
viz_logger = setup_logger("viz")
