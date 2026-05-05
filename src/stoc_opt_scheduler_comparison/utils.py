"""
Common utilities - Shared patterns across the project.
"""
from __future__ import annotations

import logging
from typing import Callable, Any
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


# ── Factory Registry ────────────────────────────────────────────────────────────────

class FactoryRegistry:
    """Generic factory pattern with registry."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Callable] = {}
        self.logger = setup_logger(name)
    
    def register(self, key: str, factory: Callable) -> None:
        """Register a factory function."""
        self._registry[key] = factory
    
    def create(self, key: str, *args, **kwargs) -> Any:
        """Create instance by key."""
        if key not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown {self.name}: {key}. Available: {available}")
        return self._registry[key](*args, **kwargs)
    
    def keys(self) -> list[str]:
        """List registered keys."""
        return list(self._registry.keys())
    
    def __contains__(self, key: str) -> bool:
        return key in self._registry


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