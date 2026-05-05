"""
Configuration module - YAML config loader + typed ExperimentConfig dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SchedulerParams:
    """Parameters for a single scheduler."""
    gamma: float = 0.95
    T_max: int = 50
    eta_min: float = 1e-6
    base_lr: float = 0.001
    max_lr: float = 0.1
    step_size_up: int = 10
    mode: str = "triangular2"
    pct_start: float = 0.3
    anneal_strategy: str = "cos"


@dataclass(frozen=True)
class ExperimentDef:
    """Definition of a single experiment (model + dataset pair)."""
    model: str
    dataset: str
    batch_size: int


@dataclass(frozen=True)
class Paths:
    """Output directory paths."""
    data_dir: str = "data/processed"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Top-level experiment configuration loaded from YAML.

    Usage:
        config = load_config("configs/experiment_config.yaml")
        config.seeds          # [42, 88, 123, 256, 333]
        config.experiments    # {"convex": ExperimentDef(...), ...}
    """
    seeds: tuple[int, ...] = (42, 88, 123, 256, 333)
    test_size: float = 0.2
    batch_size: int = 128
    epochs: int = 100
    lr: dict[str, float] = field(default_factory=lambda: {"sgd": 0.001, "adam": 0.001})
    scheduler_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    paths: Paths = field(default_factory=Paths)
    experiments: dict[str, ExperimentDef] = field(default_factory=dict)
    schedulers: tuple[str, ...] = ("none", "cosine", "exponential", "cyclic", "one-cycle")
    optimizers: tuple[str, ...] = ("sgd", "adam")
    experiment_name: str = "scheduler_comparison"


def _parse_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert raw YAML dict into ExperimentConfig-compatible kwargs."""
    parsed: dict[str, Any] = {}

    parsed["seeds"] = tuple(raw.get("seeds", [42]))
    parsed["test_size"] = float(raw.get("test_size", 0.2))
    parsed["batch_size"] = int(raw.get("batch_size", 128))
    parsed["epochs"] = int(raw.get("epochs", 100))
    # Learning rate - can be a single float or a dict with per-optimizer values
    lr_raw = raw.get("lr", 0.001)
    if isinstance(lr_raw, dict):
        parsed["lr"] = {str(k): float(v) for k, v in lr_raw.items()}
    else:
        parsed["lr"] = float(lr_raw)
    parsed["schedulers"] = tuple(raw.get("schedulers", ["none"]))
    parsed["optimizers"] = tuple(raw.get("optimizers", ["sgd"]))
    parsed["experiment_name"] = raw.get("experiment_name", "scheduler_comparison")

    # Scheduler params - convert numeric strings to floats
    sched_params_raw = raw.get("scheduler_params", {})
    parsed["scheduler_params"] = {}
    for sched_name, params in sched_params_raw.items():
        if params is None:
            parsed["scheduler_params"][sched_name] = {}
            continue
        if not isinstance(params, dict):
            continue  # Skip non-dict entries (e.g., anneal_strategy at wrong level)
        cleaned = {}
        for k, v in params.items():
            if v is None:
                continue  # Skip None values (e.g., total_steps placeholder)
            if isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    pass
            cleaned[k] = v
        parsed["scheduler_params"][sched_name] = cleaned

    # Paths
    paths_raw = raw.get("paths", {})
    parsed["paths"] = Paths(
        data_dir=paths_raw.get("data_dir", "data/processed"),
        reports_dir=paths_raw.get("reports_dir", "reports"),
        figures_dir=paths_raw.get("figures_dir", "reports/figures"),
    )

    # Experiments
    exp_raw = raw.get("experiments", {})
    parsed["experiments"] = {
    name: ExperimentDef(
        model=defn["model"],
        dataset=defn["dataset"],
        batch_size=int(defn.get("batch_size", 128)),  # ← per-esperimento con fallback
    )
    for name, defn in exp_raw.items()
}

    return parsed


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    kwargs = _parse_raw(raw)
    return ExperimentConfig(**kwargs)
