"""
Visualization module - 4 hierarchical levels + 2 analytical plots + 1 entry point.

All plots follow the visual encoding:
- Scheduler colors: SCHEDULER_COLORS
- Optimizer styles: OPTIMIZER_STYLES
- Seed styles: SEED_STYLE (Level 3 only)

Data sources:
- results: {run_name: {"history": TrainingHistory, "test_metrics": dict}}
- aggregated: {key: {"train_losses_mean": array, "train_losses_std": array, ...}}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from stoc_opt_scheduler_comparison.utils import viz_logger as logger
from stoc_opt_scheduler_comparison.evaluation.convergence import convergence_threshold

# ── Visual Encoding ─────────────────────────────────────────────────────────

SCHEDULER_COLORS: dict[str, str] = {
    "none": "#7f7f7f",    # Gray - neutral baseline
    "exponential": "#1f77b4",  # Blue - monotonic decay
    "cosine": "#2ca02c",    # Green - smooth cyclic
    "cyclic": "#ff7f0e",    # Orange - oscillating
    "one-cycle": "#d62728",  # Red - aggressive warm-up
}

OPTIMIZER_STYLES: dict[str, dict[str, Any]] = {
    "sgd": {"linestyle": "-", "linewidth": 2.0, "alpha_std": 0.05},
    "adam": {"linestyle": "--", "linewidth": 2.0, "alpha_std": 0.05},
}

SEED_STYLE: dict[str, dict[str, float]] = {
    "individual": {"linewidth": 1, "alpha": 0.5},
    "mean": {"linewidth": 2, "alpha": 1.0},
}

SCHEDULERS_LIST=["none", "exponential", "cosine", "cyclic", "one-cycle"]
OPTIMIZERS_LIST=["sgd", "adam"]

# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_run_name(name: str) -> tuple[str, str, str, int] | None:
    """Parse 'convex_cosine_sgd_42' → (problem, scheduler, optimizer, seed)."""
    import re
    match = re.match(r"(convex|non-convex)_([^_]+)_([^_]+)_(\d+)$", name)
    if match:
        return match.group(1), match.group(2), match.group(3), int(match.group(4))
    return None


def _get_scheduler_color(scheduler: str) -> str:
    return SCHEDULER_COLORS.get(scheduler, "#1f77b4")


def _get_optimizer_style(optimizer: str) -> dict[str, Any]:
    return OPTIMIZER_STYLES.get(optimizer, {"linestyle": "-", "linewidth": 2.0, "alpha_std": 0.15})


def _setup_style() -> None:
    """Apply clean academic style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "figure.titlesize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: Figure, path: str | Path) -> None:
    """Save figure and log."""
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")


def _get_aggregated_arrays(aggregated: dict, key: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Extract mean and std arrays for a scheduler-optimizer pair."""
    if key not in aggregated:
        return None, None
    data = aggregated[key]
    mean = data.get("train_losses_mean")
    std = data.get("train_losses_std")
    if mean is not None and std is not None:
        return np.asarray(mean), np.asarray(std)
    return None, None


def _get_seed_arrays(results: dict, scheduler: str, optimizer: str) -> list[np.ndarray]:
    """Extract per-seed arrays for a scheduler-optimizer pair."""
    arrays = []
    for run_name, history in results.items():
        parsed = _parse_run_name(run_name)
        if parsed is None:
            continue
        _, sched, opt, seed = parsed
        if sched == scheduler and opt == optimizer:
            if history and history.train_losses_arr is not None:
                arrays.append(history.train_losses_arr)
    return arrays

def add_figure_title(
    fig: Figure,
    title: str,
    subtitle: str | None = None,
    title_fontsize: int = 18,
    subtitle_fontsize: int = 12,
    gap_inches: float = 0.28,
) -> None:
    fig_height = fig.get_size_inches()[1]

    # Stima altezza testo in inches: fontsize punti → 1 punto ≈ 1/72 inch
    title_height    = title_fontsize    / 72
    subtitle_height = subtitle_fontsize / 72 if subtitle else 0.0

    # Spazio totale necessario = margine superiore + titolo + gap + sottotitolo + piccolo padding
    top_margin_inches = 0.10 + title_height + gap_inches + subtitle_height + 0.05

    title_y    = 1.0 - (0.10 / fig_height)
    subtitle_y = title_y - ((title_height + gap_inches) / fig_height)
    top_margin = 1.0 - (top_margin_inches / fig_height)

    fig.suptitle(title, fontsize=title_fontsize, y=title_y)

    if subtitle is not None:
        fig.text(
            0.5, subtitle_y,
            subtitle,
            ha="center",
            fontsize=subtitle_fontsize,
            style="italic",
            color="gray",
            transform=fig.transFigure,
        )

    fig.tight_layout(rect=(0, 0, 1, top_margin))

# ── Level 1: Global Comparison ─────────────────────────────────────────

def plot_global_comparison(
    results: dict,
    aggregated: dict,
    problem_type: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 1: (2, 1) subplots — loss curve overview + LR schedules.

    Top subplot:    train_loss mean per config (10 curves: 5 schedulers × 2 optimizers).
    Bottom subplot: learning_rate mean per config (same visual encoding, no std band).
                    Y-axis in log scale to highlight schedule shape differences.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # stacked vertically

    # Collect all unique (scheduler, optimizer) configs from run names
    configs = []
    for run_name in results:
        parsed = _parse_run_name(run_name)
        if parsed is None:
            continue
        _, sched, opt, _ = parsed
        if (sched, opt) not in configs:
            configs.append((sched, opt))

    # Top: train_loss mean curves
    ax = axes[0]
    for sched, opt in configs:
        key = f"{opt}_{sched}"
        mean_arr, std_arr = _get_aggregated_arrays(aggregated, key)
        if mean_arr is None:
            continue
        color = _get_scheduler_color(sched)
        style = _get_optimizer_style(opt)
        epochs = np.arange(1, len(mean_arr) + 1)

        ax.plot(epochs, mean_arr, color=color, label=f"{sched} ({opt})",
                linestyle=style["linestyle"], linewidth=style["linewidth"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"{problem_type.upper()} - Loss Curves")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom: learning_rate mean curves (log scale, no std band)
    ax = axes[1]
    for sched, opt in configs:
        key = f"{opt}_{sched}"
        data = aggregated.get(key, {})
        lr_mean = data.get("learning_rates_mean")
        if lr_mean is None:
            continue
        color = _get_scheduler_color(sched)
        style = _get_optimizer_style(opt)
        epochs = np.arange(1, len(lr_mean) + 1)

        ax.plot(epochs, lr_mean, color=color,
                linestyle=style["linestyle"], linewidth=style["linewidth"],
                label=f"{sched} ({opt})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{problem_type.upper()} - LR Schedules")
    ax.set_yscale("log")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    add_figure_title(
        fig,
        title=f"{problem_type.upper()} - Global Comparison",
        subtitle="Top: Training Loss curves | Bottom: Learning Rate schedules (log scale)",
    )

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Level 2: By Optimizer ────────────────────────────────────────────────

def plot_by_optimizer(
    results: dict,
    aggregated: dict,
    problem_type: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 2: 2×2 grid - rows=optimizers (SGD, Adam), cols=linear/log scale.

    Each subplot shows 5 scheduler curves with mean±std from aggregated arrays.
    Left column: linear scale.
    Right column: log scale for loss.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, opt in enumerate(OPTIMIZERS_LIST):
        # Left column: linear scale
        ax_linear = axes[idx * 2]
        for sched in SCHEDULERS_LIST:
            key = f"{opt}_{sched}"
            mean_arr, std_arr = _get_aggregated_arrays(aggregated, key)
            if mean_arr is None:
                continue
            color = _get_scheduler_color(sched)
            style = _get_optimizer_style(opt)
            epochs = np.arange(1, len(mean_arr) + 1)

            ax_linear.plot(epochs, mean_arr, color=color, label=sched,
                          linestyle=style["linestyle"], linewidth=style["linewidth"])
            ax_linear.fill_between(epochs, mean_arr - std_arr, mean_arr + std_arr,
                                   color=color, alpha=style["alpha_std"])

        ax_linear.set_xlabel("Epoch")
        ax_linear.set_ylabel("Training Loss")
        ax_linear.set_title(f"{problem_type.upper()} - {opt.upper()} (Linear Scale)")
        ax_linear.legend(loc="upper right")
        ax_linear.grid(True, alpha=0.3)

        # Right column: log scale
        ax_log = axes[idx * 2 + 1]
        for sched in SCHEDULERS_LIST:
            key = f"{opt}_{sched}"
            mean_arr, std_arr = _get_aggregated_arrays(aggregated, key)
            if mean_arr is None:
                continue
            color = _get_scheduler_color(sched)
            style = _get_optimizer_style(opt)
            epochs = np.arange(1, len(mean_arr) + 1)

            ax_log.plot(epochs, mean_arr, color=color, label=sched,
                        linestyle=style["linestyle"], linewidth=style["linewidth"])
            ax_log.fill_between(epochs, mean_arr - std_arr, mean_arr + std_arr,
                                   color=color, alpha=style["alpha_std"])

        ax_log.set_xlabel("Epoch")
        ax_log.set_ylabel("Training Loss (log scale)")
        ax_log.set_title(f"{problem_type.upper()} - {opt.upper()} (Log Scale)")
        ax_log.set_yscale("log")
        ax_log.legend(loc="upper right")
        ax_log.grid(True, alpha=0.3)

    # Add overall figure title with subtitle
    add_figure_title(
        fig,
        title=f"{problem_type.upper()} - Loss Curves by Optimizer",
        subtitle="Left: Linear scale | Right: Log scale"
    )        

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Level 3: Seed Variance ────────────────────────────────────────────────

def plot_seed_variance(
    results: dict,
    problem_type: str,
    L_star_global: float | None = None, 
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 3: (5×2) grid - diagnostic seed variance.

    Each cell (scheduler, optimizer): 5 thin seed lines + 1 thick mean line.
    Optional dashed convergence threshold line per cell (computed via
    convergence_threshold using the mean L0 of the seed group).
    No rolling mean - raw arrays only.
    """
    _setup_style()

    fig, axes = plt.subplots(5, 2, figsize=(12, 20), sharex=True, sharey=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes.ravel()

    all_loss_values = []

    for idx, (opt, sched) in enumerate(
        [(o, s) for s in SCHEDULERS_LIST for o in OPTIMIZERS_LIST]
    ):
        ax = axes[idx]
        color = _get_scheduler_color(sched)
        style = _get_optimizer_style(opt)

        seed_arrays = _get_seed_arrays(results, sched, opt)

        if seed_arrays:
            for arr in seed_arrays:
                ax.plot(
                    np.arange(1, len(arr) + 1), arr,
                    color=color,
                    linewidth=SEED_STYLE["individual"]["linewidth"],
                    alpha=SEED_STYLE["individual"]["alpha"],
                )
                all_loss_values.extend(arr.tolist())

            stacked  = np.stack(seed_arrays)
            mean_arr = stacked.mean(axis=0)
            ax.plot(
                np.arange(1, len(mean_arr) + 1), mean_arr,
                color=color,
                linewidth=SEED_STYLE["mean"]["linewidth"],
                alpha=SEED_STYLE["mean"]["alpha"],
            )

            # --- Convergence threshold line (per-cell) ---
            if L_star_global is not None:
                L0_ref = float(np.mean([arr[0] for arr in seed_arrays]))
                thresh = convergence_threshold(
                    L0=L0_ref,
                    L_star_global=L_star_global
                )
                if not np.isnan(thresh):
                    ax.axhline(
                        y=thresh,
                        color="black",
                        linestyle=":",
                        linewidth=0.9,
                        alpha=0.8,
                    )
                    # Annotazione con il valore numerico, allineata a destra
                    ax.annotate(
                        f"ε={thresh:.4f}",
                        xy=(0.0, thresh),
                        xycoords=("axes fraction", "data"),
                        xytext=(4, 2),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=6,
                        color="black",
                    )

        ax.set_title(f"{sched} ({opt})", fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx % 2 == 0:
            ax.set_ylabel("Training Loss")

    y_max = np.percentile(all_loss_values, 95) * 1.05
    y_min = max(0.0, np.percentile(all_loss_values, 1) * 0.95)

    for ax in axes:
        ax.set_ylim(y_min, y_max)

    add_figure_title(
        fig,
        title=f"{problem_type.upper()} - Seed Variance Analysis",
        subtitle="Each subplot: 5 thin seed lines + 1 thick mean line | dashed = convergence threshold",
    )

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Level 4: Final Performance ──────────────────────────────────────────

def plot_final_performance(
    aggregated: dict,
    problem_type: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 4: 2 subplots (stacked) - train vs test comparison.

    Subplot 1: last train loss, min train loss, test loss (bar + error bars) per config
    Subplot 2: last train acc, max train acc, test acc (bar + error bars) per config
    """
    _setup_style()

    configs = [
        f"{opt}_{sched}"
        for opt in OPTIMIZERS_LIST
        for sched in SCHEDULERS_LIST
        if f"{opt}_{sched}" in aggregated
]
    if not configs:
        logger.warning("No data for final performance plot")
        return plt.gcf()

    # ── Loss data ──
    last_train_loss_mean, last_train_loss_std = [], []
    min_train_loss_mean,  min_train_loss_std  = [], []
    test_loss_mean,       test_loss_std       = [], []

    # ── Accuracy data ──
    last_train_acc_mean, last_train_acc_std = [], []
    max_train_acc_mean,  max_train_acc_std  = [], []
    test_acc_mean,       test_acc_std       = [], []

    for c in configs:
        agg = aggregated[c]

        # train_losses_mean is array of shape (epochs,)
        tl = agg.get("train_losses_mean")
        tl_std = agg.get("train_losses_std")
        last_train_loss_mean.append(float(tl[-1])         if tl is not None else 0.0)
        last_train_loss_std.append(float(tl_std[-1])      if tl_std is not None else 0.0)
        min_train_loss_mean.append(float(tl.min())        if tl is not None else 0.0)
        min_idx = int(np.argmin(tl))                      if tl is not None else 0
        min_train_loss_std.append(float(tl_std[min_idx])  if tl_std is not None else 0.0)

        test_loss_mean.append(agg.get("test_loss_mean", 0.0))
        test_loss_std.append(agg.get("test_loss_std",   0.0))

        ta = agg.get("train_accuracies_mean")
        ta_std = agg.get("train_accuracies_std")
        last_train_acc_mean.append(float(ta[-1])          if ta is not None else 0.0)
        last_train_acc_std.append(float(ta_std[-1])       if ta_std is not None else 0.0)
        max_train_acc_mean.append(float(ta.max())         if ta is not None else 0.0)
        max_idx = int(np.argmax(ta))                      if ta is not None else 0
        max_train_acc_std.append(float(ta_std[max_idx])   if ta_std is not None else 0.0)

        test_acc_mean.append(agg.get("test_accuracy_mean", 0.0))
        test_acc_std.append(agg.get("test_accuracy_std",   0.0))

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    x = np.arange(len(configs))
    width = 0.25
    err_kw = dict(capsize=5, capthick=1.5, elinewidth=1.5)

    # ── Subplot 1: Loss ──
    ax = axes[0]
    b1 = ax.bar(x - width, last_train_loss_mean, width, label="Train Loss (last)",
                color="#4C72B0", alpha=0.85)
    ax.errorbar(x - width, last_train_loss_mean, yerr=last_train_loss_std,
                fmt="none", color="#1a3a6b", **err_kw)

    b2 = ax.bar(x,         min_train_loss_mean,  width, label="Train Loss (min)",
                color="#55A868", alpha=0.85)
    ax.errorbar(x,         min_train_loss_mean,  yerr=min_train_loss_std,
                fmt="none", color="#1f5c30", **err_kw)

    b3 = ax.bar(x + width, test_loss_mean,        width, label="Test Loss",
                color="#C44E52", alpha=0.85)
    ax.errorbar(x + width, test_loss_mean,         yerr=test_loss_std,
                fmt="none", color="#7a1a1d", **err_kw)

    ax.set_ylabel("Loss")
    ax.set_title(f"{problem_type.upper()} - Train vs Test Loss")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Subplot 2: Accuracy ──
    ax = axes[1]
    ax.bar(x - width, last_train_acc_mean, width, label="Train Acc (last)",
           color="#4C72B0", alpha=0.85)
    ax.errorbar(x - width, last_train_acc_mean, yerr=last_train_acc_std,
                fmt="none", color="#1a3a6b", **err_kw)

    ax.bar(x,         max_train_acc_mean,  width, label="Train Acc (max)",
           color="#55A868", alpha=0.85)
    ax.errorbar(x,         max_train_acc_mean,  yerr=max_train_acc_std,
                fmt="none", color="#1f5c30", **err_kw)

    ax.bar(x + width, test_acc_mean,        width, label="Test Acc",
           color="#C44E52", alpha=0.85)
    ax.errorbar(x + width, test_acc_mean,         yerr=test_acc_std,
                fmt="none", color="#7a1a1d", **err_kw)

    ax.set_ylabel("Accuracy")
    ax.set_title(f"{problem_type.upper()} - Train vs Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    add_figure_title(
        fig,
        title=f"{problem_type.upper()} - Final Performance Comparison",
        subtitle="Loss (top): last/min train + test | Accuracy (bottom): last/max train + test | Error bars = ±std across seeds"
    )

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Analytical Plot A: Loss + LR Dual Axis ─────────────────────

def plot_loss_lr_dual(
    results: dict,
    aggregated: dict,
    problem_type: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Analytical A: (2×5) grid - dual y-axis per subplot.

    Each subplot (optimizer, scheduler):
    - Left y-axis: train_loss mean±std (scheduler color, solid + std band)
    - Right y-axis: learning_rate mean (fixed orange, dashed line, log scale)
    """
    _setup_style()

    fig, axes = plt.subplots(2, 5, figsize=(28, 10), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes.ravel()

    for idx, (opt, sched) in enumerate([(o, s) for o in OPTIMIZERS_LIST for s in SCHEDULERS_LIST]):
        ax = axes[idx]
        key = f"{opt}_{sched}"

        # Left y-axis: train_loss
        mean_arr, std_arr = _get_aggregated_arrays(aggregated, key)
        if mean_arr is not None:
            color = _get_scheduler_color(sched)
            style = _get_optimizer_style(opt)
            epochs = np.arange(1, len(mean_arr) + 1)

            ax.plot(epochs, mean_arr, color=color,
                   linestyle=style["linestyle"], linewidth=style["linewidth"],
                   label="Loss")
            ax.fill_between(epochs, mean_arr - std_arr, mean_arr + std_arr,
                           color=color, alpha=style["alpha_std"])
            ax.set_ylabel("Training Loss", color=color)
            ax.tick_params(axis="y", labelcolor=color)

        # Right y-axis: learning_rate
        data = aggregated.get(key, {})
        lr_mean = data.get("learning_rates_mean")
        lr_color = "#9467bd"
        if lr_mean is not None:
            ax2 = ax.twinx()
            ax2.plot(np.arange(1, len(lr_mean) + 1), lr_mean,
                    color=lr_color, linestyle=":", linewidth=2.0, label="LR")
            ax2.set_ylabel("Learning Rate", color=lr_color)
            ax2.tick_params(axis="y", labelcolor=lr_color)
            ax2.set_yscale("log")

        ax.set_title(f"{opt.upper()} - {sched}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    # Add overall figure title with subtitle
    add_figure_title(
        fig,
        title=f"{problem_type.upper()} - Loss & Learning Rate Dual Axis",
        subtitle="Each subplot: Loss (left axis, colored) + LR (right axis, orange dashed)"
    )
    
    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Analytical Plot B: Heatmap ───────────────────────────────────

# Unica configurazione da mantenere
_LOWER_IS_BETTER = {"EtT", "AUL", "AUL_norm", "RV", "CV_final", "SI", "test_loss"}
_HAS_CRITICAL_POINT = {"rho_hat": 1.0, "R2": 0.5}  # metrica → centro naturale

def _get_heatmap_style(metric: str, matrix: np.ndarray) -> dict:
    """Returns kwargs for sns.heatmap based on metric semantics."""
    clean = metric.removesuffix("_mean").removesuffix("_std").removesuffix("_median")

    if clean in _HAS_CRITICAL_POINT:
        return {"cmap": "RdBu_r", "center": _HAS_CRITICAL_POINT[clean]}

    if clean in _LOWER_IS_BETTER:
        return {"cmap": "YlOrRd_r", "center": float(np.nanmedian(matrix))}

    return {"cmap": "RdYlGn", "center": float(np.nanmedian(matrix))}  # default: higher is better

def _get_direction_label(metric: str) -> str:
    clean = metric.removesuffix("_mean").removesuffix("_std").removesuffix("_median")
    if clean in _LOWER_IS_BETTER:
        return "↓ lower is better"
    if clean in _HAS_CRITICAL_POINT:
        return f"critical point = {_HAS_CRITICAL_POINT[clean]}"
    return "↑ higher is better"

def _robust_colorscale(matrix: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> tuple[float, float]:
    vmin = float(np.nanpercentile(matrix, lower))
    vmax = float(np.nanpercentile(matrix, upper))
    if np.isclose(vmin, vmax):  # tutti i valori uguali → fallback al range reale
        vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    return vmin, vmax


def _outlier_mask_iqr(matrix: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1 = np.nanpercentile(matrix, 25)
    q3 = np.nanpercentile(matrix, 75)
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(matrix, dtype=bool)
    return (matrix < q1 - k * iqr) | (matrix > q3 + k * iqr)

def plot_scheduler_optimizer_heatmap(
    aggregated: dict,
    problem_type: str,
    metrics: list[str] = ["EtT_mean", "AUL_norm_mean", "rho_hat_mean"],
    save_path: str | Path | None = None,
    show: bool = False,
) -> list[str]:
    """
    Analytical B: Heatmap of configurable metrics.

    Rows = schedulers, Columns = optimizers.
    Cell value = median over 5 seeds of specified metric.

    Default metrics (3 most important for project scope):
    1. EtT_mean - convergence speed
    2. AUL_norm_mean - area under loss
    3. rho_hat_mean - empirical convergence rate
    """
    
    saved_files = []

    # Build matrix for each metric
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Build matrix: rows=schedulers, cols=optimizers
        matrix = np.zeros((len(SCHEDULERS_LIST), len(OPTIMIZERS_LIST)))
        for i, sched in enumerate(SCHEDULERS_LIST):
            for j, opt in enumerate(OPTIMIZERS_LIST):
                key = f"{opt}_{sched}"
                data = aggregated.get(key, {})
                matrix[i, j] = data.get(metric, np.nan)

        style     = _get_heatmap_style(metric, matrix)
        direction = _get_direction_label(metric)
        vmin, vmax = _robust_colorscale(matrix)
        outlier_mask = _outlier_mask_iqr(matrix)

        annot_matrix = np.where(
            outlier_mask,
            np.vectorize(lambda v: f"{v:.4f}*")(matrix),
            np.vectorize(lambda v: f"{v:.4f}")(matrix)
        )

        # Create heatmap
        sns.heatmap(matrix, vmin=vmin, vmax=vmax, annot=annot_matrix, fmt="",
                    xticklabels=OPTIMIZERS_LIST, yticklabels=SCHEDULERS_LIST,
                    ax=ax, **style)
        
        
        ax.set_title(f"{problem_type.upper()} — {metric}  ({direction})")
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Scheduler")

        # Add title and subtitle
        add_figure_title(
            fig,
            title=f"{problem_type.upper()} - {metric.removesuffix('_mean').replace('_', ' ').title()}",
            subtitle=f"Rows: Schedulers | Columns: Optimizers",
        )
        
        if save_path:
            metric_suffix = metric.replace("_mean", "").lower()
            heatmap_path = str(save_path).replace(".png", f"_{metric_suffix}.png")
            _save(fig, heatmap_path)
            saved_files.append(heatmap_path)
        if show:
            plt.show()
        plt.close(fig)

    return saved_files


# ── Entry Point: plot_all ────────────────────────────────────────

def plot_all(
    results_by_problem: dict,
    aggregated: dict,
    problem_type: str,
    L_star_global: float | None = None,
    epsilon: float | None = None,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> list[str]:
    """
    Entry point: generate all 6 plots with standardized naming.

    Args:
        results_by_problem: results[problem_type] dict
        aggregated: pre-computed aggregated dict from aggregate_metrics()
        problem_type: "convex" or "non-convex"
        L_star: optional, for Level 1 horizontal line
        epsilon: optional, for Level 1 horizontal line
        save_dir: directory to save plots
        show: whether to display plots

    Returns:
        List of saved file paths.
    """
    if save_dir is None:
        save_dir = Path("reports/figures") / problem_type
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating all plots for: {problem_type}")
    
    saved_files = []
    
    # Level 1: Global Comparison
    plot_global_comparison(
        results_by_problem, aggregated, problem_type,
        save_path=save_dir / f"{problem_type}_level1_global_comparison.png",
        show=show
    )
    saved_files.append(str(save_dir / f"{problem_type}_level1_global_comparison.png"))
    
    # Level 2: By Optimizer (2x2 layout)
    plot_by_optimizer(
        results_by_problem, aggregated, problem_type,
        save_path=save_dir / f"{problem_type}_level2_by_optimizer.png",
        show=show
    )
    saved_files.append(str(save_dir / f"{problem_type}_level2_by_optimizer.png"))
    
    # Level 3: Seed Variance
    plot_seed_variance(
        results_by_problem, problem_type,
        L_star_global=L_star_global,
        save_path=save_dir / f"{problem_type}_level3_seed_variance.png",
        show=show
    )
    saved_files.append(str(save_dir / f"{problem_type}_level3_seed_variance.png"))
    
    # Level 4: Final Performance
    plot_final_performance(
        aggregated, problem_type,
        save_path=save_dir / f"{problem_type}_level4_final_performance.png",
        show=show
    )
    saved_files.append(str(save_dir / f"{problem_type}_level4_final_performance.png"))
    
    # Analytical A: Loss + LR Dual
    plot_loss_lr_dual(
        results_by_problem, aggregated, problem_type,
        save_path=save_dir / f"{problem_type}_analytical_a_loss_lr_dual.png",
        show=show
    )
    saved_files.append(str(save_dir / f"{problem_type}_analytical_a_loss_lr_dual.png"))
    
    # Analytical B: Heatmap (returns list of saved files)
    heatmap_files = plot_scheduler_optimizer_heatmap(
        aggregated, problem_type,
        metrics=["EtT_mean", "EtT_median", "AUL_norm_mean", "rho_hat_mean", "CV_final_mean", "SI_mean", "test_accuracy_mean", "test_loss_mean"],
        save_path=save_dir / f"{problem_type}_analytical_b_heatmap.png",
        show=show
    )
    
    saved_files.append(heatmap_files)
    
    logger.info(f"All plots saved for: {problem_type}")
    return saved_files
