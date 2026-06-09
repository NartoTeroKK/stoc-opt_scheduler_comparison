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
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from stoc_opt_scheduler_comparison.utils import viz_logger as logger
from stoc_opt_scheduler_comparison.evaluation.convergence import (
    compute_target_loss,
)

# ── Visual Encoding ─────────────────────────────────────────────────────────

SCHEDULER_COLORS: dict[str, str] = {
    "none": "#7f7f7f",  # Gray - neutral baseline
    "exponential": "#1f77b4",  # Blue - monotonic decay
    "cosine": "#2ca02c",  # Green - smooth cyclic
    "cyclic": "#ff7f0e",  # Orange - oscillating
    "one-cycle": "#d62728",  # Red - aggressive warm-up
}

OPTIMIZER_STYLES: dict[str, dict[str, Any]] = {
    "sgd": {"linestyle": "-", "linewidth": 2.0, "alpha_std": 0.05},
    "adam": {"linestyle": "--", "linewidth": 2.0, "alpha_std": 0.05},
}

SEED_STYLE: dict[str, dict[str, float]] = {
    "individual": {"linewidth": 1, "alpha": 0.5},
    "mean": {"linewidth": 2.5, "alpha": 1.0},
}

SCHEDULERS_LIST = ["none", "exponential", "cosine", "cyclic", "one-cycle"]
OPTIMIZERS_LIST = ["sgd", "adam"]

# E_target threshold line style (level 1: 5 %)
E_TARGET_COLOR: str = "#bf75f0"
E_TARGET_LINESTYLE: str = "-."
E_TARGET_LABEL: str = "L_target"

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
    return OPTIMIZER_STYLES.get(
        optimizer, {"linestyle": "-", "linewidth": 2.0, "alpha_std": 0.15}
    )


def _setup_style() -> None:
    """Apply clean academic style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: Figure, path: str | Path) -> None:
    """Save figure and log."""
    fig.savefig(str(path), dpi=500, bbox_inches="tight")
    logger.info(f"Saved: {path}")


def _get_aggregated_arrays(
    aggregated: dict, key: str
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
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


# ── Level 1: Global Comparison ─────────────────────────────────────────


def plot_global_comparison(
    results: dict,
    aggregated: dict,
    problem_type: str,
    L_star: float | None = None,
    L0: float | None = None,
    e_target_epsilon: float | None = 0.05,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 1: (3, 1) subplots — loss curve overview + LR schedules.

    Top subplot:    train_loss mean per config (linear scale).
    Middle subplot: train_loss mean per config (log scale) + L_target threshold line.
    Bottom subplot: learning_rate mean per config (log scale, no std band).
    """
    _setup_style()

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Pre-calcolo del valore di soglia E_target
    l_target: float | None = None
    if L_star is not None and L0 is not None and e_target_epsilon is not None:
        l_target = compute_target_loss(L_star, L0, epsilon=e_target_epsilon)

    # Raccolta delle configurazioni univoche (scheduler, optimizer) dai nomi delle run
    configs = []
    for run_name in results:
        parsed = _parse_run_name(run_name)
        if parsed is None:
            continue
        _, sched, opt, _ = parsed
        if (sched, opt) not in configs:
            configs.append((sched, opt))

    # ── Subplot 1 (Top): train_loss mean curves (Linear Scale) ──
    ax0 = axes[0]
    for sched, opt in configs:
        key = f"{opt}_{sched}"
        mean_arr, std_arr = _get_aggregated_arrays(aggregated, key)
        if mean_arr is None:
            continue
        color = _get_scheduler_color(sched)
        style = _get_optimizer_style(opt)
        epochs = np.arange(1, len(mean_arr) + 1)

        ax0.plot(
            epochs,
            mean_arr,
            color=color,
            label=f"{sched} ({opt})",
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    # Single E_target threshold line (level 1)
    if l_target is not None:
        ax0.axhline(
            y=l_target,
            color=E_TARGET_COLOR,
            linestyle=E_TARGET_LINESTYLE,
            linewidth=1,
            alpha=0.75,
            label=E_TARGET_LABEL,
        )

    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Training Loss")
    ax0.set_title(f"{problem_type.upper()} - Loss Curves (Linear Scale)")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    # ── Subplot 2 (Middle): train_loss mean curves (Log Scale) + E_target line ──
    ax1 = axes[1]
    for sched, opt in configs:
        key = f"{opt}_{sched}"
        mean_arr, std_arr = _get_aggregated_arrays(aggregated, key)
        if mean_arr is None:
            continue
        color = _get_scheduler_color(sched)
        style = _get_optimizer_style(opt)
        epochs = np.arange(1, len(mean_arr) + 1)

        ax1.plot(
            epochs,
            mean_arr,
            color=color,
            label=f"{sched} ({opt})",
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    # Single E_target threshold line (level 1)
    if l_target is not None:
        ax1.axhline(
            y=l_target,
            color=E_TARGET_COLOR,
            linestyle=E_TARGET_LINESTYLE,
            linewidth=1,
            alpha=0.75,
            label=E_TARGET_LABEL,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (Log Scale)")
    ax1.set_title(f"{problem_type.upper()} - Loss Curves (Log Scale)")
    ax1.set_yscale("log")

    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Subplot 3 (Bottom): learning_rate mean curves (log scale, no std band) ──
    ax2 = axes[2]
    for sched, opt in configs:
        key = f"{opt}_{sched}"
        data = aggregated.get(key, {})
        lr_mean = data.get("learning_rates_mean")
        if lr_mean is None:
            continue
        color = _get_scheduler_color(sched)
        style = _get_optimizer_style(opt)
        epochs = np.arange(1, len(lr_mean) + 1)

        ax2.plot(
            epochs,
            lr_mean,
            color=color,
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=f"{sched} ({opt})",
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title(f"{problem_type.upper()} - LR Schedules")
    ax2.set_yscale("log")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # ── Enforce strict x-axis from 0 to max epochs ──
    max_epochs = 0
    for sched, opt in configs:
        key = f"{opt}_{sched}"
        mean_arr, _ = _get_aggregated_arrays(aggregated, key)
        if mean_arr is not None:
            max_epochs = max(max_epochs, len(mean_arr))
        lr_mean = aggregated.get(key, {}).get("learning_rates_mean")
        if lr_mean is not None:
            max_epochs = max(max_epochs, len(lr_mean))
    for ax in axes:
        ax.set_xlim(0, max_epochs)
        ax.xaxis.set_major_locator(MultipleLocator(10))

    # ── Layout and Title ──
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    # add_figure_title(
    #     fig,
    #     title=f"{problem_type.upper()} - Global Comparison",
    #     subtitle=f"Top: Loss (Linear) | Middle: Loss (Log) | Bottom: LR Schedules (Log)",
    # )

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
    L_star: float | None = None,
    L0: float | None = None,
    e_target_epsilon: float | None = 0.05,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 2: 2×2 grid - rows=optimizers (SGD, Adam), cols=linear/log scale.

    Each subplot shows 5 scheduler curves with mean±std from aggregated arrays.
    Left column: linear scale.
    Right column: log scale for loss.

    If L_star and L0 are provided, a horizontal threshold line is drawn at
    the computed L_target.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Pre-compute E_target threshold value
    l_target: float | None = None
    if L_star is not None and L0 is not None and e_target_epsilon is not None:
        l_target = compute_target_loss(L_star, L0, epsilon=e_target_epsilon)

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

            ax_linear.plot(
                epochs,
                mean_arr,
                color=color,
                label=sched,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
            ax_linear.fill_between(
                epochs,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=color,
                alpha=style["alpha_std"],
            )

        # E_target threshold line on linear subplot
        if l_target is not None:
            ax_linear.axhline(
                y=l_target,
                color=E_TARGET_COLOR,
                linestyle=E_TARGET_LINESTYLE,
                linewidth=1.0,
                alpha=0.7,
                label=E_TARGET_LABEL if idx == 0 else "",
            )
        if idx == 0 and l_target is not None:
            ax_linear.legend(loc="upper right")

        ax_linear.set_xlabel("Epoch")
        ax_linear.set_ylabel("Training Loss")
        ax_linear.set_title(f"{problem_type.upper()} - {opt.upper()} (Linear Scale)")
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

            ax_log.plot(
                epochs,
                mean_arr,
                color=color,
                label=sched,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
            ax_log.fill_between(
                epochs,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=color,
                alpha=style["alpha_std"],
            )

        # E_target threshold line on log subplot
        if l_target is not None:
            ax_log.axhline(
                y=l_target,
                color=E_TARGET_COLOR,
                linestyle=E_TARGET_LINESTYLE,
                linewidth=1.0,
                alpha=0.7,
            )

        ax_log.set_xlabel("Epoch")
        ax_log.set_ylabel("Training Loss (log scale)")
        ax_log.set_title(f"{problem_type.upper()} - {opt.upper()} (Log Scale)")
        ax_log.set_yscale("log")
        ax_log.grid(True, alpha=0.3)

    # ── Enforce strict x-axis from 0 to max epochs ──
    max_epochs = 0
    for opt in OPTIMIZERS_LIST:
        for sched in SCHEDULERS_LIST:
            key = f"{opt}_{sched}"
            mean_arr, _ = _get_aggregated_arrays(aggregated, key)
            if mean_arr is not None:
                max_epochs = max(max_epochs, len(mean_arr))
    for ax in axes:
        ax.set_xlim(0, max_epochs)
        ax.xaxis.set_major_locator(MultipleLocator(10))

    # add_figure_title(
    #     fig,
    #     title="", #f"{problem_type.upper()} - Loss Curves by Optimizer",
    #     subtitle=f"Left: Linear scale | Right: Log scale | Dashed lines = E_target thresholds",
    # )

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
    L0: float | None = None,
    e_target_epsilon: float | None = 0.05,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 3: (5×2) grid - diagnostic seed variance.

    Each cell (scheduler, optimizer): 5 thin seed lines + 1 thick mean line.
    If L0 is provided, a global E_target threshold line is drawn across all cells.
    No rolling mean - raw arrays only.
    """
    _setup_style()

    fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True, sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes.ravel()

    # Pre-compute global E_target threshold value
    l_target: float | None = None
    if L_star_global is not None and L0 is not None and e_target_epsilon is not None:
        l_target = compute_target_loss(L_star_global, L0, epsilon=e_target_epsilon)

    all_loss_values = []

    for idx, (opt, sched) in enumerate(
        [(o, s) for s in SCHEDULERS_LIST for o in OPTIMIZERS_LIST]
    ):
        ax = axes[idx]
        color = _get_scheduler_color(sched)
        # style = _get_optimizer_style(opt)

        seed_arrays = _get_seed_arrays(results, sched, opt)

        if seed_arrays:
            for arr in seed_arrays:
                ax.plot(
                    np.arange(1, len(arr) + 1),
                    arr,
                    color=color,
                    linewidth=SEED_STYLE["individual"]["linewidth"],
                    alpha=SEED_STYLE["individual"]["alpha"],
                )
                all_loss_values.extend(arr.tolist())

            stacked = np.stack(seed_arrays)
            mean_arr = stacked.mean(axis=0)
            ax.plot(
                np.arange(1, len(mean_arr) + 1),
                mean_arr,
                color=color,
                linewidth=SEED_STYLE["mean"]["linewidth"],
                alpha=SEED_STYLE["mean"]["alpha"],
            )

            # --- Global E_target threshold line ---
            if l_target is not None:
                ax.axhline(
                    y=l_target,
                    color=E_TARGET_COLOR,
                    linestyle=E_TARGET_LINESTYLE,
                    linewidth=0.8,
                    alpha=0.6,
                )

        ax.set_title(f"{opt} - {sched}", fontsize=16)
        ax.grid(True, alpha=0.3)
        if idx % 2 == 0:
            ax.set_ylabel("Training Loss")

    # y_max = np.percentile(all_loss_values, 95) * 1.05
    # y_min = max(0.0, np.percentile(all_loss_values, 1) * 0.95)

    for ax in axes:
        #     ax.set_ylim(y_min, y_max)
        ax.set_yscale("log")

    # ── Enforce strict x-axis from 0 to max epochs ──
    max_epochs = 0
    for opt, sched in [(o, s) for s in SCHEDULERS_LIST for o in OPTIMIZERS_LIST]:
        seed_arrays = _get_seed_arrays(results, sched, opt)
        if seed_arrays:
            max_epochs = max(max_epochs, max(len(arr) for arr in seed_arrays))
    for ax in axes:
        ax.set_xlim(0, max_epochs)
        ax.xaxis.set_major_locator(MultipleLocator(10))

    # Build subtitle with E_target info
    # subtitle = "LOG SCALE subplot: 10 thin seed lines + 1 thick mean line | dashed = L_target"

    # add_figure_title(
    #     fig,
    #     title= "", #f"{problem_type.upper()} - Seed Variance Analysis",
    #     subtitle=subtitle,
    # )

    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Convergence Boxplot (Epochs-to-Target) ─────────────────────────────


def plot_convergence_boxplot(
    data: dict | pd.DataFrame,
    problem_type: str,
    L_star: float | None = None,
    L0: float | None = None,
    e_target_epsilon: float | None = 0.05,
    max_epochs: int = 100,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Boxplot raggruppato (Hue=Optimizer, X=Scheduler) che accetta sia Dict che DataFrame.
    Calcola la statistica del boxplot SOLO sulle run convergenti.
    I Timeout sono tracciati manualmente con offset precisi per mantenere
    il layout intatto anche quando intere categorie falliscono.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    opt_palette = {"ADAM": "#2b83ba", "SGD": "#d7191c"}

    # =========================================================================
    # 0. ADATTATORE DI DATI: Dict -> DataFrame piatto
    # =========================================================================
    if isinstance(data, dict):
        if "Epochs_to_Target" in data or "Scheduler" in data:
            # Caso A: È un dizionario già formattato per colonne
            df_convergence = pd.DataFrame(data)
        else:
            # Caso B: È il dizionario innestato (es. convergence_results["convex"]["aggregated"])
            rows = []
            for key, val in data.items():
                opt = str(val.get("optimizer", "unknown")).upper()
                sched = str(val.get("scheduler", "unknown"))

                # Cerca l'array dei valori raw per tracciare tutti i seed nel boxplot.
                # Assumiamo che i valori grezzi si chiamino "EtT_values", "EtT_runs" o simili.
                raw_ett = val.get(
                    "EtT_values", val.get("EtT_runs", val.get("Epochs_to_Target", []))
                )

                if (
                    isinstance(raw_ett, (list, np.ndarray, pd.Series))
                    and len(raw_ett) > 0
                ):
                    for ett in raw_ett:
                        rows.append(
                            {
                                "Optimizer": opt,
                                "Scheduler": sched,
                                "Epochs_to_Target": ett,
                            }
                        )
                else:
                    # Fallback di sicurezza: se per caso gli passi i dati aggregati SENZA
                    # gli array raw, usa la media. (N.B. Il boxplot diventerà un punto singolo)
                    fallback_val = val.get("EtT_mean", val.get("EtT", np.nan))
                    rows.append(
                        {
                            "Optimizer": opt,
                            "Scheduler": sched,
                            "Epochs_to_Target": fallback_val,
                        }
                    )

            df_convergence = pd.DataFrame(rows)
    else:
        df_convergence = data.copy()

    # Normalizza i nomi delle colonne per sicurezza (nel caso in cui passi un df con nomi minuscoli)
    df_convergence = df_convergence.rename(
        columns={
            "optimizer": "Optimizer",
            "scheduler": "Scheduler",
            "EtT": "Epochs_to_Target",
            "EtT_values": "Epochs_to_Target",
        }
    )
    df_convergence["Optimizer"] = df_convergence["Optimizer"].astype(str).str.upper()

    # =========================================================================
    # 1. Definizione Categorie (Garantisce ordine asse X)
    # =========================================================================
    sched_order = df_convergence["Scheduler"].unique()
    opt_order = ["ADAM", "SGD"]

    df_convergence["Scheduler"] = pd.Categorical(
        df_convergence["Scheduler"], categories=sched_order, ordered=True
    )
    df_convergence["Optimizer"] = pd.Categorical(
        df_convergence["Optimizer"], categories=opt_order, ordered=True
    )

    # =========================================================================
    # 2. Separazione Dati IBRIDA
    # =========================================================================
    is_timeout = (df_convergence["Epochs_to_Target"] >= max_epochs) | (
        df_convergence["Epochs_to_Target"].isna()
    )

    df_converged = df_convergence.copy()
    df_converged.loc[is_timeout, "Epochs_to_Target"] = np.nan

    df_timeout = df_convergence[is_timeout].copy()

    # Pre-calcolo del valore di soglia E_target
    L_target: float | None = None
    if L_star is not None and L0 is not None and e_target_epsilon is not None:
        L_target = compute_target_loss(L_star, L0, epsilon=e_target_epsilon)

    # =========================================================================
    # 3. PLOT 1: Boxplot e Stripplot dei Convergenti
    # =========================================================================
    sns.boxplot(
        data=df_converged,
        x="Scheduler",
        y="Epochs_to_Target",
        hue="Optimizer",
        hue_order=opt_order,
        palette=opt_palette,
        width=0.6,
        linewidth=1.5,
        whis=(0, 100),
        showfliers=False,
        ax=ax,
    )

    sns.stripplot(
        data=df_converged,
        x="Scheduler",
        y="Epochs_to_Target",
        hue="Optimizer",
        hue_order=opt_order,
        dodge=True,
        palette={opt: "black" for opt in opt_order},
        alpha=0.6,
        size=5,
        jitter=0.15,
        ax=ax,
        legend=False,
    )

    # =========================================================================
    # 4. PLOT 2: PLOT MANUALE DEI TIMEOUT
    # =========================================================================
    if not df_timeout.empty:
        x_map = {sched: i for i, sched in enumerate(sched_order)}
        offset_map = {"ADAM": -0.15, "SGD": 0.15}

        for _, row in df_timeout.iterrows():
            sched = row["Scheduler"]
            opt = row["Optimizer"]

            base_x = x_map[sched]
            offset = offset_map[opt]

            jitter = np.random.uniform(-0.05, 0.05)
            final_x = base_x + offset + jitter

            ax.plot(
                final_x,
                max_epochs,
                marker="X",
                color=opt_palette[opt],
                markersize=7,
                alpha=0.9,
                linestyle="None",
            )

    # =========================================================================
    # 5. Legenda e Layout
    # =========================================================================
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = handles[:2]
    unique_labels = labels[:2]

    if not df_timeout.empty:
        timeout_handle = Line2D(
            [0], [0], marker="X", color="w", markerfacecolor="gray", markersize=8
        )
        unique_handles.append(timeout_handle)
        unique_labels.append("Timeout")

    ax.legend(
        unique_handles,
        unique_labels,
        title="Legenda",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
    )

    # ax.set_title(
    #     f"{problem_type.upper()} - Convergence Speed (Epochs-to-Target)",
    #     fontsize=16, fontweight='bold', pad=50,
    # )
    ax.set_xlabel("Learning Rate Scheduler", fontsize=14, labelpad=15)
    ax.set_ylabel(f"Epochs to Target ({L_target:.4f})", fontsize=14, labelpad=15)
    ax.tick_params(axis="x", rotation=0, labelsize=12)

    ax.set_ylim(0, max_epochs + 1)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.axhline(
        y=max_epochs, color="gray", linestyle="--", alpha=0.5, linewidth=1.5, zorder=0
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()

    plt.close(fig)
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
    Codice ottimizzato con Dot Plot (Point Plot) e error bars.
    L'asse Y si adatta dinamicamente alla varianza senza distorcere i dati.
    """
    # _setup_style() # Decommenta la tua funzione di stile

    configs = [
        f"{opt}_{sched}"
        for opt in ["sgd", "adam"]  # Assicurati di usare OPTIMIZERS_LIST
        for sched in [
            "none",
            "exponential",
            "cosine",
            "cyclic",
            "one-cycle",
        ]  # SCHEDULERS_LIST
        if f"{opt}_{sched}" in aggregated
    ]

    if not configs:
        logger.warning("No data for final performance plot")
        return plt.gcf()

    # 1. Estrazione dati vettorializzata
    metrics = {
        "loss": {
            "last": [],
            "min_max": [],
            "test": [],
            "last_std": [],
            "min_max_std": [],
            "test_std": [],
        },
        "acc": {
            "last": [],
            "min_max": [],
            "test": [],
            "last_std": [],
            "min_max_std": [],
            "test_std": [],
        },
    }

    for c in configs:
        agg = aggregated[c]

        tl = agg.get("train_losses_mean", np.array([0.0]))
        tl_std = agg.get("train_losses_std", np.array([0.0]))
        ta = agg.get("train_accuracies_mean", np.array([0.0]))
        ta_std = agg.get("train_accuracies_std", np.array([0.0]))

        # Popolamento Loss
        metrics["loss"]["last"].append(tl[-1])
        metrics["loss"]["last_std"].append(tl_std[-1])
        min_idx = int(np.argmin(tl))
        metrics["loss"]["min_max"].append(tl[min_idx])
        metrics["loss"]["min_max_std"].append(tl_std[min_idx])
        metrics["loss"]["test"].append(agg.get("test_loss_mean", 0.0))
        metrics["loss"]["test_std"].append(agg.get("test_loss_std", 0.0))

        # Popolamento Accuracy
        metrics["acc"]["last"].append(ta[-1])
        metrics["acc"]["last_std"].append(ta_std[-1])
        max_idx = int(np.argmax(ta))
        metrics["acc"]["min_max"].append(ta[max_idx])
        metrics["acc"]["min_max_std"].append(ta_std[max_idx])
        metrics["acc"]["test"].append(agg.get("test_accuracy_mean", 0.0))
        metrics["acc"]["test_std"].append(agg.get("test_accuracy_std", 0.0))

    for category in metrics:
        for key in metrics[category]:
            metrics[category][key] = np.array(metrics[category][key])  # type: ignore

    # Altezza leggermente ridotta rispetto al figsize enorme dell'inset
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    x = np.arange(len(configs))
    # Ridotto l'offset per raggruppare i 3 punti più stretti attorno all'etichetta
    offset_w = 0.15

    # Stile globale degli error bar
    err_kw = dict(capsize=4, capthick=1.5, elinewidth=1, ls="none")

    plot_structure = [
        {
            "ax_idx": 0,
            "title": "Train vs Test Loss",
            "ylabel": "Loss",
            "data": metrics["loss"],
            "labels": ["Train Loss (last)", "Train Loss (min)", "Test Loss"],
        },
        {
            "ax_idx": 1,
            "title": "Train vs Test Accuracy",
            "ylabel": "Accuracy",
            "data": metrics["acc"],
            "labels": [
                "Train Accuracy (last)",
                "Train Accuracy (max)",
                "Test Accuracy",
            ],
        },
    ]

    # Stile per i punti: usiamo marker diversi per differenziare a colpo d'occhio
    # Cerchio per Last, Triangolo su per Min/Max, Quadrato per Test
    styles = [
        {
            "color": "#4C72B0",
            "err_color": "#1a3a6b",
            "marker": "o",
            "offset": -offset_w,
        },  # Last (Blu)
        {
            "color": "#55A868",
            "err_color": "#1f5c30",
            "marker": "^",
            "offset": 0,
        },  # Min/Max (Verde)
        {
            "color": "#C44E52",
            "err_color": "#7a1a1d",
            "marker": "s",
            "offset": offset_w,
        },  # Test (Rosso)
    ]

    for struct in plot_structure:
        ax = axes[struct["ax_idx"]]
        d = struct["data"]
        categories = ["last", "min_max", "test"]

        # Plot dei Dot con Error Bars
        for i, cat in enumerate(categories):
            means = d[cat]
            stds = d[f"{cat}_std"]
            s = styles[i]

            # Disegniamo punto e barra usando 'ecolor' per assegnare il colore alla barra di errore
            ax.errorbar(
                x + s["offset"],
                means,
                yerr=stds,
                fmt=s["marker"],
                color=s["color"],
                ecolor=s["err_color"],
                label=struct["labels"][i],
                markersize=9,
                alpha=0.9,
                **err_kw,
            )

        # 2. CALCOLO DINAMICO ASSE Y (Ora perfettamente legittimo per il Dot Plot)
        all_mins = np.concatenate([d[cat] - d[f"{cat}_std"] for cat in categories])
        all_maxs = np.concatenate([d[cat] + d[f"{cat}_std"] for cat in categories])

        global_min = np.nanmin(all_mins)
        global_max = np.nanmax(all_maxs)
        value_span = global_max - global_min

        # Margine del 5% per non toccare i bordi superiore e inferiore
        margin = max(
            value_span * 0.05, 0.01
        )  # Minimo 1% di margine se la varianza è zero

        if struct["ylabel"] == "Accuracy":
            # Per l'accuratezza tagliamo liberamente sotto, con tetto rigido a 1.01
            y_bottom = max(0.0, global_min - margin)
            y_top = min(
                1.01, global_max + margin
            )  # Permettiamo di sforare leggermente 1.0 se l'errore lo tocca
            ax.set_ylim(y_bottom, y_top)

            # Evidenziamo visivamente la linea del 100%
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3, zorder=0)
        else:
            # Per la loss, il minimo matematico è 0.0
            y_bottom = max(0.0, global_min - margin)
            ax.set_ylim(y_bottom, global_max + margin)

            # Linea dello zero per riferimento visivo
            ax.axhline(0.0, color="gray", linestyle="--", alpha=0.3, zorder=0)

        # Pulizia e formattazione dell'asse X
        ax.set_ylabel(struct["ylabel"])
        ax.set_title(f"{problem_type.upper()} - {struct['title']}")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right")

        # Aggiungiamo gridlines verticali leggere per separare visivamente le configurazioni
        for tick in x[:-1]:
            ax.axvline(tick + 0.5, color="gray", alpha=0.15, linestyle="-")

        ax.legend(loc="best", frameon=True, edgecolor="white")
        ax.grid(True, alpha=0.3, axis="y")

    # add_figure_title(
    #    fig,
    #    title=f"{problem_type.upper()} - Final Performance Comparison",
    #    subtitle="Loss (top): last/min train + test | Accuracy (bottom): last/max train + test | Error bars = ±std across seeds"
    # )
    # Aggiungi il layout dinamico
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

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

    fig, axes = plt.subplots(5, 2, figsize=(12, 20), sharex=True, sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes.ravel()

    for idx, (opt, sched) in enumerate(
        [(o, s) for s in SCHEDULERS_LIST for o in OPTIMIZERS_LIST]
    ):
        ax = axes[idx]
        key = f"{opt}_{sched}"

        # Left y-axis: train_loss
        mean_arr, std_arr = _get_aggregated_arrays(aggregated, key)
        if mean_arr is not None:
            color = _get_scheduler_color(sched)
            style = _get_optimizer_style(opt)
            epochs = np.arange(1, len(mean_arr) + 1)

            ax.plot(
                epochs,
                mean_arr,
                color=color,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label="Loss",
            )
            ax.fill_between(
                epochs,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=color,
                alpha=style["alpha_std"],
            )
            ax.set_ylabel("Training Loss", color=color)
            ax.tick_params(axis="y", labelcolor=color)

        # Right y-axis: learning_rate
        data = aggregated.get(key, {})
        lr_mean = data.get("learning_rates_mean")
        lr_color = "#9467bd"
        if lr_mean is not None:
            ax2 = ax.twinx()
            ax2.plot(
                np.arange(1, len(lr_mean) + 1),
                lr_mean,
                color=lr_color,
                linestyle=":",
                linewidth=1.0,
                label="LR",
            )
            ax2.set_ylabel("Learning Rate", color=lr_color)
            ax2.tick_params(axis="y", labelcolor=lr_color)
            # ax2.set_yscale("log")
            ax2.grid(False)
            ax2.set_ylim(bottom=1e-6, top=None)

        ax.set_title(f"{opt.upper()} - {sched}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    # ── Enforce strict x-axis from 0 to max epochs ──
    max_epochs = 0
    for opt, sched in [(o, s) for s in SCHEDULERS_LIST for o in OPTIMIZERS_LIST]:
        key = f"{opt}_{sched}"
        mean_arr, _ = _get_aggregated_arrays(aggregated, key)
        if mean_arr is not None:
            max_epochs = max(max_epochs, len(mean_arr))
        lr_mean = aggregated.get(key, {}).get("learning_rates_mean")
        if lr_mean is not None:
            max_epochs = max(max_epochs, len(lr_mean))
    for ax in axes:
        ax.set_xlim(0, max_epochs)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_ylim(bottom=0, top=None)

    # Add overall figure title with subtitle
    # add_figure_title(
    #     fig,
    #     title=f"{problem_type.upper()} - Loss & Learning Rate Dual Axis",
    #     subtitle="Each subplot: Loss (left axis) + LR (right axis, dotted)"
    # )

    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Analytical Plot B: Heatmap ───────────────────────────────────

_LOWER_IS_BETTER = {"E_target", "suboptimality_gap", "AUL", "CV", "test_loss"}


def _get_heatmap_style(metric: str) -> dict:
    """Returns kwargs for sns.heatmap based on metric semantics."""

    # Palette Accademica Single-Tone: Colore più scuro = Risultato Migliore
    if metric in _LOWER_IS_BETTER:
        return {"cmap": "Blues_r"}  # Basso = scuro, Alto = chiaro

    return {"cmap": "Greens"}  # Alto = scuro, Basso = chiaro


def _get_direction_label(metric: str) -> str:
    if metric in _LOWER_IS_BETTER:
        return "↓ lower is better"
    return "↑ higher is better"


def _robust_colorscale(matrix: np.ndarray) -> tuple[float, float]:
    """
    Computes a robust visual range for the color map by ignoring extreme outliers.
    Saturates colors at the Q1 - 1.5 * IQR and Q3 + 1.5 * IQR boundaries, while
    clamping to the real data min/max to ensure a clean visual scale.
    """
    # 1. Handle all-NaN matrices to prevent errors
    if np.isnan(matrix).all():
        return 0.0, 1.0

    # 2. Compute Q1, Q3, and the Interquartile Range (IQR)
    q1 = np.nanpercentile(matrix, 25)
    q3 = np.nanpercentile(matrix, 75)
    iqr = q3 - q1

    # 3. Handle matrices with zero variation (e.g., all values same)
    if iqr == 0:
        # Fallback to full range if there is no core variation
        return float(np.nanmin(matrix)), float(np.nanmax(matrix))

    # 4. Set robust bounds based on the Q1 - 1.5 * IQR and Q3 + 1.5 * IQR rules
    # This defines the theoretical "neighborhood" of non-outlier data.
    # We will use this to set the color gradient.
    # The user-identified "extreme outlier" 5.1... is > Q3 + 1.5 * IQR,
    # so it will be clipped to the maximum color.
    vmin_robust = q1 - 1.5 * iqr
    vmax_robust = q3 + 1.5 * iqr

    # 5. Clamp to real data bounds to prevent visual clutter
    # If all values are positive (0.05... 5.1), vmin_robust shouldn't
    # go below the real min to -0.08. We clamp to nanmin. Same for max.
    real_min = np.nanmin(matrix)
    real_max = np.nanmax(matrix)
    vmin = float(max(vmin_robust, real_min))
    vmax = float(min(vmax_robust, real_max))

    # 6. Re-check to avoid close bounds which can crash Seaborn
    if np.isclose(vmin, vmax):
        vmin, vmax = float(real_min), float(real_max)

    return vmin, vmax


def _outlier_mask_iqr(matrix: np.ndarray, k: float = 1.5) -> np.ndarray:
    if np.isnan(matrix).all():
        return np.zeros_like(matrix, dtype=bool)

    q1 = np.nanpercentile(matrix, 25)
    q3 = np.nanpercentile(matrix, 75)
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(matrix, dtype=bool)
    return (matrix < q1 - k * iqr) | (matrix > q3 + k * iqr)


def plot_scheduler_optimizer_heatmap(
    data: dict | pd.DataFrame,
    problem_type: str,
    metrics: list[str] = ["E_target", "AUL", "CV"],
    save_path: str | Path | None = None,
    show: bool = False,
) -> list[str]:
    """
    Analytical B: Heatmap of configurable metrics directly from DataFrame or nested dictionary.

    Rows = schedulers, Columns = optimizers.
    """
    saved_files = []

    # 0. Conversione automatica del dizionario in DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient="index")
    else:
        df = data.copy()

    for metric in metrics:
        # Trova la colonna reale (gestisce la differenza tra "AUL" e "AUL_mean")
        metric_col = metric
        if metric not in df.columns:
            if f"{metric}_mean" in df.columns:
                metric_col = f"{metric}_mean"
            elif metric == "E_target" and "EtT_mean" in df.columns:  # fallback mapping
                metric_col = "EtT_mean"
            else:
                print(
                    f"Warning: Metric '{metric}' (or '{metric}_mean') not found in data. Skipping."
                )
                continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # 1. Creazione elegante della matrice 2D tramite pivot table
        pivot_df = df.pivot(index="scheduler", columns="optimizer", values=metric_col)
        matrix = pivot_df.to_numpy(dtype=float)
        y_labels = pivot_df.index.tolist()
        x_labels = pivot_df.columns.tolist()

        # 2. Configurazione stile e scale
        # NOTA: Assicurati che queste helper function siano importate o definite nel file
        style = _get_heatmap_style(metric)
        direction = _get_direction_label(metric)
        vmin, vmax = _robust_colorscale(matrix)
        outlier_mask = _outlier_mask_iqr(matrix)

        # 3. Costruzione della matrice di annotazione testuale (gestisce NaN format)
        def format_cell(val: float, is_outlier: bool) -> str:
            if pd.isna(val):
                return "NaN"  # Più esplicito di stringa vuota per indicare che non è esploso
            # Usa 0 decimali se il numero è intero, 4 se è un float
            num_str = f"{val:.0f}" if float(val).is_integer() else f"{val:.4f}"
            return f"{num_str}*" if is_outlier else num_str

        annot_matrix = np.vectorize(format_cell)(matrix, outlier_mask)

        # 4. Rendering Heatmap
        sns.heatmap(
            matrix,
            vmin=vmin,
            vmax=vmax,
            annot=annot_matrix,
            fmt="",
            xticklabels=x_labels,
            yticklabels=y_labels,
            ax=ax,
            **style,
        )

        ax.set_title(f"{problem_type.upper()} — {metric} ({direction})")
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Scheduler")

        # 5. Salvataggio
        if save_path:
            # Crea un percorso unico per ogni metrica basato sul save_path originale
            heatmap_path = str(save_path).replace(".png", f"_{metric.lower()}.png")
            plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
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
    L0: float | None = None,
    e_target_epsilon: float | None = 0.05,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> list[str]:
    """
    Entry point: generate all standard plots with standardized naming.

    Args:
        results_by_problem: results[problem_type] dict
        aggregated: pre-computed aggregated dict from aggregate_metrics()
        problem_type: "convex" or "non-convex"
        L_star_global: optional reference minimum loss for threshold lines.
        L0: optional global initial loss for E_target computation.
        e_target_epsilon: epsilon value for E_target (default 0.05 = level 1).
        save_dir: directory to save plots.
        show: whether to display plots.

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
        results_by_problem,
        aggregated,
        problem_type,
        L_star=L_star_global,
        L0=L0,
        e_target_epsilon=e_target_epsilon,
        save_path=save_dir / f"{problem_type}_level1_global_comparison.png",
        show=show,
    )
    saved_files.append(str(save_dir / f"{problem_type}_level1_global_comparison.png"))

    # Level 2: By Optimizer (2x2 layout) - with E_target threshold if available
    plot_by_optimizer(
        results_by_problem,
        aggregated,
        problem_type,
        L_star=L_star_global,
        L0=L0,
        e_target_epsilon=e_target_epsilon,
        save_path=save_dir / f"{problem_type}_level2_by_optimizer.png",
        show=show,
    )
    saved_files.append(str(save_dir / f"{problem_type}_level2_by_optimizer.png"))

    # Level 3: Seed Variance - with E_target threshold if available
    plot_seed_variance(
        results_by_problem,
        problem_type,
        L_star_global=L_star_global,
        L0=L0,
        e_target_epsilon=e_target_epsilon,
        save_path=save_dir / f"{problem_type}_level3_seed_variance.png",
        show=show,
    )
    saved_files.append(str(save_dir / f"{problem_type}_level3_seed_variance.png"))

    # Level 4: Final Performance
    plot_final_performance(
        aggregated,
        problem_type,
        save_path=save_dir / f"{problem_type}_level4_final_performance.png",
        show=show,
    )
    saved_files.append(str(save_dir / f"{problem_type}_level4_final_performance.png"))

    # Analytical A: Loss + LR Dual
    plot_loss_lr_dual(
        results_by_problem,
        aggregated,
        problem_type,
        save_path=save_dir / f"{problem_type}_analytical_a_loss_lr_dual.png",
        show=show,
    )
    saved_files.append(str(save_dir / f"{problem_type}_analytical_a_loss_lr_dual.png"))

    logger.info(f"All plots saved for: {problem_type}")
    return saved_files
