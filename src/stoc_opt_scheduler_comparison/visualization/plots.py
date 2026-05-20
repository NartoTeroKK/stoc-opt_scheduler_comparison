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

from stoc_opt_scheduler_comparison.utils import viz_logger as logger
from stoc_opt_scheduler_comparison.evaluation.convergence import (
    convergence_threshold,
    compute_global_initial_loss,
    compute_target_loss,
)

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
    "mean": {"linewidth": 2.5, "alpha": 1.0},
}

SCHEDULERS_LIST = ["none", "exponential", "cosine", "cyclic", "one-cycle"]
OPTIMIZERS_LIST = ["sgd", "adam"]

# E_target threshold line styles (3 levels: strict, moderate, permissive)
E_TARGET_COLORS: list[str] = [
    "#bf75f0",  # lv1 (5%): Viola Chiaro / Malva (Soglia più alta, facile da superare)
    "#7b3fa7",  # lv2 (2.5%): Viola Medio / Lavanda d'Europa
    "#742465",  # lv3 (1%): Viola Scuro / Bizantino (Traguardo finale, asintotico)
]
E_TARGET_LINESTYLES: list[str] = ["-.", "-.", "-."]
E_TARGET_LABELS: list[str] = ["lv1 (permissive)", "lv2 (moderate)", "lv3 (strict)"]

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
    L_star: float | None = None,
    L0: float | None = None,
    e_target_levels: dict[str, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 2: 2×2 grid - rows=optimizers (SGD, Adam), cols=linear/log scale.

    Each subplot shows 5 scheduler curves with mean±std from aggregated arrays.
    Left column: linear scale.
    Right column: log scale for loss.

    If L_star, L0, and e_target_levels are provided, horizontal threshold lines
    are drawn at the computed L_target values for each tolerance level.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Pre-compute E_target threshold values if parameters provided
    e_target_values: list[float] = []
    if L_star is not None and L0 is not None and e_target_levels is not None:
        for eps in e_target_levels.values():
            e_target_values.append(compute_target_loss(L_star, eps, problem_type, L0))

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

        # E_target threshold lines on linear subplot
        for i, L_target in enumerate(e_target_values):
            ax_linear.axhline(
                y=L_target, color=E_TARGET_COLORS[i],
                linestyle=E_TARGET_LINESTYLES[i], linewidth=1.0, alpha=0.7,
                label=f"E_target {E_TARGET_LABELS[i]}" if idx == 0 else "",
            )
        if idx == 0 and e_target_values:
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

            ax_log.plot(epochs, mean_arr, color=color, label=sched,
                        linestyle=style["linestyle"], linewidth=style["linewidth"])
            ax_log.fill_between(epochs, mean_arr - std_arr, mean_arr + std_arr,
                                   color=color, alpha=style["alpha_std"])

        # E_target threshold lines on log subplot
        for i, L_target in enumerate(e_target_values):
            ax_log.axhline(
                y=L_target, color=E_TARGET_COLORS[i],
                linestyle=E_TARGET_LINESTYLES[i], linewidth=1.0, alpha=0.7,
            )

        ax_log.set_xlabel("Epoch")
        ax_log.set_ylabel("Training Loss (log scale)")
        ax_log.set_title(f"{problem_type.upper()} - {opt.upper()} (Log Scale)")
        ax_log.set_yscale("log")
        ax_log.grid(True, alpha=0.3)

    # Build subtitle with E_target info
    e_target_subtitle = ""
    if e_target_values:
        level_names = list(e_target_levels.keys()) if e_target_levels else []
        vals_str = ", ".join(f"{n}={v:.6f}" for n, v in zip(level_names, e_target_values))
        e_target_subtitle = f" | E_target thresholds: {vals_str}"

    add_figure_title(
        fig,
        title=f"{problem_type.upper()} - Loss Curves by Optimizer",
        subtitle=f"Left: Linear scale | Right: Log scale{e_target_subtitle}"
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
    L0: float | None = None,
    e_target_levels: dict[str, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Level 3: (5×2) grid - diagnostic seed variance.

    Each cell (scheduler, optimizer): 5 thin seed lines + 1 thick mean line.
    Optional dashed convergence threshold line per cell (computed via
    convergence_threshold using the mean L0 of the seed group).
    If L0 and e_target_levels are provided, global E_target threshold lines
    are drawn across all cells.
    No rolling mean - raw arrays only.
    """
    _setup_style()

    fig, axes = plt.subplots(5, 2, figsize=(12, 20), sharex=True, sharey=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes.ravel()

    # Pre-compute global E_target threshold values
    e_target_values: list[float] = []
    if L_star_global is not None and L0 is not None and e_target_levels is not None:
        for eps in e_target_levels.values():
            e_target_values.append(compute_target_loss(L_star_global, eps, problem_type, L0))

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

            # --- Global E_target threshold lines ---
            for i, L_target in enumerate(e_target_values):
                ax.axhline(
                    y=L_target, color=E_TARGET_COLORS[i],
                    linestyle=E_TARGET_LINESTYLES[i], linewidth=0.8, alpha=0.6,
                )

        ax.set_title(f"{sched} ({opt})", fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx % 2 == 0:
            ax.set_ylabel("Training Loss")

    y_max = np.percentile(all_loss_values, 95) * 1.05
    y_min = max(0.0, np.percentile(all_loss_values, 1) * 0.95)

    for ax in axes:
        ax.set_ylim(y_min, y_max)

    # Build subtitle with E_target info
    subtitle = "Each subplot: 10 thin seed lines + 1 thick mean line | dashed = per-cell threshold"
    if e_target_values:
        level_names = list(e_target_levels.keys()) if e_target_levels else []
        vals_str = ", ".join(f"{n}={v:.6f}" for n, v in zip(level_names, e_target_values))
        subtitle += f"\nGlobal E_target: {vals_str}"

    add_figure_title(
        fig,
        title=f"{problem_type.upper()} - Seed Variance Analysis",
        subtitle=subtitle,
    )

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig

# ── Convergence Boxplot (Epochs-to-Target) ─────────────────────────────

def plot_convergence_boxplot(
    df_convergence: pd.DataFrame,
    problem_type: str,
    L_target: float,
    max_epochs: int = 100,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """
    Boxplot raggruppato (Hue=Optimizer, X=Scheduler).
    Calcola la statistica del boxplot SOLO sulle run convergenti.
    I Timeout sono tracciati manualmente con offset precisi per mantenere 
    il layout intatto anche quando intere categorie falliscono.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    opt_palette = {"ADAM": "#2b83ba", "SGD": "#d7191c"}

    # 1. Definizione Categorie (Garantisce ordine asse X)
    sched_order = df_convergence["Scheduler"].unique()
    opt_order = ["ADAM", "SGD"]
    
    df_convergence["Scheduler"] = pd.Categorical(df_convergence["Scheduler"], categories=sched_order, ordered=True)
    df_convergence["Optimizer"] = pd.Categorical(df_convergence["Optimizer"], categories=opt_order, ordered=True)

    # 2. Separazione Dati IBRIDA (La vera magia)
    is_timeout = (df_convergence["Epochs_to_Target"] >= max_epochs) | (df_convergence["Epochs_to_Target"].isna())
    
    # A) Per SEABORN (Boxplot e punti neri): NON eliminiamo le righe. 
    # Mettiamo a NaN i timeout. Così Seaborn riserva lo slot ma non disegna nulla.
    df_converged = df_convergence.copy()
    df_converged.loc[is_timeout, "Epochs_to_Target"] = np.nan
    
    # B) Per MATPLOTLIB (Le 'X' rosse): Filtriamo e teniamo solo i veri timeout.
    df_timeout = df_convergence[is_timeout].copy()

    # 3. PLOT 1: Boxplot e Stripplot dei Convergenti
    # Ora passiamo df_converged con i NaN. Seaborn manterrà l'architettura intatta.
    sns.boxplot(
        data=df_converged,
        x="Scheduler",
        y="Epochs_to_Target",
        hue="Optimizer",
        hue_order=opt_order,
        palette=opt_palette,
        width=0.6,
        linewidth=1.5,
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

    # 4. PLOT 2: PLOT MANUALE DEI TIMEOUT (La Soluzione)
    if not df_timeout.empty:
        # Creiamo un dizionario che mappa ogni scheduler alla sua coordinata X (0, 1, 2, 3...)
        x_map = {sched: i for i, sched in enumerate(sched_order)}
        
        # Definiamo lo spostamento orizzontale (dodge)
        # Se usiamo width=0.6 nel boxplot, il centro del boxplot di sinistra è a -0.15, quello di destra a +0.15
        offset_map = {"ADAM": -0.15, "SGD": 0.15}

        for _, row in df_timeout.iterrows():
            sched = row["Scheduler"]
            opt = row["Optimizer"]
            
            # Calcoliamo la posizione X esatta
            base_x = x_map[sched]
            offset = offset_map[opt]
            
            # Aggiungiamo un leggero jitter manuale orizzontale per non sovrapporre le X
            jitter = np.random.uniform(-0.05, 0.05) 
            final_x = base_x + offset + jitter
            
            # Tracciamo la singola X
            ax.plot(
                final_x, max_epochs,
                marker="X",
                color=opt_palette[opt], # Coloriamo la X col colore dell'ottimizzatore per maggiore chiarezza
                markersize=7,
                alpha=0.9,
                linestyle='None'
            )

    # 5. Legenda (Pulita)
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = handles[:2]
    unique_labels = labels[:2]

    if not df_timeout.empty:
        timeout_handle = Line2D(
            [0], [0], marker='X', color='w', 
            markerfacecolor='gray', markersize=8
        )
        unique_handles.append(timeout_handle)
        unique_labels.append('Timeout (Non Converso)')

    ax.legend(
        unique_handles, unique_labels, 
        title="Legenda", loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False
    )

    # 6. Layout e Assi
    ax.set_title(
        f"{problem_type.upper()} - Convergence Velocity (Time-to-Target)",
        fontsize=16, fontweight='bold', pad=20,
    )
    ax.set_xlabel("Learning Rate Scheduler", fontsize=14, labelpad=15)
    ax.set_ylabel(f"Epochs to Reach Target ({L_target:.4f})", fontsize=14, labelpad=15)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    
    ax.set_ylim(-2, max_epochs + 5)
    ax.axhline(y=max_epochs, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
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

    fig, axes = plt.subplots(5, 2, figsize=(12, 20), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes.ravel()

    for idx, (opt, sched) in enumerate([(o, s) for s in SCHEDULERS_LIST for o in OPTIMIZERS_LIST]):
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
                    color=lr_color, linestyle=":", linewidth=1.0, label="LR")
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
        subtitle="Each subplot: Loss (left axis) + LR (right axis, dotted)"
    )
    
    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Analytical Plot B: Heatmap ───────────────────────────────────

_LOWER_IS_BETTER = {
    "E_target", "suboptimality_gap", "AUL_norm", 
    "CV_final", "SI_asymptotic", "RV", "test_loss"
}
_HAS_CRITICAL_POINT = {"R2": 0.5}  # Predisposto se aggiungerai R2

def _get_heatmap_style(metric: str) -> dict:
    """Returns kwargs for sns.heatmap based on metric semantics."""
    if metric in _HAS_CRITICAL_POINT:
        return {"cmap": "RdBu_r", "center": _HAS_CRITICAL_POINT[metric]}

    # Palette Accademica Single-Tone: Colore più scuro = Risultato Migliore
    if metric in _LOWER_IS_BETTER:
        return {"cmap": "Blues_r"}  # Basso = scuro, Alto = chiaro
    
    return {"cmap": "Blues"}        # Alto = scuro, Basso = chiaro

def _get_direction_label(metric: str) -> str:
    if metric in _LOWER_IS_BETTER:
        return "↓ lower is better"
    if metric in _HAS_CRITICAL_POINT:
        return f"critical point = {_HAS_CRITICAL_POINT[metric]}"
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
    df: pd.DataFrame,
    problem_type: str,
    metrics: list[str] = ["E_target", "AUL_norm", "CV_final"],
    save_path: str | Path | None = None,
    show: bool = False,
) -> list[str]:
    """
    Analytical B: Heatmap of configurable metrics directly from DataFrame.

    Rows = schedulers, Columns = optimizers.
    """
    saved_files = []

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame. Skipping.")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # 1. Creazione elegante della matrice 2D tramite pivot table
        pivot_df = df.pivot(index="scheduler", columns="optimizer", values=metric)
        matrix = pivot_df.to_numpy()
        y_labels = pivot_df.index.tolist()
        x_labels = pivot_df.columns.tolist()

        # 2. Configurazione stile e scale
        style = _get_heatmap_style(metric)
        direction = _get_direction_label(metric)
        vmin, vmax = _robust_colorscale(matrix)
        outlier_mask = _outlier_mask_iqr(matrix)

        # 3. Costruzione della matrice di annotazione testuale (gestisce NaN format)
        def format_cell(val: float, is_outlier: bool) -> str:
            if np.isnan(val):
                return ""
            # Usa 0 decimali se il numero è intero (es. E_target), 4 se è un float (es. CV_final)
            num_str = f"{val:.0f}" if float(val).is_integer() else f"{val:.4f}"
            return f"{num_str}*" if is_outlier else num_str

        annot_matrix = np.vectorize(format_cell)(matrix, outlier_mask)

        # 4. Rendering Heatmap
        sns.heatmap(
            matrix, vmin=vmin, vmax=vmax, annot=annot_matrix, fmt="",
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax, **style
        )
        
        ax.set_title(f"{problem_type.upper()} — {metric}  ({direction})")
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Scheduler")

        # 5. Salvataggio
        if save_path:
            heatmap_path = str(save_path).replace(".png", f"_{metric.lower()}.png")
            # Sostituisci con la tua funzione _save(fig, heatmap_path) se necessario
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
    e_target_levels: dict[str, float] | None = None,
    epsilon: float | None = None,
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
        e_target_levels: optional dict of {name: epsilon} for E_target thresholds.
        epsilon: optional, for convergence threshold display.
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
        results_by_problem, aggregated, problem_type,
        save_path=save_dir / f"{problem_type}_level1_global_comparison.png",
        show=show
    )
    saved_files.append(str(save_dir / f"{problem_type}_level1_global_comparison.png"))
    
    # Level 2: By Optimizer (2x2 layout) - with E_target thresholds if available
    plot_by_optimizer(
        results_by_problem, aggregated, problem_type,
        L_star=L_star_global,
        L0=L0,
        e_target_levels=e_target_levels,
        save_path=save_dir / f"{problem_type}_level2_by_optimizer.png",
        show=show
    )
    saved_files.append(str(save_dir / f"{problem_type}_level2_by_optimizer.png"))
    
    # Level 3: Seed Variance - with E_target thresholds if available
    plot_seed_variance(
        results_by_problem, problem_type,
        L_star_global=L_star_global,
        L0=L0,
        e_target_levels=e_target_levels,
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

    
    logger.info(f"All plots saved for: {problem_type}")
    return saved_files
