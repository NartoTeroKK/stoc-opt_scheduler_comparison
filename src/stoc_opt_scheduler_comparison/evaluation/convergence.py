"""
Convergence metrics for stochastic optimization experiments.

Metrics (from methodology):
  - EtT (Epochs to Target): first epoch loss ≤ L_target = L* + 0.05·(L0 − L*)
  - Suboptimality gap:     Δ = L_T* − L*
  - AUL (Area Under Loss): trapezoidal integral of loss curve
  - CV (Coefficient of Variation): σ/μ over full or post-warmup loss curve
  - Gradient norm statistics: mean and std of gradient ℓ2-norm per epoch
  - Convergence rate:      proportion of runs reaching L_target

All metrics are computed per-seed first, then aggregated (mean ± std).
Non-converging runs are excluded from EtT statistics.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]

# ── Constants ─────────────────────────────────────────────────────────────

# Level 1 (primary): 5 % of the initial (L0 − L*) gap.
ETT_EPSILON = 0.05


# ── L₀ / L* helpers ───────────────────────────────────────────────────────

def compute_global_L_star(results_by_problem: dict) -> float:
    """Global minimum loss across all runs for a problem type.

    Returns:
        Smallest train loss observed across every seed and configuration.
        Returns NaN when there are no valid runs.
    """
    best_loss = float("inf")
    for history in results_by_problem.values():
        if history is not None and history.train_losses:
            run_min = min(history.train_losses)
            if run_min < best_loss:
                best_loss = run_min
    return best_loss if best_loss < float("inf") else float("nan")


def compute_global_initial_loss(results_by_problem: dict) -> float:
    """Mean of the first-epoch loss across all runs (L₀).

    Serves as baseline for defining the target loss L_target.
    Returns NaN if no valid runs exist.
    """
    losses: list[float] = []
    for history in results_by_problem.values():
        if history is not None and history.train_losses:
            losses.append(history.train_losses[0])
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def compute_target_loss(L_star: float, L0: float, epsilon: float = ETT_EPSILON) -> float:
    """Target loss: L* + ε · (L₀ − L*)."""
    return L_star + epsilon * (L0 - L_star)


# ── Per-seed metric functions ─────────────────────────────────────────────

def epochs_to_target(loss_curve: FloatArray, L_target: float, max_epochs: int) -> int:
    """Earliest epoch where loss ≤ L_target.

    Returns *max_epochs* when the target is never reached (callers should
    treat this as *did not converge* and convert to NaN for aggregation).
    """
    indices = np.where(loss_curve <= L_target)[0]
    if len(indices) > 0:
        return int(indices[0])
    return max_epochs


def suboptimality_gap(loss_curve: FloatArray, L_star: float) -> float:
    """Δ = final loss − L*."""
    return float(loss_curve[-1]) - L_star


def area_under_loss(loss_curve: FloatArray) -> float:
    """Trapezoidal integral of the loss curve.  Lower = faster drop."""
    return float(np.trapezoid(loss_curve, x=np.arange(len(loss_curve))))


def coefficient_of_variation(loss_curve: FloatArray, t0: int = 0) -> float:
    """σ/μ over loss_curve[t₀:] (default: full curve).

    Args:
        loss_curve: per-epoch loss values.
        t0: warm-up epochs to exclude (0 = use whole curve).

    Returns:
        CV, or NaN if the mean is near zero.
    """
    segment = loss_curve[t0:]
    if len(segment) < 2:
        return float("nan")
    mu = float(np.mean(segment))
    if abs(mu) < 1e-12:
        return float("nan")
    return float(np.std(segment, ddof=0)) / mu


def gradient_norm_statistics(grad_norms: FloatArray) -> tuple[float, float]:
    """Mean and std of per-epoch gradient ℓ2-norms.

    Returns (NaN, NaN) when *grad_norms* is empty (e.g. not logged).
    """
    if len(grad_norms) == 0:
        return float("nan"), float("nan")
    return float(np.mean(grad_norms)), float(np.std(grad_norms, ddof=0))


# ── Run-name parsing ──────────────────────────────────────────────────────

def _parse_run_name(name: str) -> tuple[str, str, str, int] | None:
    """Parse ``convex_cosine_sgd_42`` → (problem, scheduler, optimizer, seed)."""
    parts = name.split("_")
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], int(parts[3])


# ── Batch processing ──────────────────────────────────────────────────────

def compute_all_metrics(
    loss_curve: FloatArray,
    train_accuracies: FloatArray,
    lr_history: FloatArray,
    L_star_global: float,
    L_target: float,
    max_epochs: int,
    t0_cv: int = 0,
) -> dict:
    """All convergence metrics for a single run.

    EtT is NaN for runs that never reach L_target (excluded from aggregation).
    Gradient-norm statistics return NaN when *grad_norms* are unavailable.
    """
    ett_raw = epochs_to_target(loss_curve, L_target, max_epochs)
    ett = float(ett_raw) if ett_raw < max_epochs else float("nan")

    aul = area_under_loss(loss_curve)
    subopt = suboptimality_gap(loss_curve, L_star_global)
    cv = coefficient_of_variation(loss_curve, t0=t0_cv)

    return {
        "train_losses_arr": loss_curve,
        "train_accuracies_arr": train_accuracies,
        "learning_rates_arr": lr_history,
        # Convergence metrics
        "EtT": ett,
        "AUL": round(aul, 6),
        "suboptimality_gap": round(subopt, 8),
        "CV": round(cv, 6),
        # Gradient norm statistics (filled by caller when available)
        "mean_gradient_norm": float("nan"),
        "std_gradient_norm": float("nan"),
        # Test metrics (filled by caller)
        "test_loss": 0.0,
        "test_accuracy": 0.0,
    }


def compute_convergence_metrics(
    results: dict,
    problem_type: str,
    L_star: float,
    L0: float,
    max_epochs: int,
) -> dict[str, dict]:
    """Compute convergence metrics for every run in *results*.

    Args:
        results: ``{problem_type: {run_name: TrainingHistory}}``
        problem_type: ``"convex"`` or ``"non-convex"``
        L_star: global minimum loss (from :func:`compute_global_L_star`)
        L0: global initial loss (from :func:`compute_global_initial_loss`)
        max_epochs: total number of training epochs.

    Returns:
        ``{run_name: metric_dict}`` — one entry per valid run.
    """
    if problem_type not in results:
        raise ValueError(f"Unknown problem_type: {problem_type!r}")

    results_by_problem = results[problem_type]
    L_target = compute_target_loss(L_star, L0)

    metrics_dict: dict[str, dict] = {}

    for run_name, history in results_by_problem.items():
        if history is None or history.train_losses_arr.size == 0:
            continue

        parsed = _parse_run_name(run_name)
        if parsed is None:
            continue

        _, sched, opt, seed = parsed

        m = compute_all_metrics(
            loss_curve=history.train_losses_arr,
            train_accuracies=history.train_accuracies_arr,
            lr_history=history.learning_rates_arr,
            L_star_global=L_star,
            L_target=L_target,
            max_epochs=max_epochs,
        )

        test_metrics = history.test_metrics
        m["test_loss"] = test_metrics.get("loss", 0.0)
        m["test_accuracy"] = test_metrics.get("accuracy", 0.0)
        m["optimizer"] = opt
        m["scheduler"] = sched
        m["seed"] = seed

        metrics_dict[run_name] = m

    return metrics_dict


# ── Aggregation ───────────────────────────────────────────────────────────

def _is_valid(v) -> bool:
    """True for finite numeric values; False for None, NaN, Inf."""
    if v is None:
        return False
    try:
        return not np.isnan(v) and not np.isinf(v)
    except (TypeError, ValueError):
        return True


SCALAR_METRICS = [
    "AUL",
    "suboptimality_gap",
    "CV",
    "mean_gradient_norm",
    "std_gradient_norm",
    "test_loss",
    "test_accuracy",
]


def aggregate_metrics(metrics_dict: dict[str, dict]) -> dict:
    """Aggregate per-seed metrics by (optimizer, scheduler).

    For each scalar metric: mean ± std across **valid** entries.
    EtT is aggregated **only over converging runs** (non-NaN).
    Convergence rate: ``n_converged / n_total``.

    Training curves (losses, accuracies, learning rates) are stacked and
    their mean ± std per epoch is stored as arrays.
    """
    groups = defaultdict(list)
    for m in metrics_dict.values():
        groups[f"{m['optimizer']}_{m['scheduler']}"].append(m)

    aggregated: dict = {}
    for key, group in groups.items():
        opt, sched = key.split("_", 1)
        agg: dict = {"optimizer": opt, "scheduler": sched, "n_runs": len(group)}

        # -- Scalar metrics: mean ± std --
        for metric_key in SCALAR_METRICS:
            values = [
                m[metric_key]
                for m in group
                if metric_key in m and _is_valid(m[metric_key])
            ]
            if values:
                agg[f"{metric_key}_mean"] = float(np.mean(values))
                agg[f"{metric_key}_std"] = float(np.std(values))

        # -- EtT: converging runs only + convergence rate --
        ett_all = [m["EtT"] for m in group if "EtT" in m]
        ett_valid = [v for v in ett_all if _is_valid(v)]
        n_total = len(ett_all)
        n_converged = len(ett_valid)

        agg["EtT_convergence_rate"] = (
            n_converged / n_total if n_total > 0 else float("nan")
        )

        if ett_valid:
            agg["EtT_mean"] = float(np.mean(ett_valid))
            agg["EtT_std"] = float(np.std(ett_valid))
            agg["EtT_median"] = float(np.median(ett_valid))
        else:
            agg["EtT_mean"] = float("nan")
            agg["EtT_std"] = float("nan")
            agg["EtT_median"] = float("nan")

        # -- Training curve arrays --
        for arr_key, mean_key, std_key in [
            ("train_losses_arr", "train_losses_mean", "train_losses_std"),
            ("train_accuracies_arr", "train_accuracies_mean", "train_accuracies_std"),
            ("learning_rates_arr", "learning_rates_mean", "learning_rates_std"),
        ]:
            curves = [m[arr_key] for m in group if arr_key in m and m[arr_key].size > 0]
            if curves:
                stacked = np.stack(curves)
                agg[mean_key] = stacked.mean(axis=0)
                agg[std_key] = stacked.std(axis=0)

        aggregated[key] = agg

    return aggregated
