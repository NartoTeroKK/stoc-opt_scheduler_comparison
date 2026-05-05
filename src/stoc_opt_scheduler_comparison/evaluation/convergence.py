"""
Convergence metrics for stochastic optimization experiments.

Six metrics: EtT, AUL, rho_hat, RV/CV, SI, Robbins-Monro check.
Plus: compute_convergence_metrics() and aggregate_metrics() for batch processing.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats
import numpy.typing as npt


from stoc_opt_scheduler_comparison.evaluation.metrics import TrainingHistory

# Alias riutilizzabile in tutto il progetto
FloatArray = npt.NDArray[np.float64]

# ── Individual Metrics ───────────────────────────────────────────────────────────

def convergence_threshold(
    L0: float,
    L_star_global: float,
    epsilon: float = 0.10,
    relative: bool = True,
) -> float:
    """
    Compute the convergence threshold used by epochs_to_threshold.

    If relative=True:  threshold = L_star + epsilon * (L0 - L_star)
                       → loss value at which epsilon-fraction of initial gap remains.
    If relative=False: threshold = L_star + epsilon (absolute tolerance).

    Returns nan if gap < 1e-12 (L0 already at global minimum).

    Intended for external use: pass the returned value as a horizontal
    reference line in loss curve plots to visualise the convergence target.
    """
    gap = L0 - L_star_global
    if gap < 1e-12:
        return float("nan")
    return (L_star_global + epsilon * gap) if relative else (L_star_global + epsilon)


def epochs_to_threshold(
    loss_curve: FloatArray,
    L_star_global: float,
    epsilon: float = 0.10,
    relative: bool = True,
) -> float:
    L0 = float(loss_curve[0])
    threshold = convergence_threshold(L0, L_star_global, epsilon, relative)

    if np.isnan(threshold):
        return float("nan")

    if L0 <= threshold:
        return 0.0

    indices = np.where(np.array(loss_curve, dtype=float) <= threshold)[0]
    return float(indices[0]) if len(indices) > 0 else float("nan")


def area_under_loss(loss_curve: FloatArray, L_star_global: float) -> tuple[float, float]:
    """
    Trapezoidal integral of loss curve.
    Returns (aul, aul_norm). Lower = faster convergence.

    Normalization: area under a constant curve at initial value above L_star.
    """
    x = np.arange(len(loss_curve))
    aul = float(np.trapezoid(loss_curve, x=x))

    denominator = len(loss_curve) * float(loss_curve[0] - L_star_global)
    aul_norm = aul / denominator if denominator > 1e-12 else float("inf")

    return aul, aul_norm


def empirical_convergence_rate(loss_curve: FloatArray, L_star_global: float, warm_up: int = 5) -> dict:
    """
    Estimate per-epoch convergence rate via log-linear regression on the optimality gap.

    Regression is applied only on the asymptotic tail (second half of post-warm-up curve)
    to better capture the true linear convergence regime, avoiding transient dynamics.

    rho_hat << 1 = fast convergence
    rho_hat ~  1 = slow / stalling
    rho_hat >  1 = divergence

    If R² < 0.80, the loss trajectory is not log-linear (e.g. non-convex oscillations),
    and rho_hat is set to None to avoid misleading estimates.
    """
    # Skip warm-up epochs to avoid transient dynamics
    post_warmup = np.asarray(loss_curve[warm_up:], dtype=float)

    if len(post_warmup) < 4:
        return {"rho_hat": None, "log_E0": 0.0, "R2": 0.0, "slope": 0.0, "std_err": 0.0}

    # Use only the asymptotic tail (second half) to fit the linear convergence regime
    half = len(post_warmup) // 2
    loss_arr = post_warmup[half:]

    # Optimality gap: clipped to avoid log(0) or log of negative values
    E_k = np.clip(loss_arr - L_star_global, a_min=1e-12, a_max=None)
    log_E = np.log(E_k)
    k = np.arange(len(log_E), dtype=float)

    lr_result = stats.linregress(k, log_E)

    r2 = float(lr_result.rvalue ** 2)  # type: ignore[union-attr]

    # If R² < 0.80, the log-linear model does not fit: convergence is non-linear
    rho_hat = float(np.exp(lr_result.slope)) if r2 >= 0.80 else float("nan") # type: ignore[union-attr]

    return {
        "rho_hat": rho_hat,
        "log_E0": float(lr_result.intercept),  # type: ignore[union-attr]
        "R2": r2,
        "slope": float(lr_result.slope),       # type: ignore[union-attr]
        "std_err": float(lr_result.stderr),    # type: ignore[union-attr]
    }


def rolling_variance_cv(loss_curve: FloatArray, window_frac: float = 0.15) -> dict:
    """Variance and CV on final window of loss curve. Measures stability."""
    W = max(5, int(len(loss_curve) * window_frac))
    window = loss_curve[-W:]
    mu = float(np.mean(window))
    rv = float(np.var(window, ddof=0))
    cv = float(np.std(window, ddof=0) / mu) if mu > 1e-12 else float("inf")
    return {"RV": rv, "CV_final": cv, "mu_final": mu, "W": W}


def smoothness_index(loss_curve: FloatArray) -> dict:
    """Mean absolute epoch-to-epoch change. Lower = smoother trajectory."""
    diffs = np.abs(np.diff(loss_curve))
    mid = len(diffs) // 2
    return {
        "SI":            float(np.mean(diffs)),           # global
        "SI_asymptotic": float(np.mean(diffs[mid:])),     # stability phase
    }



# ── Batch Processing ─────────────────────────────────────────────────────────────

_DEFAULT_PARAMS = {
    "convex": {
        "epsilon": 0.10,           # ε-relative suboptimality: threshold at 90% of (L0 - L_star) gap
        "relative_epsilon": True,  # if True: threshold = L_star + ε*(L0 - L_star), else absolute
        "window_frac": 0.10,       # trailing window for CV_final and RV: last 10% of epochs
        "warm_up": 5,              # epochs excluded from rho_hat estimate: 5% of total (transient phase)
    },
    "non-convex": {
        "epsilon": 0.10,           # same ε for cross-problem comparability
        "relative_epsilon": True,
        "window_frac": 0.10,       # last 10 epochs out of 100 for asymptotic stability estimate
        "warm_up": 10,             # 10% warm-up exclusion: higher gradient variance with dropout
    },
}


def compute_L_star_global(results_by_problem: dict) -> tuple[float, str]:
    """
    Find global minimum loss across all runs for a problem type.

    Returns:
        (L_star, best_run_name) - minimum loss value and the run that achieved it.
    """
    best_loss = float("inf")
    best_run = ""

    for run_name, history in results_by_problem.items():
        if history and history.train_losses:
            run_min = min(history.train_losses)
            if run_min < best_loss:
                best_loss = run_min
                best_run = run_name

    return best_loss if best_run else 0.0, best_run

def _parse_run_name(name: str) -> tuple[str, str, str, int] | None:
    """Parse 'convex_cosine_sgd_42' → (problem, scheduler, optimizer, seed)."""
    parts = name.split('_')
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], int(parts[3])


def compute_convergence_metrics(
    results: dict,
    problem_type: str,
    L_star: float,
    **metric_kwargs,
) -> dict[str, dict]:
    """
    Compute convergence metrics for all experiment runs.

    Args:
        results: {problem_type: {run_name: {"history": TrainingHistory, "test_metrics": {...}}}}
        problem_type: "convex" or "non-convex"
        optimizer: Optional filter (e.g., "sgd")
        scheduler: Optional filter (e.g., "cosine")
        **metric_kwargs: Override default epsilon, window_frac, warm_up

    Returns:
        List of metric dicts, one per run.
    """
    if problem_type not in results:
        raise ValueError(f"Unknown problem_type: {problem_type}")
    
    results_by_problem = results[problem_type]

    params = {**_DEFAULT_PARAMS.get(problem_type, {}), **metric_kwargs}
    # return
    metrics_dict: dict[str, dict] = {} 


    for run_name, history in results_by_problem.items():
        if history is None:
            continue

        test_metrics = history.test_metrics

        if history.train_losses_arr.size == 0:
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
            **params,
        )
        m["test_loss"] = test_metrics.get("loss", 0.0)
        m["test_accuracy"] = test_metrics.get("accuracy", 0.0)
        m["optimizer"] = opt
        m["scheduler"] = sched
        m["seed"] = seed

        metrics_dict[run_name] = m   

    return metrics_dict


def compute_all_metrics(
    loss_curve: FloatArray,
    train_accuracies: FloatArray,
    lr_history: FloatArray,
    L_star_global: float,
    epsilon: float = 1e-3,
    relative_epsilon: bool = True,
    window_frac: float = 0.15,
    warm_up: int = 5,
) -> dict:
    """Compute all six metrics for a single run. Returns flat dict."""
    ett = epochs_to_threshold(loss_curve, L_star_global, epsilon, relative_epsilon)
    aul, aul_n = area_under_loss(loss_curve, L_star_global)
    conv = empirical_convergence_rate(loss_curve, L_star_global, warm_up)
    rv_cv = rolling_variance_cv(loss_curve, window_frac)
    si = smoothness_index(loss_curve)

    return {
        "train_losses_arr": loss_curve,
        "train_accuracies_arr": train_accuracies,
        "learning_rates_arr": lr_history,
        "EtT": ett,
        "AUL": round(aul, 6),
        "AUL_norm": round(aul_n, 6),
        "rho_hat": round(conv["rho_hat"], 6) if not np.isnan(conv["rho_hat"]) else float("nan"),
        "R2": round(conv["R2"], 4),
        "RV": round(rv_cv["RV"], 8),
        "CV_final": round(rv_cv["CV_final"], 6),
        "mu_final": round(rv_cv["mu_final"], 6),
        "SI": round(si["SI"], 6),
        "SI_asymptotic": round(si["SI_asymptotic"], 6),
    }

# Metriche con aggregazione standard: media + std
STANDARD_METRICS = ["AUL", "AUL_norm", "rho_hat", "R2", "RV", "CV_final", "SI_asymptotic",
                    "test_loss", "test_accuracy"]

def aggregate_metrics(metrics_dict: dict[str, dict]) -> dict:
    """
    Aggregate metrics by optimizer-scheduler combination (mean ± std across seeds).

    For each (optimizer, scheduler) pair:
    - Scalar metrics: mean and std across seeds
    - Training curves (train_losses, train_accuracies, learning_rates):
      Stack arrays, compute mean and std along axis=0 → returns (mean_array, std_array)

    Returns:
        Dict with keys: "{optimizer}_{scheduler}" -> aggregated metrics dict
    """
    groups = defaultdict(list)
    for m in metrics_dict.values():
        groups[f"{m['optimizer']}_{m['scheduler']}"].append(m)

    aggregated = {}
    for key, group in groups.items():
        opt, sched = key.split("_", 1)
        agg = {"optimizer": opt, "scheduler": sched, "n_runs": len(group)}

        # Scalar metrics: mean and std across seeds (only keys that exist)
        all_keys = set()
        for m in group:
            all_keys.update(m.keys())

        # EtT separato: media + std + mediana + n_converged
        for metric_key in STANDARD_METRICS:
            if metric_key in all_keys:
                values = [m[metric_key] for m in group
                        if metric_key in m and m[metric_key] is not None]
                if values:
                    agg[f"{metric_key}_mean"] = float(np.mean(values))
                    agg[f"{metric_key}_std"]  = float(np.std(values))

        # EtT: aggregazione robusta
        ett_all = [m["EtT"] for m in group if "EtT" in m]
        ett_valid = [v for v in ett_all if v is not None and not np.isnan(v)]

        n_total = len(ett_all)
        n_valid = len(ett_valid)

        agg["EtT_convergence_rate"] = n_valid / n_total if n_total > 0 else float("nan")

        if ett_valid:
            agg["EtT_mean"]   = float(np.mean(ett_valid))
            agg["EtT_std"]    = float(np.std(ett_valid))
            agg["EtT_median"] = float(np.median(ett_valid))
        else:
            agg["EtT_mean"]   = float("nan")
            agg["EtT_std"]    = float("nan")
            agg["EtT_median"] = float("nan")

        # Training curves: stack numpy arrays, compute mean/std
        loss_curves = [m["train_losses_arr"] for m in group if "train_losses_arr" in m]
        acc_curves = [m["train_accuracies_arr"] for m in group if "train_accuracies_arr" in m]
        lr_curves = [m["learning_rates_arr"] for m in group if "learning_rates_arr" in m]

        if loss_curves:
            stacked = np.stack(loss_curves)
            agg["train_losses_mean"] = stacked.mean(axis=0)
            agg["train_losses_std"] = stacked.std(axis=0)

        if acc_curves:
            stacked = np.stack(acc_curves)
            agg["train_accuracies_mean"] = stacked.mean(axis=0)
            agg["train_accuracies_std"] = stacked.std(axis=0)

        if lr_curves:
            stacked = np.stack(lr_curves)
            agg["learning_rates_mean"] = stacked.mean(axis=0)
            agg["learning_rates_std"] = stacked.std(axis=0)

        aggregated[key] = agg

    return aggregated
