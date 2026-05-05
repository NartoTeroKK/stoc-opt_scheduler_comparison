"""
MLflow tracking manager - MLflowLogger for experiment lifecycle management.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import mlflow


class MLflowLogger:
    """
    Manages MLflow experiment tracking with nested run support.

    Usage:
        logger = MLflowLogger("scheduler_comparison", tracking_uri="./mlruns")
        with logger.parent_run("convex_sgd"):
            with logger.child_run("none_seed42"):
                logger.log_params({"lr": 0.01, "scheduler": "none"})
                logger.log_metrics({"train_loss": 0.5, "accuracy": 0.85})
    """

    def __init__(self, experiment_name: str, tracking_uri: str = "./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def parent_run(self, run_name: str, tags: dict | None = None):
        """Start a parent run for an experiment group."""
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run

    @contextmanager
    def child_run(self, run_name: str, tags: dict | None = None):
        """Start a nested child run for a single configuration."""
        with mlflow.start_run(run_name=run_name, tags=tags or {}, nested=True) as run:
            yield run

    @staticmethod
    def log_params(params: dict) -> None:
        """Log a dict of parameters."""
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict, step: int | None = None) -> None:
        """Log a dict of metrics."""
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        """Log a file artifact."""
        mlflow.log_artifact(local_path, artifact_path)

    @staticmethod
    def log_figure(fig, filename: str) -> None:
        """Log a matplotlib figure as an image."""
        mlflow.log_figure(fig, filename)

    @staticmethod
    def log_history(history) -> None:
        """Log full training history as metrics."""
        for epoch_idx in range(history.num_epochs):
            mlflow.log_metrics(
                {
                    "train_loss": history.train_losses[epoch_idx],
                    "train_accuracy": history.train_accuracies[epoch_idx],
                    "lr": history.learning_rates[epoch_idx],
                },
                step=epoch_idx + 1,
            )

        if history.test_metrics:
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in history.test_metrics.items()
                 if isinstance(v, (int, float))}
            )

from collections import defaultdict
from stoc_opt_scheduler_comparison.evaluation.metrics import TrainingHistory 


def load_results_from_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
) -> dict[str, dict[str, tuple[TrainingHistory, dict]]]:
    """
    Reconstruct experiment results from MLflow child runs.

    Returns a nested dict mirroring the structure expected by
    compute_convergence_metrics:

        results[problem_type][run_name] = (TrainingHistory, test_metrics)

    where run_name follows the convention: "{problem_type}_{scheduler}_{optimizer}_seed{seed}"
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()

    # Retrieve experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Fetch all runs (child runs included)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time ASC"],
    )

    results: dict[str, dict] = defaultdict(dict)

    for run in runs:
        params = run.data.params

        # Skip parent runs (no problem_type param = orchestrator run)
        if "problem_type" not in params:
            continue

        problem_type  = params["problem_type"]       # "convex" | "non-convex"
        scheduler     = params["scheduler"]
        optimizer     = params["optimizer"]
        seed          = params["seed"]
        run_id        = run.info.run_id
        run_name      = f"{problem_type}_{scheduler}_{optimizer}_{seed}"

        # --- Reconstruct per-epoch metric history ---
        # MLflow stores metrics per step; fetch all metric history
        train_losses     = _fetch_metric_history(client, run_id, "train_loss")
        train_accuracies = _fetch_metric_history(client, run_id, "train_accuracy")
        learning_rates   = _fetch_metric_history(client, run_id, "lr")

        # --- Reconstruct test_metrics from final logged test_ metrics ---
        test_metrics = {
            k.removeprefix("test_"): v
            for k, v in run.data.metrics.items()
            if k.startswith("test_")
        }

        # --- Build TrainingHistory ---
        history = TrainingHistory(
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            learning_rates=learning_rates,
            test_metrics=test_metrics,
        )
        history.to_arrays()  # popola i numpy array

        results[problem_type][run_name] = history

    return dict(results)


def _fetch_metric_history(
    client: mlflow.MlflowClient,
    run_id: str,
    metric_key: str,
) -> list[float]:
    """
    Fetch all per-step values for a metric, sorted by step.
    Returns an empty list if the metric was never logged.
    """
    history = client.get_metric_history(run_id, metric_key)
    # Ordina per step (epoch_idx + 1 nel tuo log_history)
    history_sorted = sorted(history, key=lambda m: m.step)
    return [m.value for m in history_sorted]