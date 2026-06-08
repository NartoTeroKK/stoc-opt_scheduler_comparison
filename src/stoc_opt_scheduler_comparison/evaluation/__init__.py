"""Evaluation metrics module."""

from .metrics import (
    TrainingHistory,
    compute_accuracy,
    compute_confusion_matrix,
    compute_stability,
    evaluate_model,
)
from .convergence import (
    compute_convergence_metrics,
    aggregate_metrics,
    compute_all_metrics,
    compute_global_initial_loss,
    compute_target_loss,
    suboptimality_gap,
    epochs_to_target,
    area_under_loss,
    coefficient_of_variation,
    gradient_norm_statistics,
    ETT_EPSILON,
)

__all__ = [
    "TrainingHistory",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_stability",
    "evaluate_model",
    "compute_convergence_metrics",
    "aggregate_metrics",
    "compute_all_metrics",
    "compute_global_initial_loss",
    "compute_target_loss",
    "suboptimality_gap",
    "epochs_to_target",
    "area_under_loss",
    "coefficient_of_variation",
    "gradient_norm_statistics",
    "ETT_EPSILON",
]
