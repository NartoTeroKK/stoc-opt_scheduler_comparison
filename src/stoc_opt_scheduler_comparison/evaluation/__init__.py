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
    compute_L_star_global,
    compute_global_initial_loss,
    compute_target_loss,
    compute_suboptimality_gap,
    compute_epochs_to_target,
    build_convergence_dataframe,
    compute_all_e_target_levels,
    compute_aggregated_config_metrics,
    E_TARGET_LEVELS,
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
    "compute_L_star_global",
    "compute_global_initial_loss",
    "compute_target_loss",
    "compute_suboptimality_gap",
    "compute_epochs_to_target",
    "build_convergence_dataframe",
    "compute_all_e_target_levels",
    "compute_aggregated_config_metrics",
    "E_TARGET_LEVELS",
]
