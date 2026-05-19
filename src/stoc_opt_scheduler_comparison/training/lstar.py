"""
Optimal loss (L*) computation for convergence analysis.

Two protocols:
- Convex (logistic regression): L-BFGS solver to find analytical global minimum.
- Non-convex (MLP on MNIST): Extended run of best config + LR decay -> reference empirical minimum.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from stoc_opt_scheduler_comparison.data_load.loaders import load_breast_cancer_wrapper, get_dataloaders
from stoc_opt_scheduler_comparison.models.architectures import create_model
from stoc_opt_scheduler_comparison.training.engine import train_one_epoch, evaluate
from stoc_opt_scheduler_comparison.training.optimizers import get_optimizer
from stoc_opt_scheduler_comparison.training.schedulers import get_scheduler
from stoc_opt_scheduler_comparison.evaluation.metrics import TrainingHistory


def compute_convex_L_star(
    test_size: float = 0.2,
    seed: int = 42,
) -> float:
    """
    Compute the exact global minimum loss (L*) for convex logistic regression
    on the Breast Cancer dataset using L-BFGS full-batch optimization.

    Uses sklearn's LogisticRegression with L-BFGS solver (deterministic,
    second-order quasi-Newton method) to find the unique global minimum.

    Args:
        test_size: Train/test split ratio.
        seed: Random seed for reproducibility.

    Returns:
        L_star: Cross-entropy loss at the global minimum.
    """
    ds = load_breast_cancer_wrapper(test_size=test_size, seed=seed)

    clf = LogisticRegression(
        solver='lbfgs',
        penalty=None,
        max_iter=10000,
        tol=1e-12,
        random_state=seed,
    )
    clf.fit(ds.X_train, ds.y_train)

    y_pred_proba = clf.predict_proba(ds.X_train)
    L_star = log_loss(ds.y_train, y_pred_proba)

    return float(L_star)


def _find_best_config(results_by_problem: dict) -> tuple[str, str, float]:
    """Find the (optimizer, scheduler) with the lowest final training loss."""
    best_loss = float("inf")
    best_opt = ""
    best_sched = ""

    for run_name, history in results_by_problem.items():
        if history is None or not history.train_losses:
            continue
        parts = run_name.split("_")
        if len(parts) != 4:
            continue
        _, sched, opt, _ = parts
        final_loss = history.train_losses[-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_opt = opt
            best_sched = sched

    return best_opt, best_sched, best_loss


def compute_nonconvex_L_star(
    results_by_problem: dict,
    exp_def,
    scheduler_params: dict,
    lr_config: dict | float,
    device: torch.device,
    extended_epochs: int = 200,
    lr_decay_factor: float = 0.5,
    lr_decay_interval: int = 10,
    seed: int = 42,
) -> float:
    """
    Compute the reference empirical minimum (L*) for the non-convex task.

    Protocol:
    1. Find best (optimizer, scheduler) config from initial runs.
    2. Run extended training for `extended_epochs` with LR decay in the second half.
    3. Return the minimum training loss achieved.

    Args:
        results_by_problem: Results dict from initial experiments.
        exp_def: ExperimentDef for the non-convex task.
        scheduler_params: All scheduler params from config.
        lr_config: LR value(s) from config (dict or float).
        device: torch device.
        extended_epochs: Total epochs for extended run.
        lr_decay_factor: Factor to multiply LR at each decay step.
        lr_decay_interval: Decay every N epochs during the decay phase.
        seed: Random seed for data loading.

    Returns:
        L_star_non_convex: Minimum training loss achieved.
    """
    best_opt, best_sched, _ = _find_best_config(results_by_problem)
    if not best_opt:
        raise ValueError("No valid runs found in results_by_problem")

    print(f"Best config for extended run: {best_opt} + {best_sched}")

    dataloaders = get_dataloaders(
        dataset_name=exp_def.dataset,
        batch_size=exp_def.batch_size,
        seed=seed,
        test_size=0.2,
    )

    model = create_model(
        name=exp_def.model,
        input_dim=dataloaders["n_features"],
        output_dim=dataloaders["n_classes"],
    )

    lr = lr_config.get(best_opt, 0.001) if isinstance(lr_config, dict) else lr_config
    optimizer = get_optimizer(model, name=best_opt, lr=lr)

    sched_params = dict(scheduler_params.get(best_sched, {}))
    if best_sched == "one-cycle":
        sched_params["total_steps"] = extended_epochs * len(dataloaders["train"])
        if best_opt == "adam":
            sched_params["cycle_momentum"] = False
    if best_sched == "cyclic":
        sched_params["step_size_up"] = 2 * len(dataloaders["train"])

    scheduler = get_scheduler(optimizer, name=best_sched, **sched_params)

    decay_epoch = extended_epochs // 2

    history = TrainingHistory()
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(extended_epochs):
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion,
            optimizer, scheduler, best_sched, device,
        )

        # Enforce aggressive LR decay in the second half
        if epoch >= decay_epoch and (epoch - decay_epoch) % lr_decay_interval == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor

        lr_val = float(optimizer.param_groups[0]["lr"])
        history.add_epoch(train_loss=train_loss, train_accuracy=train_acc, lr=lr_val)

        if best_sched not in {"cyclic", "one-cycle"}:
            scheduler.step()

    test_metrics = evaluate(model, dataloaders["test"], criterion, device)
    history.set_test_metrics(test_metrics)

    L_star = float(min(history.train_losses))
    print(f"Extended run complete ({extended_epochs} epochs). L* = {L_star:.6f}")

    return L_star
