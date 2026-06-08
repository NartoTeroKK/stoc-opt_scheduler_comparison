"""
Optimal loss (L*) computation for convergence analysis.

Two protocols:
- Convex (logistic regression): L-BFGS solver to find analytical global minimum.
- Non-convex (MLP on MNIST): Extended run of best config + LR decay -> reference empirical minimum.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from stoc_opt_scheduler_comparison.data_load.loaders import (
    load_breast_cancer_wrapper,
    get_dataloaders,
)
from stoc_opt_scheduler_comparison.models.architectures import create_model
from stoc_opt_scheduler_comparison.training.engine import train_one_epoch, evaluate
from stoc_opt_scheduler_comparison.training.optimizers import get_optimizer
from stoc_opt_scheduler_comparison.training.schedulers import (
    get_dynamic_scheduler_params,
    get_scheduler,
)
from stoc_opt_scheduler_comparison.evaluation.metrics import TrainingHistory


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


def compute_empirical_L_star(
    results_by_problem: dict,
    exp_def,
    scheduler_params: dict,
    lr_config: dict | float,
    device: torch.device,
    extended_epochs: int = 200,
    use_lr_decay: bool = True,
    lr_decay_factor: float = 0.5,
    lr_decay_interval: int = 10,
    seed: int = 42,
) -> float:
    """
    Compute the reference empirical minimum (L*) for the non-convex task.

    Protocol:
    1. Find best (optimizer, scheduler) config from initial runs.
    2. Run extended training for `extended_epochs` with optional LR decay in the second half.
    3. Return the minimum training loss achieved.

    Args:
        results_by_problem: Results dict from initial experiments.
        exp_def: ExperimentDef for the non-convex task.
        scheduler_params: All scheduler params from config.
        lr_config: LR value(s) from config (dict or float).
        device: torch device.
        extended_epochs: Total epochs for extended run.
        use_lr_decay: If True, applies an aggressive manual LR decay in the second half.
        lr_decay_factor: Factor to multiply LR at each decay step.
        lr_decay_interval: Decay every N epochs during the decay phase.
        seed: Random seed for data loading.

    Returns:
        L_star_non_convex: Minimum training loss achieved.
    """
    # ─── Determinism: seed all global generators ─────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # ──────────────────────────────────────────────────────────────────────────

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

    # Ottieni la lunghezza del dataloader (steps per epoch)
    steps = len(dataloaders["train"])

    # Richiami la funzione
    sched_params = get_dynamic_scheduler_params(
        scheduler_name=best_sched,
        lr=lr,
        epochs=extended_epochs,
        steps_per_epoch=steps,
        base_sched_params=sched_params,
    )

    scheduler = get_scheduler(optimizer, name=best_sched, **sched_params)

    decay_epoch = extended_epochs // 2

    history = TrainingHistory()
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(extended_epochs):
        train_loss, train_acc, _ = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scheduler,
            best_sched,
            device,
        )

        # ─── LOGICA DI DECAY CONDIZIONALE ──────────────────────────────────────
        if (
            use_lr_decay
            and epoch >= decay_epoch
            and (epoch - decay_epoch) % lr_decay_interval == 0
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor
        # ───────────────────────────────────────────────────────────────────────

        lr_val = float(optimizer.param_groups[0]["lr"])
        history.add_epoch(train_loss=train_loss, train_accuracy=train_acc, lr=lr_val)

        if best_sched not in {"cyclic", "one-cycle"}:
            scheduler.step()

    test_metrics = evaluate(model, dataloaders["test"], criterion, device)
    history.set_test_metrics(test_metrics)

    L_star = float(min(history.train_losses))
    print(f"Extended run complete ({extended_epochs} epochs). L* = {L_star:.6f}")

    return L_star
