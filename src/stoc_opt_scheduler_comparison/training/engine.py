"""
Training engine - train_one_epoch(), evaluate(), train_loop().

Device-agnostic, works with any nn.Module, optimizer, scheduler, and DataLoader.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from stoc_opt_scheduler_comparison.evaluation.metrics import TrainingHistory


PER_BATCH_SCHEDULERS = {"cyclic", "one-cycle"}

def _compute_grad_norm(model: nn.Module) -> float:
    """ℓ2-norm of the gradient across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    return total_norm ** 0.5


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: _LRScheduler,
    scheduler_name: str,
    device: torch.device,
    log_grad_norms: bool = False,
) -> tuple[float, float, float]:
    """
    Train model for one epoch.

    Handles both per-batch schedulers (cyclic, one-cycle) and per-epoch
    schedulers (exponential, cosine, none). For per-batch schedulers,
    scheduler.step() is called after every optimizer step inside the batch loop.

    Args:
        log_grad_norms: If True, computes and returns the mean ℓ2 gradient norm
                        across batches for the epoch.

    Returns:
        (average_loss, accuracy, mean_grad_norm) for the epoch.
        mean_grad_norm is NaN when log_grad_norms=False.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        if log_grad_norms:
            total_grad_norm += _compute_grad_norm(model)
            n_batches += 1

        optimizer.step()

        if scheduler_name in PER_BATCH_SCHEDULERS:
            scheduler.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    mean_grad_norm = (total_grad_norm / n_batches) if log_grad_norms and n_batches > 0 else float("nan")
    return total_loss / total, correct / total, mean_grad_norm


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate model on test set.

    Returns:
        {"loss": float, "accuracy": float, "predictions": np.ndarray, "labels": np.ndarray}
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    import numpy as np
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: _LRScheduler,
    scheduler_name: str,                
    epochs: int,
    device: torch.device,
    verbose: int = 10,
) -> TrainingHistory:
    """
    Full training loop.

    Args:
        model:          PyTorch model.
        train_loader:   Training DataLoader.
        test_loader:    Test DataLoader.
        optimizer:      PyTorch optimizer.
        scheduler:      LR scheduler instance (NoneScheduler for constant LR).
        scheduler_name: Scheduler identifier string used to dispatch per-batch
                        vs per-epoch step() calls. Must match PER_BATCH_SCHEDULERS.
        epochs:         Number of training epochs.
        device:         torch.device.
        verbose:        Print progress every N epochs (0 to disable).

    Returns:
        TrainingHistory with per-epoch metrics.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = TrainingHistory()

    for epoch in range(epochs):
        train_loss, train_acc, mean_grad_norm = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scheduler_name, device,
        )

        # LR registrato DOPO lo step — riflette il valore usato alla prossima epoca
        lr = float(scheduler.get_last_lr()[0])

        history.add_epoch(
            train_loss=train_loss,
            train_accuracy=train_acc,
            lr=lr,
            grad_norm=None if np.isnan(mean_grad_norm) else mean_grad_norm,
        )

        # Per-epoch step — skippato per scheduler per-batch (già steppati dentro train_one_epoch)
        if scheduler_name not in PER_BATCH_SCHEDULERS:
            scheduler.step()
            
        if verbose > 0 and (epoch + 1) % verbose == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}, lr={lr:.6f}")

    test_metrics = evaluate(model, test_loader, criterion, device)
    history.set_test_metrics(test_metrics)

    return history
