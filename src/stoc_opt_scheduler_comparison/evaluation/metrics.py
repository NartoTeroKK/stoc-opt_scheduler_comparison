"""
Evaluation metrics - TrainingHistory, accuracy, confusion matrix, stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """
    Track per-epoch training metrics (stored as lists during training).

    Usage:
        history = TrainingHistory()
        history.add_epoch(train_loss=0.5, train_accuracy=0.85, lr=0.01)
        history.set_test_metrics({"loss": 0.4, "accuracy": 0.88})
        history.to_arrays()  # Convert to numpy arrays after training
    """
    train_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)

    # Numpy arrays (populated by to_arrays())
    _train_losses_arr: np.ndarray | None = field(default=None, repr=False)
    _train_accuracies_arr: np.ndarray | None = field(default=None, repr=False)
    _learning_rates_arr: np.ndarray | None = field(default=None, repr=False)

    def add_epoch(self, train_loss: float, train_accuracy: float, lr: float) -> None:
        """Record metrics for one epoch (list append - efficient)."""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
        self.learning_rates.append(lr)

    def set_test_metrics(self, metrics: dict) -> None:
        """Store final test evaluation results."""
        self.test_metrics = metrics

    def to_arrays(self) -> None:
        """Convert all list attributes to numpy arrays. Call once after training."""
        self._train_losses_arr = np.array(self.train_losses, dtype=float)
        self._train_accuracies_arr = np.array(self.train_accuracies, dtype=float)
        self._learning_rates_arr = np.array(self.learning_rates, dtype=float)

    @property
    def train_losses_arr(self) -> np.ndarray:
        if self._train_losses_arr is None:
            self.to_arrays()
        assert self._train_losses_arr is not None  # narrowing esplicito
        return self._train_losses_arr

    @property
    def train_accuracies_arr(self) -> np.ndarray:
        if self._train_accuracies_arr is None:
            self.to_arrays()
        assert self._train_accuracies_arr is not None
        return self._train_accuracies_arr

    @property
    def learning_rates_arr(self) -> np.ndarray:
        if self._learning_rates_arr is None:
            self.to_arrays()
        assert self._learning_rates_arr is not None
        return self._learning_rates_arr
    
    @property
    def num_epochs(self) -> int:
        return len(self.train_losses)

    def __repr__(self) -> str:
        return f"TrainingHistory(epochs={self.num_epochs}, test_acc={self.test_metrics.get('accuracy', 0):.4f})"


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion matrix (supports multi-class)."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cm[i, j] = int(np.sum((y_true == true_cls) & (y_pred == pred_cls)))
    return cm


def compute_stability(values: list[float] | np.ndarray) -> dict:
    """Compute stability metrics (mean, std, variance, min, max)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "variance": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a DataLoader.

    Returns:
        {"loss": float, "accuracy": float, "confusion_matrix": np.ndarray}
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    n_samples = len(all_labels)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "loss": total_loss / n_samples,
        "accuracy": correct / total,
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
    }
