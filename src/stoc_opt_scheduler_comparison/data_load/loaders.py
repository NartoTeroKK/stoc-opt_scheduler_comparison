"""
Data loaders - dataset factories and DataLoader creation.

Datasets: synthetic (sklearn), breast_cancer (sklearn), mnist (torchvision).
Split: train/test only (no validation set).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split

from stoc_opt_scheduler_comparison.data_load.transforms import standard_normalize, mnist_normalize


@dataclass
class Dataset:
    """Container for a train/test split."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int

    def __repr__(self) -> str:
        return f"Dataset(train={self.X_train.shape}, test={self.X_test.shape}, classes={self.n_classes})"


def _split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> tuple:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return X_train, X_test, y_train, y_test


def load_synthetic(test_size: float, seed: int, n_samples: int = 50000, class_sep: float = 0.8) -> Dataset:
    """
    Synthetic binary classification dataset for convex problems (logistic regression).
    20 features (10 informative, 10 redundant), standardized.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=10,
        n_redundant=10, class_sep=class_sep, random_state=seed,
    )
    X_train, X_test, y_train, y_test = _split(X, y, test_size, seed)
    X_train, X_test = standard_normalize(X_train, X_test)

    return Dataset(
        X_train=X_train, X_test=X_test,
        y_train=y_train.astype(np.int64), y_test=y_test.astype(np.int64),
        n_features=X_train.shape[1], n_classes=int(len(np.unique(y_train))),
    )


def load_breast_cancer_wrapper(test_size: float, seed: int) -> Dataset:
    """
    Breast Cancer Wisconsin dataset (binary classification, 30 features).
    Standardized per-feature.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target # type: ignore
    X_train, X_test, y_train, y_test = _split(X, y, test_size, seed)
    X_train, X_test = standard_normalize(X_train, X_test)

    return Dataset(
        X_train=X_train, X_test=X_test,
        y_train=y_train.astype(np.int64), y_test=y_test.astype(np.int64),
        n_features=X_train.shape[1], n_classes=int(len(np.unique(y_train))),
    )


def load_mnist(test_size: float, seed: int) -> Dataset:
    """
    Full MNIST dataset (70K samples: 60K train + 10K test concatenated).
    Custom stratified split, normalized with canonical MNIST statistics.
    """
    from torchvision import datasets, transforms

    full_train = datasets.MNIST(root="../data/mnist", train=True, download=True, transform=transforms.ToTensor())
    official_test = datasets.MNIST(root="../data/mnist", train=False, download=True, transform=transforms.ToTensor())

    X = np.concatenate([
        full_train.data.numpy().reshape(-1, 784),
        official_test.data.numpy().reshape(-1, 784),
    ], axis=0).astype(np.float32) / 255.0

    y = np.concatenate([full_train.targets.numpy(), official_test.targets.numpy()], axis=0)

    X_train, X_test, y_train, y_test = _split(X, y, test_size, seed)
    X_train, X_test = mnist_normalize(X_train, X_test)

    return Dataset(
        X_train=X_train, X_test=X_test,
        y_train=y_train.astype(np.int64), y_test=y_test.astype(np.int64),
        n_features=X_train.shape[1], n_classes=int(len(np.unique(y_train))),
    )


# ── Dataset Factory ────────────────────────────────────────────────────────────────

_DATASET_REGISTRY: dict[str, Any] = {
    "synthetic": load_synthetic,
    "breast_cancer": load_breast_cancer_wrapper,
    "mnist": load_mnist,
}


def get_dataset(name: str, seed: int = 42, test_size: float = 0.2, **kwargs) -> Dataset:
    """Load a dataset by name."""
    if name not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
    return _DATASET_REGISTRY[name](test_size=test_size, seed=seed, **kwargs)


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    seed: int = 42,
    test_size: float = 0.2,
    **kwargs,
) -> dict:
    """
    Create train/test DataLoaders for a dataset.

    Returns:
        {
            "train": DataLoader,
            "test": DataLoader,
            "n_features": int,
            "n_classes": int,
        }
    """
    ds = get_dataset(dataset_name, seed=seed, test_size=test_size, **kwargs)

    return {
        "train": DataLoader(
            TensorDataset(torch.from_numpy(ds.X_train), torch.from_numpy(ds.y_train).long()),
            batch_size=batch_size, shuffle=True,
        ),
        "test": DataLoader(
            TensorDataset(torch.from_numpy(ds.X_test), torch.from_numpy(ds.y_test).long()),
            batch_size=batch_size, shuffle=False,
        ),
        "n_features": ds.n_features,
        "n_classes": ds.n_classes,
    }
