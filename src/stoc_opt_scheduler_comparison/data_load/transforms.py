"""
Data transforms - normalization and preprocessing utilities.

Fit parameters on train only — never on test (no data leakage).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler


# MNIST channel-wise statistics (computed on official 60K train set, /255)
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081


def standard_normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-feature standardization (zero-mean, unit-variance).

    Fits scaler on train only, transforms both train and test.
    """
    scaler = _SklearnStandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)


def mnist_normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Global channel normalization with canonical MNIST statistics.

    Expects input already in [0, 1] via /255.
    """
    X_train_norm = (X_train - MNIST_MEAN) / MNIST_STD
    X_test_norm = (X_test - MNIST_MEAN) / MNIST_STD
    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)
