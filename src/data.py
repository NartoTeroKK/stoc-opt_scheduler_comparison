import os
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.datasets import load_breast_cancer, make_classification
import numpy as np


def load_breast_cancer_data():
    """
    Load the Wisconsin Breast Cancer dataset.
    Returns a TensorDataset with features and target as torch tensors.
    """
    data = load_breast_cancer()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)
    return TensorDataset(X, y)


def generate_synthetic(seed: int):
    """
    Generate a synthetic binary classification dataset.
    Parameters:
        seed: random seed for reproducibility.
    Returns:
        TensorDataset with 10,000 samples, 20 features, 2 informative.
    """
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[0.5, 0.5],
        class_sep=1.0,
        hypercube=False,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=seed,
    )
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


def _split_dataset(dataset, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split a dataset into train, validation, test sets.
    Ratios: validation and test are taken from the remaining after train.
    Default split: 70% train, 15% validation, 15% test.
    """
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len
    # Ensure non-negative
    if train_len < 0:
        # Adjust ratios if too large
        train_len = max(1, total_len - val_len - test_len)
        # Recompute if needed
        if train_len + val_len + test_len > total_len:
            # Scale down proportionally
            scale = total_len / (val_len + test_len + train_len)
            train_len = int(train_len * scale)
            val_len = int(val_len * scale)
            test_len = total_len - train_len - val_len
    lengths = [train_len, val_len, test_len]
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, lengths, generator=generator)


def process_and_save_all(config, root='data'):
    """
    Load breast cancer and synthetic datasets, split them, and save as .pt files.
    Expects config dict with keys 'seed' and optionally 'augmentation' (ignored for tabular).
    Saves to <root>/processed/ as:
        breast_cancer_train.pt, breast_cancer_val.pt, breast_cancer_test.pt,
        synthetic_train.pt, synthetic_val.pt, synthetic_test.pt
    """
    seed = config.get('seed', 42)
    # augmentation ignored for tabular datasets
    processed_dir = os.path.join(root, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Load datasets
    breast_cancer = load_breast_cancer_data()
    synthetic = generate_synthetic(seed)

    # Split each dataset
    breast_cancer_splits = _split_dataset(breast_cancer, seed=seed)
    synthetic_splits = _split_dataset(synthetic, seed=seed)

    # Unpack
    breast_cancer_train, breast_cancer_val, breast_cancer_test = breast_cancer_splits
    synthetic_train, synthetic_val, synthetic_test = synthetic_splits

    # Save each split as a tuple of tensors (features, labels)
    def _save_split(dataset, prefix):
        # Since dataset is a Subset, we need to get the underlying tensors
        # However random_split returns Subset instances that reference the original dataset.
        # We can extract by iterating? Simpler: we can convert to TensorDataset by stacking?
        # Instead, we'll save the subset by saving indices? But easier: we can create new TensorDataset
        # by accessing the original dataset's tensors via indices.
        # However Subset does not expose indices directly in a simple way? Actually Subset has .indices attribute.
        indices = dataset.indices
        base_dataset = dataset.dataset
        X = base_dataset.tensors[0][indices]
        y = base_dataset.tensors[1][indices]
        torch.save((X, y), os.path.join(processed_dir, f'{prefix}.pt'))

    _save_split(breast_cancer_train, 'breast_cancer_train')
    _save_split(breast_cancer_val, 'breast_cancer_val')
    _save_split(breast_cancer_test, 'breast_cancer_test')
    _save_split(synthetic_train, 'synthetic_train')
    _save_split(synthetic_val, 'synthetic_val')
    _save_split(synthetic_test, 'synthetic_test')