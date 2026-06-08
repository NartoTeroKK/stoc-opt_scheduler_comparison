"""
Model architectures - LogisticRegression, MLP, SimpleCNN + create_model() factory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Logistic regression: single linear layer + sigmoid."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLP(nn.Module):
    """MLP with 2 hidden layers (256→128) + ReLU + Dropout. Designed for non-convex problems (e.g., MNIST)."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(
            128, output_dim
        )  # no activation — raw logits for CrossEntropyLoss
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform initialization — correct variance for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten (B, C, H, W) → (B, input_dim)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        return self.fc3(x)  # no dropout before output layer

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())



# ── Model Factory ──────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "logistic": LogisticRegression,
    "mlp": MLP,
}


def create_model(name: str, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
    """Create a model by name from the registry."""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available: {available}")

    model_cls = _MODEL_REGISTRY[name]

    return model_cls(input_dim=input_dim, output_dim=output_dim, **kwargs)
