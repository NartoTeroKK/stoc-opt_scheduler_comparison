import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Logistic regression model using a single linear layer."""

    def __init__(self, input_dim: int, output_dim: int = 2):
        """
        Initialize the logistic regression model.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output classes (default: 2 for binary classification).
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.linear(x)