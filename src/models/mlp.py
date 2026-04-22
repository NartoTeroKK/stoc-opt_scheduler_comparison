import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron with 2-3 hidden layers and ReLU activation."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = 2):
        """
        Initialize the MLP.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of integers specifying the number of neurons in each hidden layer.
                         Length can be 2 or 3 for 2 or 3 hidden layers.
            output_dim: Number of output classes (default: 2 for binary classification).
        """
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)