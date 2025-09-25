"""Linear classifier model for MNIST."""

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Single-layer linear classifier for MNIST digits.

    Args:
        input_size: Number of input features (default: 784 for 28x28 images)
        num_classes: Number of output classes (default: 10 for digits 0-9)
        dropout_rate: Dropout probability for regularization (default: 0.0)
    """

    def __init__(self, input_size: int = 784, num_classes: int = 10, dropout_rate: float = 0.0):
        super(LinearClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(input_size, num_classes)

        # Initialize weights
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)