"""Model architectures for MNIST classification."""

from .linear import LinearClassifier
from .cnn import CNNClassifier

__all__ = ["LinearClassifier", "CNNClassifier"]