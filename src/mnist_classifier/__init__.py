"""MNIST Classifier - A comprehensive ML engineering showcase."""

__version__ = "0.2.0"

from .models import LinearClassifier, CNNClassifier
from .utils import load_data, evaluate_model
from .train import train_model

__all__ = [
    "LinearClassifier",
    "CNNClassifier",
    "load_data",
    "evaluate_model",
    "train_model"
]