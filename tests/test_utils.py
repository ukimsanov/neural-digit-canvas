"""Tests for utility functions."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.mnist_classifier.utils import load_data, evaluate_model
from src.mnist_classifier.utils.evaluation import calculate_per_class_metrics
from src.mnist_classifier.models import LinearClassifier


class TestDataLoading:
    """Test data loading utilities."""

    def test_load_data(self):
        """Test data loading function."""
        train_loader, val_loader, test_loader = load_data(
            batch_size=32,
            val_split=0.1,
            download=True
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Check batch size
        batch = next(iter(train_loader))
        images, labels = batch
        assert images.shape[0] <= 32
        assert images.shape[1:] == (1, 28, 28)
        assert labels.shape[0] <= 32

    def test_data_augmentation(self):
        """Test data augmentation option."""
        train_loader_no_aug, _, _ = load_data(
            batch_size=1,
            augment=False
        )

        train_loader_aug, _, _ = load_data(
            batch_size=1,
            augment=True
        )

        # Both should load successfully
        assert len(train_loader_no_aug) > 0
        assert len(train_loader_aug) > 0


class TestEvaluation:
    """Test evaluation utilities."""

    def test_evaluate_model(self):
        """Test model evaluation function."""
        # Create a simple model and data
        model = LinearClassifier()
        model.eval()

        # Create dummy data loader
        dummy_images = torch.randn(10, 1, 28, 28)
        dummy_labels = torch.randint(0, 10, (10,))
        dummy_dataset = list(zip(
            [img.unsqueeze(0) for img in dummy_images],
            [label.unsqueeze(0) for label in dummy_labels]
        ))

        class DummyLoader:
            def __iter__(self):
                for img, label in dummy_dataset:
                    yield img, label

            def __len__(self):
                return len(dummy_dataset)

        dummy_loader = DummyLoader()

        # Evaluate
        metrics = evaluate_model(model, dummy_loader)

        assert 'accuracy' in metrics
        assert 'correct' in metrics
        assert 'total' in metrics
        assert 'predictions' in metrics
        assert 'labels' in metrics

        assert metrics['total'] == 10
        assert 0 <= metrics['accuracy'] <= 100
        assert len(metrics['predictions']) == 10
        assert len(metrics['labels']) == 10

    def test_per_class_metrics(self):
        """Test per-class metrics calculation."""
        # Create sample predictions and labels
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Perfect predictions

        report = calculate_per_class_metrics(y_true, y_pred)

        assert isinstance(report, dict)
        assert 'accuracy' in report
        assert report['accuracy'] == 1.0

        # Check that all classes are present
        for i in range(10):
            assert str(i) in report
            assert report[str(i)]['precision'] == 1.0
            assert report[str(i)]['recall'] == 1.0
            assert report[str(i)]['f1-score'] == 1.0