"""Tests for model architectures."""

import pytest
import torch
import torch.nn as nn
from src.mnist_classifier.models import LinearClassifier, CNNClassifier


class TestLinearClassifier:
    """Test LinearClassifier model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = LinearClassifier()
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = LinearClassifier()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = model(input_tensor)

        assert output.shape == (batch_size, 10)

    def test_parameter_count(self):
        """Test correct number of parameters."""
        model = LinearClassifier()
        param_count = model.get_num_parameters()
        # 784 * 10 + 10 = 7850
        assert param_count == 7850

    def test_dropout(self):
        """Test model with dropout."""
        model = LinearClassifier(dropout_rate=0.5)
        model.train()
        input_tensor = torch.randn(10, 1, 28, 28)

        # Run multiple times to check dropout effect
        outputs = []
        for _ in range(2):
            outputs.append(model(input_tensor))

        # Outputs should be different in training mode with dropout
        assert not torch.allclose(outputs[0], outputs[1])


class TestCNNClassifier:
    """Test CNNClassifier model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = CNNClassifier()
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = CNNClassifier()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = model(input_tensor)

        assert output.shape == (batch_size, 10)

    def test_parameter_count(self):
        """Test model has reasonable parameter count."""
        model = CNNClassifier()
        param_count = model.get_num_parameters()
        # CNN should have more parameters than linear model
        assert param_count > 10000
        # But not too many (less than 1M for efficiency)
        assert param_count < 1000000

    def test_eval_mode(self):
        """Test model behavior in eval mode."""
        model = CNNClassifier(dropout_rate=0.5)
        model.eval()
        input_tensor = torch.randn(10, 1, 28, 28)

        # Run multiple times - should be deterministic in eval mode
        outputs = []
        for _ in range(2):
            with torch.no_grad():
                outputs.append(model(input_tensor))

        # Outputs should be identical in eval mode
        assert torch.allclose(outputs[0], outputs[1])


class TestModelComparison:
    """Test comparing different models."""

    def test_output_compatibility(self):
        """Test that all models produce compatible outputs."""
        linear_model = LinearClassifier()
        cnn_model = CNNClassifier()

        input_tensor = torch.randn(2, 1, 28, 28)

        linear_output = linear_model(input_tensor)
        cnn_output = cnn_model(input_tensor)

        # Both should output (batch_size, 10)
        assert linear_output.shape == cnn_output.shape == (2, 10)

    def test_model_efficiency(self):
        """Test that linear model is more parameter-efficient."""
        linear_model = LinearClassifier()
        cnn_model = CNNClassifier()

        linear_params = linear_model.get_num_parameters()
        cnn_params = cnn_model.get_num_parameters()

        # Linear model should have fewer parameters
        assert linear_params < cnn_params