#!/usr/bin/env python3
"""Inference script for MNIST classifier."""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mnist_classifier.models import LinearClassifier, CNNClassifier
from torchvision import transforms


def load_model(model_path: str, model_type: str = 'linear') -> torch.nn.Module:
    """Load a trained model.

    Args:
        model_path: Path to model weights
        model_type: Type of model ('linear' or 'cnn')

    Returns:
        Loaded model
    """
    if model_type == 'linear':
        model = LinearClassifier()
    elif model_type == 'cnn':
        model = CNNClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(model_path, map_location='cpu')
    # Check if it's a full checkpoint or just state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess an image for inference.

    Args:
        image_path: Path to the image

    Returns:
        Preprocessed image tensor
    """
    # Load and convert to grayscale
    image = Image.open(image_path).convert('L')

    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Invert if necessary (MNIST has white digits on black background)
    image_array = np.array(image)
    if np.mean(image_array) > 127:
        image = ImageOps.invert(image)

    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    top_k: int = 3
) -> dict:
    """Make prediction on an image.

    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        top_k: Number of top predictions to return

    Returns:
        Dictionary with predictions
    """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)

    # Get top k predictions
    top_probs, top_classes = probabilities.topk(top_k, dim=1)

    predictions = {
        'predicted_class': top_classes[0][0].item(),
        'confidence': top_probs[0][0].item(),
        'top_k_predictions': [
            {'class': c.item(), 'probability': p.item()}
            for c, p in zip(top_classes[0], top_probs[0])
        ],
        'all_probabilities': probabilities[0].numpy()
    }

    return predictions


def visualize_prediction(
    image_path: str,
    predictions: dict,
    save_path: str = None
):
    """Visualize prediction results.

    Args:
        image_path: Path to original image
        predictions: Prediction results
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display image
    image = Image.open(image_path).convert('L')
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f"Predicted: {predictions['predicted_class']} "
                  f"(Confidence: {predictions['confidence']:.2%})")
    ax1.axis('off')

    # Display probability distribution
    classes = list(range(10))
    probs = predictions['all_probabilities']

    bars = ax2.bar(classes, probs)
    bars[predictions['predicted_class']].set_color('green')

    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel('Probability')
    ax2.set_title('Prediction Probabilities')
    ax2.set_xticks(classes)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}',
                ha='center', va='bottom' if prob > 0.05 else 'top',
                fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Run inference on MNIST classifier')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--model-type', type=str, default='linear',
                        choices=['linear', 'cnn'],
                        help='Type of model architecture')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--save-viz', type=str, default=None,
                        help='Path to save visualization')

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)

    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type)

    # Preprocess image
    print(f"Processing image: {args.image}")
    image_tensor = preprocess_image(args.image)

    # Make prediction
    predictions = predict(model, image_tensor, args.top_k)

    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Digit: {predictions['predicted_class']}")
    print(f"Confidence: {predictions['confidence']:.2%}")
    print(f"\nTop {args.top_k} Predictions:")
    for i, pred in enumerate(predictions['top_k_predictions'], 1):
        print(f"  {i}. Digit {pred['class']}: {pred['probability']:.2%}")

    # Visualize if requested
    if args.visualize or args.save_viz:
        visualize_prediction(args.image, predictions, args.save_viz)


if __name__ == '__main__':
    main()