"""Visualization utilities for model analysis."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def plot_training_history(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training history with loss and accuracy.

    Args:
        history: Dictionary containing training metrics
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None,
    num_images: int = 12,
    figsize: Tuple[int, int] = (12, 9)
) -> plt.Figure:
    """Visualize model predictions on sample images.

    Args:
        images: Tensor of images
        true_labels: True labels
        predicted_labels: Predicted labels
        probabilities: Prediction probabilities
        num_images: Number of images to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    num_images = min(num_images, len(images))
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel() if rows > 1 else [axes]

    for i in range(num_images):
        # Prepare image
        img = images[i].squeeze().cpu().numpy()

        # Display image
        axes[i].imshow(img, cmap='gray')

        # Prepare title
        true = true_labels[i].item()
        pred = predicted_labels[i].item()

        if true == pred:
            color = 'green'
            symbol = '✓'
        else:
            color = 'red'
            symbol = '✗'

        title = f'{symbol} True: {true}, Pred: {pred}'
        if probabilities is not None:
            conf = probabilities[i][pred].item()
            title += f'\n(Conf: {conf:.2%})'

        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Model Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_feature_embeddings(
    model: nn.Module,
    data_loader,
    device: Optional[torch.device] = None,
    num_samples: int = 5000,
    layer_name: Optional[str] = None,
    method: str = 'tsne',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Visualize learned feature embeddings using t-SNE or UMAP.

    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device to use
        num_samples: Number of samples to visualize
        layer_name: Name of layer to extract features from
        method: Dimensionality reduction method ('tsne' or 'umap')
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)

    features = []
    labels = []
    count = 0

    # Extract features
    with torch.no_grad():
        for images, batch_labels in data_loader:
            if count >= num_samples:
                break

            images = images.to(device)

            # Get features from specified layer or last layer before output
            if hasattr(model, 'flatten'):
                # For linear model
                features_batch = model.flatten(images)
            else:
                # For CNN - get features before final FC layer
                x = images
                for name, layer in model.named_children():
                    if name == 'fc2' or name == layer_name:
                        break
                    x = layer(x)
                    if name == 'fc1':
                        features_batch = x
                        break
                else:
                    features_batch = torch.flatten(x, 1)

            features.append(features_batch.cpu().numpy())
            labels.extend(batch_labels.numpy())

            count += len(batch_labels)

    # Combine features
    features = np.vstack(features)[:num_samples]
    labels = np.array(labels)[:num_samples]

    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each class with different color
    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=str(digit),
            alpha=0.6,
            s=10
        )

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'Feature Embeddings Visualization ({method.upper()})', fontsize=14)
    ax.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = ['accuracy', 'loss', 'parameters'],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot comparison of different models.

    Args:
        results: Dictionary with model names as keys and metrics as values
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    model_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    for idx, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in model_names]

        bars = axes[idx].bar(model_names, values, color=colors)
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylabel('Value')
        axes[idx].set_xlabel('Model')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if metric == 'accuracy':
                label = f'{value:.1f}%'
            elif metric == 'parameters':
                label = f'{int(value):,}'
            else:
                label = f'{value:.3f}'

            axes[idx].text(
                bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom'
            )

        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig