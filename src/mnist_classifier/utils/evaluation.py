"""Model evaluation utilities."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Evaluate model on given data.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function (optional)
        device: Device to run evaluation on

    Returns:
        Dictionary containing accuracy, loss, and other metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)

    correct = 0
    total = 0
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(data_loader) if criterion else 0.0

    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure object
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add accuracy for each class
    accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(accuracies):
        ax.text(
            len(class_names) + 0.5, i + 0.5,
            f'{acc:.2%}',
            ha='left', va='center',
            fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def get_misclassified_examples(
    model: nn.Module,
    data_loader: DataLoader,
    num_examples: int = 10,
    device: Optional[torch.device] = None
) -> List[Dict]:
    """Get examples of misclassified images.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        num_examples: Number of misclassified examples to return
        device: Device to run on

    Returns:
        List of dictionaries containing misclassified examples
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)

    misclassified = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Find misclassified
            mask = predicted != labels
            if mask.any():
                wrong_images = images[mask].cpu()
                wrong_labels = labels[mask].cpu()
                wrong_preds = predicted[mask].cpu()
                wrong_probs = probabilities[mask].cpu()

                for i in range(min(len(wrong_images), num_examples - len(misclassified))):
                    misclassified.append({
                        'image': wrong_images[i],
                        'true_label': wrong_labels[i].item(),
                        'predicted_label': wrong_preds[i].item(),
                        'confidence': wrong_probs[i][wrong_preds[i]].item(),
                        'probabilities': wrong_probs[i].numpy()
                    })

                if len(misclassified) >= num_examples:
                    break

    return misclassified[:num_examples]


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """Calculate per-class precision, recall, and F1 scores.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class

    Returns:
        Dictionary containing classification report and metrics
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    return report