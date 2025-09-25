#!/usr/bin/env python3
"""Train MNIST classifier models."""

import argparse
import json
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mnist_classifier.models import LinearClassifier, CNNClassifier
from src.mnist_classifier.utils import load_data, evaluate_model, plot_confusion_matrix
from src.mnist_classifier.utils.visualization import plot_training_history, plot_model_comparison
from src.mnist_classifier.train import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--model', type=str, default='linear',
                        choices=['linear', 'cnn', 'both'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for regularization')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_data(
        batch_size=args.batch_size,
        val_split=0.1,
        augment=args.augment
    )

    results = {}

    # Train models
    models_to_train = []
    if args.model == 'linear':
        models_to_train = [('Linear', LinearClassifier(dropout_rate=args.dropout))]
    elif args.model == 'cnn':
        models_to_train = [('CNN', CNNClassifier(dropout_rate=args.dropout))]
    elif args.model == 'both':
        models_to_train = [
            ('Linear', LinearClassifier(dropout_rate=args.dropout)),
            ('CNN', CNNClassifier(dropout_rate=args.dropout))
        ]

    for model_name, model in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name} Classifier")
        print(f"{'='*60}")

        # Create model-specific output directory
        model_output_dir = output_dir / model_name.lower()
        model_output_dir.mkdir(exist_ok=True)

        # Initialize trainer
        trainer = Trainer(model, checkpoint_dir=model_output_dir / 'checkpoints')

        # Train model
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            optimizer_name=args.optimizer,
            early_stopping_patience=5
        )

        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set...")
        criterion = torch.nn.CrossEntropyLoss()
        metrics = evaluate_model(model, test_loader, criterion, trainer.device)

        print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Test Loss: {metrics['loss']:.4f}")

        # Save results
        results[model_name] = {
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'parameters': model.get_num_parameters(),
            'history': history
        }

        # Plot training history
        fig = plot_training_history(
            history,
            save_path=model_output_dir / 'training_history.png'
        )

        # Plot confusion matrix
        fig = plot_confusion_matrix(
            metrics['labels'],
            metrics['predictions'],
            save_path=model_output_dir / 'confusion_matrix.png'
        )

        # Save model
        model_path = model_output_dir / 'final_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Compare models if training both
    if args.model == 'both':
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}")

        comparison_metrics = {}
        for model_name, model_results in results.items():
            comparison_metrics[model_name] = {
                'accuracy': model_results['accuracy'],
                'loss': model_results['loss'],
                'parameters': model_results['parameters']
            }
            print(f"{model_name}: Acc={model_results['accuracy']:.2f}%, "
                  f"Params={model_results['parameters']:,}")

        # Plot comparison
        fig = plot_model_comparison(
            comparison_metrics,
            save_path=output_dir / 'model_comparison.png'
        )

    # Save results to JSON
    results_file = output_dir / 'results.json'
    # Convert history to serializable format
    for model_name in results:
        for key in results[model_name]['history']:
            results[model_name]['history'][key] = [
                float(x) for x in results[model_name]['history'][key]
            ]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {output_dir}")
    print("Training complete!")


if __name__ == '__main__':
    main()