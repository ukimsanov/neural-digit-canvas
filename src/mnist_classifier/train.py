"""Training utilities and functions."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import time
from pathlib import Path


class Trainer:
    """Trainer class for MNIST models."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        learning_rate: float = 0.001,
        optimizer_name: str = 'adam',
        scheduler_config: Optional[Dict] = None,
        early_stopping_patience: int = 5
    ) -> Dict[str, List]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            learning_rate: Learning rate
            optimizer_name: Name of optimizer ('sgd', 'adam', 'adamw')
            scheduler_config: Configuration for learning rate scheduler
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dictionary
        """
        # Setup optimizer
        if optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Setup loss function
        criterion = nn.CrossEntropyLoss()

        # Setup scheduler
        scheduler = None
        if scheduler_config:
            if scheduler_config['type'] == 'step':
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get('step_size', 10),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_config['type'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs
                )

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Optimizer: {optimizer_name}, LR: {learning_rate}")
        print("=" * 50)

        # Training loop
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.checkpoint_dir:
                        self.save_checkpoint(epoch, optimizer, best=True)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
            else:
                val_loss, val_acc = 0, 0

            # Update scheduler
            if scheduler:
                scheduler.step()

            # Track time
            epoch_time = time.time() - start_time
            self.history['epoch_times'].append(epoch_time)

            # Print progress
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")

            # Save checkpoint
            if self.checkpoint_dir and epoch % 5 == 0:
                self.save_checkpoint(epoch, optimizer)

        print("=" * 50)
        print("Training completed!")

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        best: bool = False
    ):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            optimizer: Current optimizer
            best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history
        }

        if best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 0.01,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict]:
    """Simple training function for backward compatibility.

    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Tuple of (trained_model, history)
    """
    trainer = Trainer(model, device)
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer_name='sgd'
    )

    return model, history