"""Data loading and preprocessing utilities."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_transforms(augment: bool = False) -> transforms.Compose:
    """Get data transformations for MNIST.

    Args:
        augment: Whether to apply data augmentation (default: False)

    Returns:
        Composed transformation pipeline
    """
    transform_list = []

    if augment:
        transform_list.extend([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    return transforms.Compose(transform_list)


def load_data(
    data_dir: str = './data',
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
    augment: bool = False,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare MNIST dataset.

    Args:
        data_dir: Directory to save/load data
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading
        augment: Whether to apply data augmentation
        download: Whether to download the dataset if not found

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms(augment=augment)
    test_transform = get_transforms(augment=False)

    # Load datasets
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )

    # Split training data into train/validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def get_data_loaders(
    batch_size: int = 64,
    data_dir: str = './data',
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Get simple train and test data loaders (backward compatibility).

    Args:
        batch_size: Batch size for DataLoaders
        data_dir: Directory to save/load data
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader