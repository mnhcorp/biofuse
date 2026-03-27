"""
Data loading utilities for BioFuse.

Provides functions to load various datasets (MedMNIST, ImageNet, custom)
with consistent interfaces and sensible defaults.
"""

import os
import glob
import random
from typing import Tuple, Optional, Union, List
from pathlib import Path
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .datasets import BioFuseImageDataset, ImageNetTestDataset


def _load_medmnist_backend():
    """Import medmnist lazily so the package can import without the dataset backend."""
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:
        raise ImportError(
            "MedMNIST support requires the optional `medmnist` package."
        ) from exc

    return medmnist, INFO


def load_medmnist(
    dataset_name: str,
    split: str = 'train',
    img_size: int = 224,
    root: str = '/data/medmnist',
    download: bool = True
) -> Tuple[BioFuseImageDataset, int]:
    """
    Load a MedMNIST dataset.

    Args:
        dataset_name: Name of MedMNIST dataset (e.g., 'pathmnist', 'chestmnist')
        split: Data split ('train', 'val', 'test')
        img_size: Target image size
        root: Root directory for dataset storage
        download: Whether to download if not present

    Returns:
        Tuple of (dataset, num_classes)

    Raises:
        ValueError: If dataset name is invalid

    Example:
        >>> dataset, num_classes = load_medmnist('pathmnist', split='train')
        >>> print(f"Loaded {len(dataset)} samples with {num_classes} classes")
    """
    medmnist, info_map = _load_medmnist_backend()

    if dataset_name not in info_map:
        available = ', '.join(info_map.keys())
        raise ValueError(
            f"Unknown MedMNIST dataset '{dataset_name}'. "
            f"Available: {available}"
        )

    # Get dataset info
    info = info_map[dataset_name]
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    # Resize and convert to tensors here; model-specific normalization happens later.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Load dataset
    medmnist_data = DataClass(
        split=split,
        download=download,
        transform=transform,
        root=root,
        size=img_size
    )

    # Extract images and labels
    images = medmnist_data.imgs
    labels = medmnist_data.labels
    if hasattr(labels, 'squeeze'):
        labels = labels.squeeze()

    # Wrap the raw arrays so the dataloader returns collatable tensors.
    dataset = BioFuseImageDataset(
        images=images,
        labels=labels,
        transform=transform,
        from_paths=False,
        target_mode='RGB',
        img_size=None,
    )

    return dataset, num_classes


def load_imagenet(
    root: str,
    split: str = 'val',
    batch_size: int = 32,
    num_workers: int = 4,
    subset_size: float = 1.0,
    img_size: int = 224
) -> Tuple[DataLoader, int]:
    """
    Load ImageNet dataset.

    Args:
        root: Path to ImageNet root directory
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        subset_size: Fraction of dataset to use (0-1)
        img_size: Target image size

    Returns:
        Tuple of (dataloader, num_classes)

    Example:
        >>> loader, num_classes = load_imagenet(
        ...     '/data/imagenet', split='val', subset_size=0.1
        ... )
    """
    # Keep tensors in raw image space; model-specific normalization happens later.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    # Load dataset
    if split == 'test':
        test_dir = os.path.join(root, 'test')
        dataset = ImageNetTestDataset(test_dir, transform=transform)
    else:
        dataset = ImageNet(root=root, split=split, transform=transform)

    # Create subset if requested
    if subset_size < 1.0:
        total_size = len(dataset)
        subset_count = int(total_size * subset_size)
        random.seed(42)
        indices = random.sample(range(total_size), subset_count)
        dataset = Subset(dataset, indices)

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return loader, 1000  # ImageNet has 1000 classes


def load_busi(
    root: str,
    split: str = 'train',
    img_size: int = 224,
    random_state: int = 42
) -> Tuple[BioFuseImageDataset, int]:
    """
    Load BUSI (Breast Ultrasound Images) dataset.

    Combines benign and normal as class 0, malignant as class 1.
    Performs stratified train/val/test split (49%/21%/30%).

    Args:
        root: Path to BUSI dataset directory
        split: Data split ('train', 'val', 'test')
        img_size: Target image size
        random_state: Random seed for splitting

    Returns:
        Tuple of (dataset, num_classes)

    Example:
        >>> dataset, num_classes = load_busi('/data/busi', split='train')
    """
    # Find image paths (exclude masks)
    benign_paths = [
        p for p in glob.glob(os.path.join(root, 'benign', '*.png'))
        if '_mask' not in p
    ]
    normal_paths = [
        p for p in glob.glob(os.path.join(root, 'normal', '*.png'))
        if '_mask' not in p
    ]
    malignant_paths = [
        p for p in glob.glob(os.path.join(root, 'malignant', '*.png'))
        if '_mask' not in p
    ]

    # Combine: benign + normal = 0, malignant = 1
    images = benign_paths + normal_paths + malignant_paths
    labels = [0] * (len(benign_paths) + len(normal_paths)) + [1] * len(malignant_paths)

    # Stratified split: 70% train, 15% val, 15% test
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels,
        test_size=0.3,
        random_state=random_state,
        stratify=labels
    )

    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_labels
    )

    # Select split
    split_map = {
        'train': (train_images, train_labels),
        'val': (val_images, val_labels),
        'test': (test_images, test_labels)
    }

    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")

    split_images, split_labels = split_map[split]

    # Keep tensors in raw image space; model-specific normalization happens later.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = BioFuseImageDataset(
        images=split_images,
        labels=split_labels,
        transform=transform,
        from_paths=True,
        target_mode='RGB',
        img_size=None,
    )

    return dataset, 2


def load_custom_directory(
    directory: Union[str, Path],
    img_size: int = 224,
    split: Optional[str] = None,
    class_subdirs: bool = True
) -> Tuple[List[str], List[int], int]:
    """
    Load images from a custom directory.

    Args:
        directory: Path to directory containing images
        img_size: Target image size
        split: Optional split name (for subdirectory organization)
        class_subdirs: If True, expects class subdirectories

    Returns:
        Tuple of (image_paths, labels, num_classes)

    Directory structures:
        With class subdirectories:
            directory/
                class_0/
                    img1.jpg
                    img2.jpg
                class_1/
                    img3.jpg

        Flat (no class subdirs):
            directory/
                0_img1.jpg
                1_img2.jpg

    Example:
        >>> paths, labels, num_classes = load_custom_directory(
        ...     '/data/my_dataset',
        ...     class_subdirs=True
        ... )
    """
    directory = Path(directory)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    image_paths = []
    labels = []

    # Check for subdirectories
    subdirs = [d for d in directory.iterdir() if d.is_dir()]

    if class_subdirs and subdirs:
        # Class-based subdirectory structure
        for class_idx, class_dir in enumerate(sorted(subdirs)):
            for ext in image_extensions:
                # Search for both lowercase and uppercase extensions
                for pattern in [f'*{ext}', f'*{ext.upper()}']:
                    class_images = list(class_dir.glob(pattern))
                    image_paths.extend([str(p) for p in class_images])
                    labels.extend([class_idx] * len(class_images))

        num_classes = len(subdirs)

    else:
        # Flat directory structure
        for ext in image_extensions:
            for pattern in [f'*{ext}', f'*{ext.upper()}']:
                found_images = list(directory.glob(pattern))
                image_paths.extend([str(p) for p in found_images])

                # Try to extract labels from filenames
                for img_path in found_images:
                    filename = img_path.stem
                    try:
                        # Try to parse label from filename (e.g., "0_img.jpg" -> 0)
                        if '_' in filename:
                            label = int(filename.split('_')[0])
                        else:
                            label = int(filename)
                        labels.append(label)
                    except ValueError:
                        # Default to 0 if can't parse
                        labels.append(0)

        num_classes = len(set(labels)) if labels else 0

    if not image_paths:
        raise ValueError(f"No images found in {directory}")

    return image_paths, labels, num_classes


def create_custom_dataset(
    images: Union[List[str], np.ndarray],
    labels: Union[List[int], np.ndarray],
    img_size: int = 224,
    from_paths: bool = True
) -> BioFuseImageDataset:
    """
    Create a custom BioFuse dataset from images and labels.

    Args:
        images: List of image paths or numpy array of images
        labels: Corresponding labels
        img_size: Target image size
        from_paths: Whether images are paths (True) or arrays (False)

    Returns:
        BioFuseImageDataset instance

    Example:
        >>> # From paths
        >>> paths = ['/path/img1.jpg', '/path/img2.jpg']
        >>> labels = [0, 1]
        >>> dataset = create_custom_dataset(paths, labels, from_paths=True)
        >>>
        >>> # From arrays
        >>> images = np.random.rand(100, 224, 224, 3)
        >>> labels = np.random.randint(0, 10, 100)
        >>> dataset = create_custom_dataset(images, labels, from_paths=False)
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return BioFuseImageDataset(
        images=images,
        labels=labels,
        transform=transform,
        from_paths=from_paths,
        target_mode='RGB',
        img_size=img_size
    )
