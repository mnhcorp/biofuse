"""
Dataset classes for BioFuse.

Provides dataset wrappers for different data formats (paths, arrays, etc.).
"""

import logging
from typing import Union, List, Optional, Callable
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BioFuseImageDataset(Dataset):
    """
    Dataset class for loading and processing images for BioFuse.

    Supports both image paths and pre-loaded numpy arrays with flexible
    preprocessing options.
    """

    def __init__(
        self,
        images: Union[List[str], List[Path], np.ndarray],
        labels: Union[List[int], np.ndarray],
        transform: Optional[Callable] = None,
        from_paths: bool = True,
        target_mode: str = 'RGB',
        img_size: Optional[int] = None
    ):
        """
        Initialize image dataset.

        Args:
            images: List of image paths or numpy array of images
            labels: Corresponding labels
            transform: Optional transform to apply to images
            from_paths: If True, images contains paths. If False, contains arrays
            target_mode: Image mode to convert to ('RGB', 'L', etc.)
            img_size: If provided, resize images to this size

        Example:
            >>> # From paths
            >>> dataset = BioFuseImageDataset(
            ...     images=['/path/img1.jpg', '/path/img2.jpg'],
            ...     labels=[0, 1],
            ...     from_paths=True
            ... )
            >>> # From arrays
            >>> dataset = BioFuseImageDataset(
            ...     images=np.random.rand(100, 28, 28),
            ...     labels=np.random.randint(0, 10, 100),
            ...     from_paths=False
            ... )
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.from_paths = from_paths
        self.target_mode = target_mode
        self.img_size = img_size

        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Get item at index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        label = self.labels[idx]

        try:
            if self.from_paths:
                # Load from path
                img_path = self.images[idx]
                image = Image.open(img_path)
            else:
                # Load from numpy array
                img_array = self.images[idx]

                # Handle different array shapes
                if len(img_array.shape) == 2:
                    # Grayscale
                    image = Image.fromarray(img_array)
                elif len(img_array.shape) == 3:
                    # RGB or multi-channel
                    if img_array.shape[0] in [1, 3]:
                        # Channel-first format [C, H, W]
                        img_array = np.transpose(img_array, (1, 2, 0))
                    image = Image.fromarray(img_array.astype('uint8'))
                else:
                    raise ValueError(f"Unexpected image shape: {img_array.shape}")

            # Convert to target mode
            if self.target_mode:
                image = image.convert(self.target_mode)

            # Resize if requested
            if self.img_size:
                image = image.resize(
                    (self.img_size, self.img_size),
                    Image.Resampling.BILINEAR
                )

            # Apply transform
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            self.logger.error(f"Error loading image at index {idx}: {e}")
            # Return a dummy tensor to avoid breaking the dataloader
            if self.transform:
                # Try to infer output shape from transform
                dummy = torch.zeros(3, self.img_size or 224, self.img_size or 224)
            else:
                dummy = Image.new('RGB', (self.img_size or 224, self.img_size or 224))
            return dummy, label


class ImageNetTestDataset(Dataset):
    """
    Custom dataset for ImageNet test data with flat directory structure.

    Useful for evaluating on ImageNet test sets where labels may not be available.
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: tuple = ('.JPEG', '.jpg', '.jpeg', '.png')
    ):
        """
        Initialize ImageNet test dataset.

        Args:
            root: Directory containing test images
            transform: Optional transform to apply
            extensions: Tuple of valid file extensions
        """
        self.root = Path(root)
        self.transform = transform

        # Find all images
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(sorted(self.root.glob(f'*{ext}')))

        if not self.image_paths:
            raise ValueError(f"No images found in {root} with extensions {extensions}")

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Get image at index.

        Args:
            idx: Image index

        Returns:
            Tuple of (image, filename)
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return filename as "label" for test set
        return image, image_path.name
