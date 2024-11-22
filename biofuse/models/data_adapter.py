from .image_dataset import BioFuseImageDataset
import os
from typing import List, Union, Optional, Tuple
import numpy as np
import medmnist
from medmnist import INFO
from torchvision.datasets import ImageNet, Dataset
import glob
from PIL import Image

class ImageNetTestDataset(Dataset):
    """Custom dataset for ImageNet test data with flat directory structure"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Get all JPEG files and sort them alphabetically
        self.image_paths = sorted(glob.glob(os.path.join(root, '*.JPEG')))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Return filename (without path) as label for test set
        return image, os.path.basename(image_path)

from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import random

class DataAdapter:
    """Adapter class for loading different dataset formats into BioFuseImageDataset"""
    
    @classmethod
    def from_imagenet(cls, root: str, split: str, batch_size: int = 32, num_workers: int = 1, subset_size: float = 1.0) -> Tuple[DataLoader, int]:
        """Create DataLoader from ImageNet directory structure
        
        Args:
            root: Path to ImageNet directory
            split: One of 'train', 'val', or 'test'
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            subset_size: Fraction of the dataset to use (default: 1.0)
            
        Returns:
            tuple: (DataLoader, num_classes)
        """
        # Define standard ImageNet transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load dataset based on split
        if split == 'test':
            test_dir = os.path.join(root, 'test')
            dataset = ImageNetTestDataset(test_dir, transform=transform)
        else:
            dataset = ImageNet(root=root, split=split, transform=transform)
        
        # Create subset if needed
        if subset_size < 1.0:
            total_size = len(dataset)
            subset_size = int(total_size * subset_size)
            
            # Use random seed for reproducibility
            random.seed(42)
            indices = random.sample(range(total_size), subset_size)
            dataset = Subset(dataset, indices)
        
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Shuffle only training data
            #num_workers=num_workers
        )
        
        return loader, 1000  # ImageNet has 1000 classes
    
    @classmethod
    def from_medmnist(cls, dataset_name: str, split: str, img_size: int, root: str = '/data/medmnist') -> Tuple[BioFuseImageDataset, int]:
        """Create dataset from MedMNIST dataset name
        
        Args:
            dataset_name: Name of the MedMNIST dataset (e.g. 'breastmnist')
            split: One of 'train', 'val', or 'test'
            img_size: Size of the images
            root: Root directory for dataset storage
            
        Returns:
            tuple: (BioFuseImageDataset, num_classes)
        """
        # Get dataset information and class
        info = INFO[dataset_name]
        num_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        
        # Load raw MedMNIST dataset
        data = DataClass(split=split, download=True, size=img_size, root=root)
        
        # Set up image paths based on image size
        save_dir = f'/data/medmnist/{dataset_name}_{split}'
        if img_size != 28:
            save_dir = f'{save_dir}/{dataset_name}_{img_size}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Create BioFuseImageDataset
        dataset = BioFuseImageDataset(
            images=data.imgs,
            labels=data.labels.squeeze() if hasattr(data.labels, 'squeeze') else data.labels,
            path=False,
            rgb=False  # MedMNIST images are grayscale
        )
        
        # Save the dataset if it doesn't exist
        if not os.path.exists(save_dir):
            dataset.save(save_dir)
            
        return dataset, num_classes
    
    @classmethod
    def from_custom(cls,
                   images: Union[List[str], np.ndarray],
                   labels: Union[List[int], np.ndarray],
                   dataset_type: str = 'path') -> BioFuseImageDataset:
        """Create dataset from custom image paths or arrays"""
        return BioFuseImageDataset(
            images=images,
            labels=labels,
            path=(dataset_type == 'path')
        )
