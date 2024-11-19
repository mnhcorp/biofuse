from .image_dataset import BioFuseImageDataset
import os
import glob
from typing import List, Union, Optional, Tuple
import numpy as np
import medmnist
from medmnist import INFO

class DataAdapter:
    """Adapter class for loading different dataset formats into BioFuseImageDataset"""
    
    @classmethod
    def from_imagenet(cls, path: str, split: str = 'train') -> Tuple[BioFuseImageDataset, int]:
        """Create dataset from ImageNet directory structure
        
        Args:
            path: Path to ImageNet directory for specific split
            split: One of 'train', 'val', or 'test'
            
        Returns:
            tuple: (BioFuseImageDataset, num_classes)
        """
        image_paths = []
        labels = []
        num_classes = 1000  # ImageNet has 1000 classes
        
        if split == 'train':
            # Hierarchical structure for training data
            class_dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            
            for class_idx, class_dir in enumerate(class_dirs):
                class_path = os.path.join(path, class_dir)
                for img_path in glob.glob(os.path.join(class_path, "*.JPEG")):
                    image_paths.append(img_path)
                    labels.append(class_idx)
        else:
            # Flat structure for val/test data
            image_paths = sorted(glob.glob(os.path.join(path, "*.JPEG")))
            
            # Load labels from mapping file
            mapping_file = os.path.join(os.path.dirname(path), f"imagenet_{split}_labels.txt")
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    labels = [int(line.strip()) for line in f]
            else:
                raise FileNotFoundError(f"Label mapping file not found: {mapping_file}")
                
        dataset = BioFuseImageDataset(
            images=image_paths,
            labels=labels,
            path=True,
            rgb=True  # ImageNet is RGB
        )
        
        return dataset, num_classes
    
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
