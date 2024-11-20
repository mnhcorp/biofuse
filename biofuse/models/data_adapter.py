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
    def from_imagenet(cls, path: str, split: str, val_size: int = 5000) -> Tuple[BioFuseImageDataset, int]:
        """Create dataset from ImageNet directory structure
        
        Args:
            path: Path to ImageNet directory (train or val)
            split: One of 'train', 'val', or 'test'
            val_size: Number of samples to use for validation set (default: 5000)
            
        Returns:
            tuple: (BioFuseImageDataset, num_classes)
        """
        image_paths = []
        labels = []
        
        # Count number of classes
        class_dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        num_classes = len(class_dirs)
        
        # If it's the training set, use all images from training directory
        if split == 'train':
            for class_idx, class_dir in enumerate(class_dirs):
                class_path = os.path.join(path, class_dir)
                for img_path in glob.glob(os.path.join(class_path, "*.JPEG")):
                    image_paths.append(img_path)
                    labels.append(class_idx)
        
        # If it's validation or test, work with validation directory
        elif split in ['val', 'test']:
            # First collect all validation images
            all_val_images = []
            all_val_labels = []
            for class_idx, class_dir in enumerate(class_dirs):
                class_path = os.path.join(path, class_dir)
                class_images = glob.glob(os.path.join(class_path, "*.JPEG"))
                all_val_images.extend(class_images)
                all_val_labels.extend([class_idx] * len(class_images))
            
            if split == 'val':
                # Create deterministic random state for reproducibility
                rng = np.random.RandomState(42)
                indices = np.arange(len(all_val_images))
                rng.shuffle(indices)
                
                # Take first val_size images for validation
                selected_indices = indices[:val_size]
                image_paths = [all_val_images[i] for i in selected_indices]
                labels = [all_val_labels[i] for i in selected_indices]
            
            else:  # test
                # Use all validation images for test
                image_paths = all_val_images
                labels = all_val_labels
        
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        
        return BioFuseImageDataset(
            images=image_paths,
            labels=labels,
            path=True,
            rgb=True  # ImageNet is RGB
        ), num_classes
    
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
