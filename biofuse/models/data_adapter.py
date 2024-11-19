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
    def from_imagenet(cls, path: str) -> BioFuseImageDataset:
        """Create dataset from ImageNet directory structure"""
        # ImageNet structure: root/class_folder/image.jpg
        image_paths = []
        labels = []
        
        for class_idx, class_dir in enumerate(sorted(os.listdir(path))):
            class_path = os.path.join(path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for img_path in glob.glob(os.path.join(class_path, "*.JPEG")):
                image_paths.append(img_path)
                labels.append(class_idx)
                
        return BioFuseImageDataset(
            images=image_paths,
            labels=labels,
            path=True,
            rgb=True  # ImageNet is RGB
        )
    
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
        
        return BioFuseImageDataset(
            images=data.imgs,  # MedMNIST stores images in .imgs
            labels=data.labels.squeeze() if hasattr(data.labels, 'squeeze') else data.labels,
            path=False,
            rgb=False  # MedMNIST images are grayscale
        ), num_classes
    
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
