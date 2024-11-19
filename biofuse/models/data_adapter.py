from .image_dataset import BioFuseImageDataset
import os
import glob
from typing import List, Union, Optional
import numpy as np

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
    def from_medmnist(cls, data) -> BioFuseImageDataset:
        """Create dataset from MedMNIST data object"""
        return BioFuseImageDataset(
            images=data.imgs,  # MedMNIST stores images in .imgs
            labels=data.labels,
            path=False
        )
    
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
