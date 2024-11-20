import logging
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
from typing import Union, List

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BioFuseImageDataset(Dataset):
    """Dataset class for loading and processing images for BioFuse"""
    
    def __init__(self, 
                 images: Union[List[str], np.ndarray],
                 labels: Union[List[int], np.ndarray],
                 path: bool = True,
                 rgb: bool = False,
                 resize: bool = False,
                 img_size: int = 224):
        """
        Args:
            images: List of image paths or numpy array of images
            labels: List/array of corresponding labels
            path: If True, images contains paths. If False, contains image arrays
            rgb: If True, convert images to RGB mode
            resize: If True, resize images to img_size
            img_size: Target size for image resizing if resize is True
        """
        self.images = images
        self.labels = labels
        self.path = path
        self.rgb = rgb
        self.resize = resize
        self.img_size = img_size
        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        if self.path:
            img_path = self.images[idx]
            try:
                image = Image.open(img_path)
                if self.rgb:
                    image = image.convert('RGB')
                if self.resize:
                    image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            except (OSError, IOError) as e:
                self.logger.warning(f"Error loading image {img_path}: {e}")
                return None, self.labels[idx]
        else:
            # Handle numpy array images
            image = Image.fromarray(self.images[idx])
            if self.rgb:
                image = image.convert('RGB')
        
        return image, self.labels[idx]
