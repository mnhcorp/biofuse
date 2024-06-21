from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BioFuseImageDataset(Dataset):
    def __init__(self, images, labels, path=True, rgb=False):
        self.images = images
        self.labels = labels
        self.path = path
        self.rgb = rgb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if self.path:
            try:
                image = Image.open(img_path)
                if self.rgb:
                    image = image.convert('RGB')
            except (OSError, IOError) as e:
                self.logger.warning(f"Error loading image {img_path}: {e}")
                return None, label
        else:
            image = self.images[idx]
        
        label = self.labels[idx]
        return image, label
