from torch.utils.data import Dataset
from PIL import Image

class BioFuseImageDataset(Dataset):
    def __init__(self, images, labels, path=True):
        self.images = images
        self.labels = labels
        self.path = path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if self.path:
            image = Image.open(img_path).convert("RGB")
        else:
            image = self.images[idx]
        
        label = self.labels[idx]
        return image, label
