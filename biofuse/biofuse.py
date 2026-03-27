from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    create_custom_dataset,
    load_custom_directory,
    load_imagenet,
    load_medmnist,
)
from .models.biofuse_model import BioFuseModel
from .models.embedding_extractor import PreTrainedEmbedding
from .utils.reproducibility import get_device


class BioFuse:
    """
    Main interface for BioFuse - a multi-modal fusion framework for biomedical images.
    """

    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'

    MEDMNIST = 'medmnist'
    IMAGENET = 'imagenet'
    CUSTOM = 'custom'

    def __init__(
        self,
        models: List[str],
        fusion_method: str = 'concat',
        projection_dim: int = 512,
        device: Optional[str] = None,
    ):
        self.model_names = models
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.device = (
            device
            if isinstance(device, torch.device)
            else get_device(device)
        )
        self.models = [
            PreTrainedEmbedding(model_name, device=self.device)
            for model_name in models
        ]
        self.biofuse_model = None

    def generate_embeddings(
        self,
        train_data,
        val_data=None,
        task_type=None,
        dataset_type=CUSTOM,
        batch_size=32,
        num_workers=4,
        img_size=224,
        dataset_name=None,
        root=None,
    ):
        """
        Generate embeddings for training and validation data and create a BioFuseModel.
        """
        self.biofuse_model = BioFuseModel(
            self.model_names,
            fusion_method=self.fusion_method,
            projection_dim=self.projection_dim,
        ).to(self.device)
        self.biofuse_model.eval()

        train_loader, _ = self._prepare_data(
            train_data,
            dataset_type,
            'train',
            batch_size,
            num_workers,
            img_size,
            dataset_name,
            root,
        )
        train_embeddings, train_labels = self._extract_features(train_loader)

        val_embeddings = None
        val_labels = None
        if val_data is not None:
            val_loader, _ = self._prepare_data(
                val_data,
                dataset_type,
                'val',
                batch_size,
                num_workers,
                img_size,
                dataset_name,
                root,
            )
            val_embeddings, val_labels = self._extract_features(val_loader)

        return train_embeddings, train_labels, val_embeddings, val_labels, self.biofuse_model

    def embed(
        self,
        data,
        dataset_type=CUSTOM,
        split='test',
        batch_size=32,
        num_workers=4,
        img_size=224,
        dataset_name=None,
        root=None,
    ):
        """
        Generate embeddings for new data using the existing BioFuseModel.
        """
        if self.biofuse_model is None:
            raise ValueError("BioFuseModel not initialized. Call generate_embeddings first.")

        loader, _ = self._prepare_data(
            data,
            dataset_type,
            split,
            batch_size,
            num_workers,
            img_size,
            dataset_name,
            root,
        )
        return self._extract_features(loader)

    def _prepare_data(
        self,
        data,
        dataset_type,
        split='train',
        batch_size=32,
        num_workers=4,
        img_size=224,
        dataset_name=None,
        root=None,
    ):
        """Prepare data for embedding extraction based on dataset type."""
        if isinstance(data, DataLoader):
            return data, None

        if dataset_type == self.MEDMNIST:
            if dataset_name is None:
                raise ValueError("dataset_name is required for medmnist dataset type")

            dataset, num_classes = load_medmnist(
                dataset_name,
                split=split,
                img_size=img_size,
                root=root or '/data/medmnist',
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=self.device.type == 'cuda',
            )
            return loader, num_classes

        if dataset_type == self.IMAGENET:
            return load_imagenet(
                root=root or '/data/imagenet',
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                img_size=img_size,
            )

        if isinstance(data, str):
            image_paths, labels, num_classes = load_custom_directory(
                data,
                img_size=img_size,
            )
            dataset = create_custom_dataset(
                image_paths,
                labels,
                img_size=img_size,
                from_paths=True,
            )
        else:
            dataset = data
            num_classes = len(getattr(dataset, 'classes', [])) or None

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
        )
        return loader, num_classes

    def _extract_features(self, dataloader):
        """Extract features from a dataloader using the shared fusion path."""
        if self.biofuse_model is None:
            raise ValueError("BioFuseModel not initialized. Call generate_embeddings first.")

        all_embeddings = []
        all_labels = []

        self.biofuse_model.to(self.device)
        self.biofuse_model.eval()

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images = batch
                    labels = None

                model_embeddings = [model(images) for model in self.models]
                fused_embedding = self.biofuse_model(model_embeddings)
                all_embeddings.append(fused_embedding.detach().cpu().numpy())

                if labels is not None:
                    if torch.is_tensor(labels):
                        all_labels.append(labels.detach().cpu().numpy())
                    else:
                        all_labels.append(np.asarray(labels))

        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels) if all_labels else None

        return embeddings, labels

    def save(self, path):
        """Save the BioFuseModel to a file."""
        if self.biofuse_model is None:
            raise ValueError("No BioFuseModel to save. Call generate_embeddings first.")
        torch.save(self.biofuse_model.state_dict(), path)

    def load(self, path):
        """Load a BioFuseModel from a file."""
        if self.biofuse_model is None:
            self.biofuse_model = BioFuseModel(
                self.model_names,
                fusion_method=self.fusion_method,
                projection_dim=self.projection_dim,
            ).to(self.device)
        self.biofuse_model.load_state_dict(torch.load(path, map_location=self.device))
        self.biofuse_model.eval()
        return self.biofuse_model
