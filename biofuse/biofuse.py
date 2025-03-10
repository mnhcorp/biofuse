import torch
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from .models.embedding_extractor import PreTrainedEmbedding
from .models.biofuse_model import BioFuseModel
from .models.data_adapter import DataAdapter

class BioFuse:
    """
    Main interface for BioFuse - a multi-modal fusion framework for biomedical images.
    """
    
    # Task types
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'
    
    # Dataset types
    MEDMNIST = 'medmnist'
    IMAGENET = 'imagenet'
    CUSTOM = 'custom'
    
    def __init__(self, models: List[str], fusion_method: str = 'concat', projection_dim: int = 512):
        """
        Initialize BioFuse with specified models.
        
        Args:
            models: List of model names to use for feature extraction
            fusion_method: Method to fuse embeddings ('concat', 'mean', etc.)
            projection_dim: Dimension for projection layers (0 for no projection)
        """
        self.model_names = models
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.models = [PreTrainedEmbedding(model_name) for model_name in models]
        self.biofuse_model = None
        
    def generate_embeddings(self, train_data, val_data=None, task_type=None, dataset_type=CUSTOM, 
                           batch_size=32, num_workers=4, img_size=224, dataset_name=None, root=None):
        """
        Generate embeddings for training and validation data and create a BioFuseModel.
        
        Args:
            train_data: Training dataset or path to training data
            val_data: Validation dataset or path to validation data
            task_type: Type of task (binary, multiclass, multilabel)
            dataset_type: Type of dataset ('medmnist', 'imagenet', 'custom')
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            img_size: Image size for resizing
            dataset_name: Name of the dataset (required for medmnist)
            root: Root directory for dataset storage
            
        Returns:
            train_embeddings: Embeddings for training data
            train_labels: Labels for training data
            val_embeddings: Embeddings for validation data (if provided)
            val_labels: Labels for validation data (if provided)
            biofuse_model: Trained BioFuseModel instance
        """
        # Create BioFuseModel
        self.biofuse_model = BioFuseModel(self.models, fusion_method=self.fusion_method, 
                                         projection_dim=self.projection_dim)
        
        # Process training data
        train_loader, num_classes = self._prepare_data(train_data, dataset_type, 'train', 
                                                     batch_size, num_workers, img_size, 
                                                     dataset_name, root)
        train_embeddings, train_labels = self._extract_features(train_loader)
        
        # Process validation data if provided
        val_embeddings = None
        val_labels = None
        if val_data is not None:
            val_loader, _ = self._prepare_data(val_data, dataset_type, 'val', 
                                             batch_size, num_workers, img_size, 
                                             dataset_name, root)
            val_embeddings, val_labels = self._extract_features(val_loader)
        
        return train_embeddings, train_labels, val_embeddings, val_labels, self.biofuse_model
    
    def embed(self, data, dataset_type=CUSTOM, split='test', batch_size=32, num_workers=4, 
             img_size=224, dataset_name=None, root=None):
        """
        Generate embeddings for new data using the existing BioFuseModel.
        
        Args:
            data: Dataset or path to data
            dataset_type: Type of dataset ('medmnist', 'imagenet', 'custom')
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            img_size: Image size for resizing
            dataset_name: Name of the dataset (required for medmnist)
            root: Root directory for dataset storage
            
        Returns:
            embeddings: Embeddings for the data
            labels: Labels for the data (if available)
        """
        if self.biofuse_model is None:
            raise ValueError("BioFuseModel not initialized. Call generate_embeddings first.")
        
        loader, _ = self._prepare_data(data, dataset_type, split, batch_size, num_workers, 
                                     img_size, dataset_name, root)
        embeddings, labels = self._extract_features(loader)
        return embeddings, labels
    
    def _prepare_data(self, data, dataset_type, split='train', batch_size=32, num_workers=4, 
                     img_size=224, dataset_name=None, root=None):
        """
        Prepare data for embedding extraction based on dataset type.
        
        Args:
            data: Dataset, path to data, or None (for medmnist/imagenet)
            dataset_type: Type of dataset ('medmnist', 'imagenet', 'custom')
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            img_size: Image size for resizing
            dataset_name: Name of the dataset (required for medmnist)
            root: Root directory for dataset storage
            
        Returns:
            data_loader: DataLoader for the data
            num_classes: Number of classes in the dataset
        """
        # Handle different dataset types
        if dataset_type == self.MEDMNIST:
            if dataset_name is None:
                raise ValueError("dataset_name is required for medmnist dataset type")
            
            root = root or '/data/medmnist'
            dataset, num_classes = DataAdapter.from_medmnist(
                dataset_name=dataset_name, 
                split=split, 
                img_size=img_size, 
                root=root
            )
            
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=(split == 'train'),
                num_workers=num_workers, pin_memory=True
            )
            return loader, num_classes
            
        elif dataset_type == self.IMAGENET:
            root = root or '/data/imagenet'
            return DataAdapter.from_imagenet(
                root=root, 
                split=split, 
                batch_size=batch_size, 
                num_workers=num_workers
            )
            
        else:  # CUSTOM
            # Handle different data types
            if isinstance(data, str):
                # Assume it's a path to a directory with images
                images, labels = DataAdapter.from_directory(data)
                dataset = DataAdapter.from_custom(images, labels)
                num_classes = len(set(labels))
            elif hasattr(data, '__iter__') and not hasattr(data, '__len__'):
                # It's already a dataloader
                return data, None  # Can't determine num_classes
            else:
                # It's a dataset, create a dataloader
                dataset = data
                # Try to determine num_classes
                if hasattr(dataset, 'classes'):
                    num_classes = len(dataset.classes)
                else:
                    num_classes = None
            
            # Create dataloader if needed
            if not (hasattr(data, '__iter__') and not hasattr(data, '__len__')):
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=(split == 'train'),
                    num_workers=num_workers, pin_memory=True
                )
                return loader, num_classes
            return data, None
    
    def _extract_features(self, dataloader):
        """
        Extract features from a dataloader using the BioFuseModel.
        
        Args:
            dataloader: DataLoader for the data
            
        Returns:
            embeddings: Extracted embeddings
            labels: Corresponding labels
        """
        all_embeddings = []
        all_labels = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.biofuse_model.to(device)
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images = batch
                    labels = None
                
                if isinstance(images, (list, tuple)):
                    images = images[0]  # Take the first element if it's a list
                
                images = images.to(device)
                
                # Process images through each model
                model_embeddings = []
                for model in self.models:
                    embedding = model(images)
                    model_embeddings.append(embedding)
                
                # Fuse embeddings using BioFuseModel
                fused_embedding = self.biofuse_model._ifusion(model_embeddings)
                all_embeddings.append(fused_embedding.cpu().numpy())
                
                if labels is not None:
                    all_labels.append(labels.cpu().numpy())
        
        # Concatenate all embeddings and labels
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
            self.biofuse_model = BioFuseModel(self.models, fusion_method=self.fusion_method, 
                                             projection_dim=self.projection_dim)
        self.biofuse_model.load_state_dict(torch.load(path))
        return self.biofuse_model
