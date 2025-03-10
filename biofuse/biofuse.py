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
        
    def generate_embeddings(self, train_data, val_data=None, task_type=None, batch_size=32, num_workers=4):
        """
        Generate embeddings for training and validation data and create a BioFuseModel.
        
        Args:
            train_data: Training dataset or path to training data
            val_data: Validation dataset or path to validation data
            task_type: Type of task (binary, multiclass, multilabel)
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            
        Returns:
            train_embeddings: Embeddings for training data
            val_embeddings: Embeddings for validation data (if provided)
            biofuse_model: Trained BioFuseModel instance
        """
        # Create BioFuseModel
        self.biofuse_model = BioFuseModel(self.models, fusion_method=self.fusion_method, 
                                         projection_dim=self.projection_dim)
        
        # Process training data
        train_loader = self._prepare_data(train_data, batch_size, num_workers)
        train_embeddings, train_labels = self._extract_features(train_loader)
        
        # Process validation data if provided
        val_embeddings = None
        val_labels = None
        if val_data is not None:
            val_loader = self._prepare_data(val_data, batch_size, num_workers)
            val_embeddings, val_labels = self._extract_features(val_loader)
        
        return train_embeddings, train_labels, val_embeddings, val_labels, self.biofuse_model
    
    def embed(self, data, batch_size=32, num_workers=4):
        """
        Generate embeddings for new data using the existing BioFuseModel.
        
        Args:
            data: Dataset or path to data
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            
        Returns:
            embeddings: Embeddings for the data
            labels: Labels for the data (if available)
        """
        if self.biofuse_model is None:
            raise ValueError("BioFuseModel not initialized. Call generate_embeddings first.")
        
        loader = self._prepare_data(data, batch_size, num_workers)
        embeddings, labels = self._extract_features(loader)
        return embeddings, labels
    
    def _prepare_data(self, data, batch_size=32, num_workers=4):
        """
        Prepare data for embedding extraction.
        
        Args:
            data: Dataset or path to data
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            
        Returns:
            data_loader: DataLoader for the data
        """
        # Handle different data types
        if isinstance(data, str):
            # Assume it's a path to a dataset
            # You'll need to implement logic to determine dataset type
            # For now, let's assume it's a directory with images
            return DataAdapter.from_directory(data, batch_size=batch_size, num_workers=num_workers)
        else:
            # Assume it's already a dataset or dataloader
            if hasattr(data, '__iter__') and not hasattr(data, '__len__'):
                # It's already a dataloader
                return data
            else:
                # It's a dataset, create a dataloader
                return torch.utils.data.DataLoader(
                    data, batch_size=batch_size, shuffle=False, 
                    num_workers=num_workers, pin_memory=True
                )
    
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
