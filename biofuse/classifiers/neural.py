"""
Neural network classifiers for BioFuse.

Includes various neural architectures (MLP, CNN, ResNet) for classification
tasks with foundation model embeddings.
"""

import time
from typing import Optional, Dict, Any, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .base import BaseClassifier, ClassifierFactory


# Neural Network Architectures
class MLPArchitecture(nn.Module):
    """Two-layer MLP with dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class CNNArchitecture(nn.Module):
    """1D CNN for feature processing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Unflatten(1, (2, input_dim)),
            nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(hidden_dim // 2),
            nn.Flatten(),
            nn.Linear(hidden_dim * (hidden_dim // 2), output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ResidualArchitecture(nn.Module):
    """Residual network with skip connections."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = self.dropout(F.relu(self.fc2(h1))) + h1  # Skip connection
        return self.fc3(h2)


class NeuralNetClassifier(BaseClassifier):
    """
    Neural network classifier with multiple architecture options.

    Supports different architectures (MLP, CNN, ResNet) for classification
    of foundation model embeddings.
    """

    ARCHITECTURES = {
        'mlp': MLPArchitecture,
        'cnn': CNNArchitecture,
        'resnet': ResidualArchitecture,
    }

    def __init__(
        self,
        architecture: Literal['mlp', 'cnn', 'resnet'] = 'mlp',
        hidden_dim: int = 256,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        num_epochs: int = 100,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: int = 10,
        **kwargs
    ):
        """
        Initialize neural network classifier.

        Args:
            architecture: Neural architecture to use ('mlp', 'cnn', 'resnet')
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to use ('cuda', 'cpu', or None for auto)
            random_state: Random seed
            verbose: Print loss every N epochs (0 for no printing)
            **kwargs: Additional parameters
        """
        super().__init__()

        if architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"Architecture must be one of {list(self.ARCHITECTURES.keys())}"
            )

        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.scaler = StandardScaler()
        self.model = None
        self.training_time = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'NeuralNetClassifier':
        """Train the neural network classifier."""
        start_time = time.time()

        # Set random seed
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Determine task type
        self.num_classes = len(np.unique(y))
        self.multi_label = len(y.shape) > 1 and y.shape[1] > 1

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        if self.num_classes == 2 and not self.multi_label:
            y_tensor = torch.FloatTensor(y).to(self.device)
            output_dim = 1
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_tensor = torch.LongTensor(y).squeeze().to(self.device)
            output_dim = self.num_classes
            criterion = nn.CrossEntropyLoss()

        # Create model
        architecture_class = self.ARCHITECTURES[self.architecture]
        self.model = architecture_class(
            input_dim=X_scaled.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            dropout=self.dropout
        ).to(self.device)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        n_samples = len(X_tensor)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            # Shuffle indices
            indices = torch.randperm(n_samples)

            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Handle binary case
                if output_dim == 1:
                    outputs = outputs.squeeze()

                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Print progress
            if self.verbose > 0 and (epoch + 1) % self.verbose == 0:
                avg_loss = epoch_loss / (n_samples // self.batch_size + 1)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        self.training_time = time.time() - start_time

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            if self.num_classes == 2 and not self.multi_label:
                # Binary classification
                predictions = (torch.sigmoid(outputs) >= 0.5).squeeze()
            else:
                # Multi-class classification
                predictions = torch.argmax(outputs, dim=1)

            return predictions.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            if self.num_classes == 2 and not self.multi_label:
                # Binary classification
                probas = torch.sigmoid(outputs).cpu().numpy()
                return np.hstack([1 - probas, probas])
            else:
                # Multi-class classification
                probas = torch.softmax(outputs, dim=1).cpu().numpy()
                return probas

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        return {
            'architecture': self.architecture,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'device': str(self.device),
            'random_state': self.random_state,
        }


# Register neural network classifiers
ClassifierFactory.register('nn_mlp', lambda **kwargs: NeuralNetClassifier(architecture='mlp', **kwargs))
ClassifierFactory.register('nn_cnn', lambda **kwargs: NeuralNetClassifier(architecture='cnn', **kwargs))
ClassifierFactory.register('nn_resnet', lambda **kwargs: NeuralNetClassifier(architecture='resnet', **kwargs))
