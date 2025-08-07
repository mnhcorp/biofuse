import warnings
import json

# Disable all warnings
warnings.filterwarnings('ignore')

import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from catboost import CatBoostClassifier
from medmnist import BreastMNIST
import xgboost as xgb
# progressbar
from tqdm import tqdm
import sys, os, glob, csv
from biofuse.models.image_dataset import BioFuseImageDataset
import numpy as np
import copy
import medmnist
from medmnist import INFO
# --- MedMNIST-C robustness evaluation --------------------------------------
sys.path.append('/data/biofuse.git2/medmnistc')
from medmnistc.dataset import CorruptedMedMNIST
from medmnistc.eval import Evaluator
from medmnistc.corruptions.registry import CORRUPTIONS_DS
MEDMNIST_ROOT = '/data/medmnist'
MEDMNISTC_ROOT = '/data/medmnist-c'
# ---------------------------------------------------------------------------
import random
import argparse
#import ipdb
import itertools
import os
import pickle
import time
from datetime import datetime

# Trainable layer imports
import torch.optim as optim
import torch.nn as nn

# Preprocessor
from biofuse.models.processor import MultiModelPreprocessor

# import F from pytorch
import torch.nn.functional as F



PATIENCE = 25
CACHE_DIR = '/data/biofuse-embedding-cache3'

def get_cache_path(dataset, model, img_size, split):
    return os.path.join(CACHE_DIR, f'{dataset}_{model}_{img_size}_{split}.pkl')

def save_embeddings_to_cache(embeddings, labels, dataset, model, img_size, split):
    cache_path = get_cache_path(dataset, model, img_size, split)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((embeddings, labels), f)

def load_embeddings_from_cache(dataset, model, img_size, split):
    cache_path = get_cache_path(dataset, model, img_size, split)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def set_seed(seed: int = 42) -> None:
    """
    Set seed for generating random numbers to ensure reproducibility.

    Args:
    - seed (int): seed value
    """
    # Set seed that controls randomness related to PyTorch operations
    torch.manual_seed(seed)

    # Seed for randomness in CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)    # For multi-GPU

    # Set NumPy seed
    np.random.seed(seed)

    # Seed for random module
    random.seed(seed)

    # Control non-deterministic behavior for convolutional operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def log_resource_usage(dataset, model, extraction_time, vram_allocated, vram_reserved):
    file_path = 'embedding_extraction_log.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'dataset', 'model', 'extraction_time_seconds', 'vram_allocated_mb', 'vram_reserved_mb'])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, dataset, model, f'{extraction_time:.2f}', f'{vram_allocated:.2f}', f'{vram_reserved:.2f}'])


class LogisticRegression2(nn.Module):
    """
    Implements a logistic regression model using a linear layer.

    This model is a simple linear classifier that maps input features
    to output classes without applying a non-linear activation function
    like sigmoid or softmax in the forward pass. It's primarily used for
    binary classification tasks.

    Attributes:
        linear (nn.Linear): The linear transformation layer.

    Parameters:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Number of output classes.
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression2, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        #outputs = torch.sigmoid(self.linear(x))
        return self.linear(x)
    
# class MLPClassifier(nn.Module):
#     """
#     Implements a Multi-Layer Perceptron (MLP) for classification.

#     This model consists of a simple feedforward neural network with one
#     hidden layer and a ReLU activation function. It can be used for binary
#     or multi-class classification tasks.

#     Attributes:
#         mlp (nn.Sequential): The sequential container of layers forming the MLP.

#     Parameters:
#         input_dim (int): Dimensionality of the input features.
#         hidden_dim (int): Number of neurons in the hidden layer.
#         output_dim (int): Number of output classes, typically 1 for binary classification.
#     """
#     def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
#         super(MLPClassifier, self).__init__()
#         self.layer1 = nn.Linear(input_dim, hidden_dim)
#         self.layer2 = nn.Linear(hidden_dim, hidden_dim)
#         self.layer3 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()

#     # def forward(self, x):
#     #     # if not x.requires_grad:
#     #     #     x = x.requires_grad_(True)
#     #     return self.mlp(x)

#     def forward(self, x):
#         # Add debug prints
#         print(f"Input grad_fn: {x.grad_fn}")
        
#         x = self.layer1(x)
#         print(f"After layer1 grad_fn: {x.grad_fn}")
        
#         x = self.relu(x)
#         print(f"After relu1 grad_fn: {x.grad_fn}")
        
#         x = self.layer2(x)
#         print(f"After layer2 grad_fn: {x.grad_fn}")
        
#         x = self.relu(x)
#         print(f"After relu2 grad_fn: {x.grad_fn}")
        
#         x = self.layer3(x)
#         print(f"After layer3 grad_fn: {x.grad_fn}")
        
#         return x

class MLPArchitecture(nn.Module):
    """Two-layer MLP with dropout"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
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
    """1D CNN for feature processing"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),  # Expand features
            nn.Unflatten(1, (2, input_dim)),  # Reshape for 1D conv
            nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(hidden_dim // 2),
            nn.Flatten(),
            nn.Linear(hidden_dim * (hidden_dim // 2), output_dim)
        )

    def forward(self, x):
        return self.network(x)

class CNNArchitectureAdvanced(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Initial feature processing
            nn.BatchNorm1d(input_dim),  # Normalize input
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            
            # Reshape for CNN
            nn.Unflatten(1, (2, input_dim)),
            
            # Single but effective conv layer
            nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Pooling and final processing
            nn.AdaptiveAvgPool1d(hidden_dim // 2),
            nn.Flatten(),
            nn.Linear(hidden_dim * (hidden_dim // 2), hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        if self.training:
            x = x.requires_grad_()
        return self.network(x)

    # def predict(self, X):
    #     if isinstance(X, np.ndarray):
    #         X = torch.FloatTensor(X).to(next(self.parameters()).device)
    #     self.eval()
    #     with torch.no_grad():
    #         outputs = self(X)
    #         predictions = torch.sigmoid(outputs) >= 0.5
    #         return predictions.cpu().numpy().astype(int)

    # def predict_proba(self, X):
    #     if isinstance(X, np.ndarray):
    #         X = torch.FloatTensor(X).to(next(self.parameters()).device)
    #     self.eval()
    #     with torch.no_grad():
    #         outputs = self(X)
    #         probas = torch.sigmoid(outputs)
    #         return np.hstack([1 - probas.cpu().numpy(), probas.cpu().numpy()])



class ResidualArchitecture(nn.Module):
    """Residual network with skip connections"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = self.dropout(F.relu(self.fc2(h1))) + h1  # Skip connection
        return self.fc3(h2)

class NeuralNetClassifier(nn.Module):
    """Wrapper class for different neural architectures"""
    def __init__(self, input_dim, hidden_dim, output_dim, architecture='mlp', multi_label=False):
        super().__init__()
        self.multi_label = multi_label
        
        architectures = {
            'mlp': MLPArchitecture,
            'cnn': CNNArchitecture,
            'resnet': ResidualArchitecture,
            'cnn_adv': CNNArchitectureAdvanced
        }
        
        if architecture not in architectures:
            raise ValueError(f"Architecture must be one of {list(architectures.keys())}")
            
        self.network = architectures[architecture](input_dim, hidden_dim, output_dim)
        
    def forward(self, x):
        return self.network(x)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            if outputs.shape[1] == 1:  # binary
                predictions = torch.sigmoid(outputs) >= 0.5
            else:  # multi-class
                predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            if outputs.shape[1] == 1:  # binary
                probas = torch.sigmoid(outputs)
                return np.hstack([1 - probas.cpu().numpy(), probas.cpu().numpy()])
            else:  # multi-class
                return torch.softmax(outputs, dim=1).cpu.numpy()
        
    # def predict(self, X):
    #     # Convert numpy array to torch tensor if necessary
    #     if isinstance(X, np.ndarray):
    #         X = torch.FloatTensor(X).to(next(self.parameters()).device)
        
    #     # Set to evaluation mode
    #     self.eval()
        
    #     # Disable gradient computation for prediction
    #     with torch.no_grad():
    #         outputs = self(X)
    #         predictions = torch.sigmoid(outputs) >= 0.5
    #         return predictions.cpu().numpy().astype(int)

    # def predict_proba(self, X):
    #     # Convert numpy array to torch tensor if necessary
    #     if isinstance(X, np.ndarray):
    #         X = torch.FloatTensor(X).to(next(self.parameters()).device)
        
    #     # Set to evaluation mode
    #     self.eval()
        
    #     # Disable gradient computation for prediction
    #     with torch.no_grad():
    #         outputs = self(X)
    #         probas = torch.sigmoid(outputs)
    #         # Return probabilities for both classes
    #         return np.hstack([1 - probas.cpu().numpy(), probas.cpu().numpy()])




def train_nn_classifier(features, labels, num_classes, multi_label=False, architecture='mlp'):
    print(f"Training Neural Network classifier with {architecture} architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Print essential info
    print(f"num_classes: {num_classes}")
    print(f"device: {device}")
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"multi_label: {multi_label}")

    # Convert numpy arrays to torch tensors
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    if isinstance(labels, np.ndarray):
        if num_classes == 2:
            labels = torch.FloatTensor(labels)
        else:
            # Convert to long tensor and reshape for multi-class
            labels = torch.LongTensor(labels).squeeze()  # Remove extra dimension

    # Move to device
    features = features.to(device)
    labels = labels.to(device)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    output_dim = 1 if num_classes == 2 else num_classes
    hidden_dim = 256

    # Initialize model
    model = NeuralNetClassifier(input_dim=features.shape[1], hidden_dim=hidden_dim, output_dim=output_dim, multi_label=multi_label, architecture=architecture).to(device)
    
    # Crucial change: wrap the model in nn.DataParallel
    model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 128
    n_samples = len(features)
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        
        indices = torch.randperm(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = features[batch_indices]
            batch_y = labels[batch_indices]
            
            # Zero gradients
            model.zero_grad()
            optimizer.zero_grad()
            
            # Forward pass with gradient computation enabled
            with torch.set_grad_enabled(True):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Debug prints for first batch
                if epoch == 0 and start_idx == 0:
                    print(f"outputs shape: {outputs.shape}")
                    print(f"batch_y shape: {batch_y.shape}")
                    print(f"outputs device: {outputs.device}")
                    print(f"batch_y device: {batch_y.device}")
                    print(f"Model parameters require grad:")
                    for name, param in model.named_parameters():
                        print(f"{name}: {param.requires_grad}")
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= (n_samples // batch_size)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    return model.module, scaler  # Return the underlying model without DataParallel wrapper


def train_nn_classifier_bad(features, labels, num_classes, multi_label=False, architecture='mlp'):
    """
    Trains a neural network classifier using PyTorch.
    """
    print(f"Training Neural Network classifier with {architecture} architecture...")
    # print num_classes
    print(f"num_classes: {num_classes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")  
    
    # Scale features but keep as numpy for scaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Convert to PyTorch tensors with requires_grad=True
    X = torch.FloatTensor(features_scaled).requires_grad_(True).to(device)
    
    # Handle different label formats
    if multi_label or num_classes == 2:
        y = torch.FloatTensor(labels).to(device)
    else:
        y = torch.LongTensor(labels).squeeze().to(device)
        
    if y.dim() > 1:
        y = y.flatten()

    # print the shape of X and y
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # print the device of X and y
    print(f"X device: {X.device}")
    print(f"y device: {y.device}")
    
    # Initialize model
    input_dim = features.shape[1]
    hidden_dim = 256
    #output_dim = labels.shape[1] if multi_label else num_classes
    output_dim = 1 if num_classes == 2 else num_classes  # Single output neuron for binary

    class NeuralNetClassifier2(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, multi_label=False):
            super().__init__()
            self.multi_label = multi_label
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            # if not x.requires_grad:
            #     x = x.detach().requires_grad_(True)
            x = x.float()
            x.requires_grad_(True)
            return self.network(x)

        def check_gradients(self):
            """Debug method to check gradients of parameters"""
            for name, param in self.named_parameters():
                print(f"{name} requires_grad: {param.requires_grad}")

    model = NeuralNetClassifier2(input_dim, hidden_dim, output_dim, 
                               multi_label=multi_label).to(device)
    
    def check_model_devices(model):
        print(f"\nModel Device Check:")
        print(f"Model type: {type(model)}")
        
        # Check if any part of the model is on CUDA
        print(f"Is any part of model on CUDA: {next(model.parameters()).is_cuda}")
        
        # Check each parameter
        for name, param in model.named_parameters():
            print(f"Parameter {name}:")
            print(f"  - Device: {param.device}")
            print(f"  - Shape: {param.shape}")
            print(f"  - Requires grad: {param.requires_grad}")
        
        # Check if model is in training mode
        print(f"Model in training mode: {model.training}")

    check_model_devices(model)

    # model = NeuralNetClassifier(input_dim, hidden_dim, output_dim, 
    #                            architecture=architecture, 
    #                            multi_label=multi_label).to(device)

    #print model device
    #print(f"model device: {model.device}")

    # print input_dim, hidden_dim, output_dim
    print(f"input_dim: {input_dim}")
    print(f"hidden_dim: {hidden_dim}")
    print(f"output_dim: {output_dim}")
    
    # Training parameters
    if multi_label or num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    print(f"criterion: {criterion}")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    batch_size = min(128, len(features))  # Adjust batch size for small datasets
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start = time.time()
    best_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in loader:
            # Enable gradients for input
            batch_X = batch_X.clone().detach().requires_grad_(True)
            
            optimizer.zero_grad()
            
            # Check gradients before forward pass
            print("Before forward pass:")
            print(f"batch_X.requires_grad: {batch_X.requires_grad}")
            model.check_gradients()
            
            outputs = model(batch_X)
            
            # Check gradients after forward pass
            print("\nAfter forward pass:")
            print(f"outputs.requires_grad: {outputs.requires_grad}")
            print(f"outputs grad_fn: {outputs.grad_fn}")
            
            if num_classes == 2:
                loss = criterion(outputs, batch_y.unsqueeze(1).float())
            else:
                loss = criterion(outputs, batch_y)
                
            # Check loss gradients
            print("\nLoss info:")
            print(f"loss.requires_grad: {loss.requires_grad}")
            print(f"loss grad_fn: {loss.grad_fn}")
            
            if not loss.requires_grad:
                raise RuntimeError("Loss doesn't require grad!")
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(loader)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    
    return model, scaler

def custom_collate_fn(batch):
    """
    Custom collate function for handling batches in data loading.

    Args:
        batch (list): A list of tuples containing image and label pairs.

    Returns:
        tuple: A tuple containing a list of images and a tensor of labels.
    """
    # Filter out None values
    batch = [(img, label) for img, label in batch if img is not None]
    
    if len(batch) == 0:
        return [], torch.tensor([])

    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def parse_labels_from_path(path):
    filename = os.path.basename(path)
    parts = filename.split('_')
    
    # Extract all parts after the first one (train16672)
    label_part = parts[1:]
    
    # For the last part, remove the file extension
    label_part[-1] = label_part[-1].split('.')[0]
    
    return [int(label) for label in label_part]

def load_data(dataset, img_size, data_root=None, batch_size=32):
    """
    Load data for a given dataset.

    Args:
        dataset (str): The name of the dataset ('imagenet' or medmnist datasets)
        img_size (int): The desired image size
        data_root (str, optional): Root directory for dataset storage, required for ImageNet.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, 
               test data loader, and the number of classes.
    """
    print(f"Loading data for {dataset}...")
    
    from biofuse.models.data_adapter import DataAdapter

    if dataset in ['imagenet', 'imagenet-mini']:
        if data_root is None:
            raise ValueError("data_root must be specified for ImageNet dataset")
        
        # Configure parameters based on dataset type
        subset_size = 1.0 if dataset == 'imagenet' else 0.01  # Use 1% for imagenet-mini
        
        # Load ImageNet data using the simplified approach
        train_loader, num_classes = DataAdapter.from_imagenet(
            root=data_root, 
            split='train',
            batch_size=batch_size,
            subset_size=subset_size
        )
        
        # Load validation data
        val_loader, _ = DataAdapter.from_imagenet(
            root=data_root, 
            split='val',
            batch_size=batch_size,
            subset_size=subset_size
        )
        
        # Load test data
        test_loader, _ = DataAdapter.from_imagenet(
            root=data_root, 
            split='test',
            batch_size=batch_size,
            subset_size=subset_size
        )       
    elif dataset == 'busi':
        train_dataset, num_classes = DataAdapter.from_busi(data_root, 'train', img_size)
        val_dataset, _ = DataAdapter.from_busi(data_root, 'val', img_size)
        test_dataset, _ = DataAdapter.from_busi(data_root, 'test', img_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
    else:
        # MedMNIST loading logic
        train_dataset, num_classes = DataAdapter.from_medmnist(dataset, 'train', img_size)
        val_dataset, _ = DataAdapter.from_medmnist(dataset, 'val', img_size)
        test_dataset, _ = DataAdapter.from_medmnist(dataset, 'test', img_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Number of classes: {num_classes}")

    # # Print the first sample from the training set, both x and y
    # print("First sample from the training set:")
    # print('--'*20)
    # print(next(iter(train_loader)))
    # print('--'*20)
    
    return train_loader, val_loader, test_loader, num_classes

def extract_features(dataloader, biofuse_model):
    """
    Extracts features from the given dataloader using the provided biofuse_model.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the images 
        and labels. biofuse_model (torch.nn.Module): The biofuse model used for feature extraction.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the 
        extracted features, and the second array contains the corresponding labels.
    """
    print("Extracting features...")
    features = []
    labels = []    
    # use progress bar
    for image, label in tqdm(dataloader):
        embedding = biofuse_model(image)
        features.append(embedding.squeeze(0).detach().numpy())
        labels.append(label.numpy())
   
    return np.array(features), np.array(labels)
    
def print_cuda_mem_stats():
    """
    Prints the current CUDA memory statistics.

    This function prints the amount of memory allocated and reserved on the CUDA device.
    """
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.0f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.0f} MB")

def generate_embeddings(dataloader, biofuse_model, cache_raw_embeddings=False, is_training=True, is_test=False, progress_bar=False):
    """
    Generate embeddings for a given dataloader using a BioFuse model.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        biofuse_model (torch.nn.Module): The BioFuse model used for generating embeddings.
        cache_raw_embeddings (bool, optional): Whether to cache raw embeddings during training. Defaults to False.
        is_training (bool, optional): Whether the model is in training mode. Defaults to True.
        is_test (bool, optional): Whether the embeddings are generated for testing. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        torch.Tensor: The generated embeddings.
        torch.Tensor: The corresponding labels.
    """
    
    embeddings = []
    labels = []    

    data_iter = enumerate(dataloader)
    if progress_bar:
        data_iter = enumerate(tqdm(dataloader))

    
    #for image, label in dataloader:
    for index, (image, label) in data_iter:
        if is_test:
            # use forward_test
            embedding = biofuse_model.forward_test(image)
        else:
            embedding = biofuse_model(image, cache_raw_embeddings=cache_raw_embeddings, index=index, is_training=is_training)

        #print_cuda_mem_stats()
        # generate a random tensor for now
        #embedding = torch.randn(512)
        embeddings.append(embedding)
        labels.append(label)
    
    # Embeddings is a list of tensors, stack them and remove the batch dimension
    embeddings_tensor = torch.stack(embeddings).squeeze(1)
    labels_tensor = torch.tensor(labels)        
    
    return embeddings_tensor, labels_tensor

def print_trainable_parameters(model):
    """
    Prints the trainable parameters of the given model.

    Args:
        model: The model whose trainable parameters need to be printed.

    Returns:
        None
    """
    print("Trainable parameters:")
    for name, param in model.named_parameters():   
        if param.requires_grad:
            print(name, param.shape)
            print(name, param.numel())

def log_projection_layer_weights(model, epoch, stage):
    """
    Logs the weights of the projection layers in the model.

    Args:
        model (nn.Module): The model containing the projection layers.
        epoch (int): The current epoch number.
        stage (str): The current stage of training.

    Returns:
        None
    """
    for i, layer in enumerate(model.projection_layers):
        print(f"Epoch [{epoch}] - {stage} - Projection Layer {i} Weights:")
        for name, param in layer.named_parameters():  # Iterate through MLP parameters
            weights = param.data
            print(f"  - {name}: {weights.mean().item():.6f} ± {weights.std().item():.6f}")

def log_projection_layer_gradients(model, epoch, stage):
    """
    Logs the gradients of the projection layers in the model.

    Args:
        model (nn.Module): The model containing the projection layers.
        epoch (int): The current epoch number.
        stage (str): The current stage of training.

    Returns:
        None
    """
    for i, layer in enumerate(model.projection_layers):
        print(f"Epoch [{epoch}] - {stage} - Projection Layer {i} Gradients:")
        for name, param in layer.named_parameters():  # Iterate through MLP parameters
            if param.grad is not None:
                grad = param.grad.data
                print(f"  - {name}: {grad.mean().item():.6f} ± {grad.std().item():.6f}")
            else:
                print(f"  - {name}: None")

def train_classifier(features, labels, num_classes):
    """
    Trains a logistic regression classifier on the given features and labels.

    Args:
        features (array-like): The input features for training the classifier.
        labels (array-like): The target labels for training the classifier.
        num_classes (int): The number of classes in the classification problem.

    Returns:
        tuple: A tuple containing the trained classifier and the scaler used for feature scaling.
    """
    print("Training classifier...")

    scaler = StandardScaler()
    
    # Scale features
    features = scaler.fit_transform(features)

    if num_classes > 2:
        if len(features) < 1000:
            classifier = LogisticRegression(max_iter=1000, solver='liblinear', multi_class='ovr')
        else:
            classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    else:
        classifier = LogisticRegression(max_iter=1000, solver='liblinear')

    classifier.fit(features, labels)
    return classifier, scaler

def get_xgb_params(n_samples, params=None):
    """
    Returns XGBoost parameters based on dataset size (n_samples).
    
    Args:
        n_samples: Number of samples in your dataset
        
    Returns:
        Dictionary of XGBoost parameters with sensible defaults
    """
    if params:
        return {
            'max_depth': params.get('max_depth', 6),
            'n_estimators': params.get('n_estimators', 250),
            'learning_rate': params.get('learning_rate', 0.1),
            'reg_alpha': params.get('reg_alpha', 0),
            'reg_lambda': params.get('reg_lambda', 0),
            'subsample': params.get('subsample', 1),
            'colsample_bytree': params.get('colsample_bytree', 1),
            'min_child_weight': params.get('min_child_weight', 1),
            'max_bin': params.get('max_bin', 256),
            'max_leaves': params.get('max_leaves', 0)
        }

    params = dict()

    if n_samples < 1000:
        # Very small dataset configuration
        params.update({
            'max_depth': 3,
            'n_estimators': 50,
            'learning_rate': 0.2
        })
    elif 1000 <= n_samples < 10000:
        # Small dataset configuration
        params.update({
            'max_depth': 4,
            'n_estimators': 100,
            'learning_rate': 0.1
        })
    elif 10000 <= n_samples < 100000:
        # Medium dataset configuration
        params.update({
            'max_depth': 6,
            'n_estimators': 200,
            'learning_rate': 0.05
        })
    else:  # 100k+ samples
        # Large dataset configuration
        params.update({
            'max_depth': 8,
            'n_estimators': 500,
            'learning_rate': 0.01    
        })

    # Fix the params to n_estiamtors:100, learning_rate:0.1, max_depth:6
    params.update({
        'n_estimators': 250,
        'learning_rate': 0.1,
        'max_depth': 6
    })

    return params

def train_classifier2(features, labels, num_classes, multi_label=False, params=None):
    """
    Trains an XGBoost classifier using the provided features and labels.

    Args:
        features (array-like): The input features for training the classifier.
        labels (array-like): The corresponding labels for the input features.
        num_classes (int): The number of classes in the classification problem.
        multi_label (bool): Whether this is a multi-label classification problem.
        params (dict, optional): Custom XGBoost parameters.

    Returns:
        tuple: (classifier, scaler, training_time) - The trained classifier, scaler, and training time.
    """
    print("Training XGBoost classifier...")

    scaler = StandardScaler()
    
    # Scale features
    features = scaler.fit_transform(features)
    # record time for training
    import time
    start = time.time()
    
    # Get XGBoost parameters based on dataset size
    xgb_params = get_xgb_params(len(features), params)

    if num_classes > 2 and not multi_label:
        print("Multi-class classification")
        classifier = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            n_estimators=xgb_params['n_estimators'],
            learning_rate=xgb_params['learning_rate'],
            max_depth=xgb_params['max_depth'],
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=32,
            tree_method='gpu_hist',
            **{k: v for k, v in xgb_params.items() if k not in ['n_estimators', 'learning_rate', 'max_depth']}
        )
    else:
        print("Binary classification")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=xgb_params['n_estimators'],
            learning_rate=xgb_params['learning_rate'],
            max_depth=xgb_params['max_depth'],
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=32,
            tree_method='gpu_hist',
            **{k: v for k, v in xgb_params.items() if k not in ['n_estimators', 'learning_rate', 'max_depth']}
        )

        if multi_label:
            classifier = OneVsRestClassifier(xgb_model)
        else:
            classifier = xgb_model               

    classifier.fit(features, labels)
    print('==========================')
    # if multi_label:
    #     classifier.fit(features, labels)
    # else:
    #     classifier.fit(features, labels, verbose=True, early_stopping_rounds=50, eval_set=[(features, labels)])


    end = time.time()
    training_time = end - start
    #print(f"Time taken to train XGBoost classifier: {training_time:.2f} seconds")

    return classifier, scaler, training_time

def train_catboost_classifier(features, labels, num_classes, multi_label=False):
    """
    Trains a CatBoost classifier using the provided features and labels.

    Args:
        features (array-like): The input features for training the classifier.
        labels (array-like): The corresponding labels for the input features.
        num_classes (int): The number of classes in the classification problem.
        multi_label (bool): Whether this is a multi-label classification problem.

    Returns:
        tuple: (classifier, scaler) - The trained classifier and the scaler used for feature scaling.
    """        
    print("Training CatBoost classifier...")

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Record training time
    start = time.time()
    
    # Common parameters for all scenarios
    base_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'task_type': 'GPU',
        'devices': '0',  # Use first GPU
        'verbose': 100,
        'thread_count': 32
    }

    if num_classes > 2 and not multi_label:
        print("Multi-class classification")
        classifier = CatBoostClassifier(
            **base_params,
            loss_function='MultiClass',
            classes_count=num_classes,
            eval_metric='MultiClass'
        )
    else:
        print("Binary classification")
        cat_model = CatBoostClassifier(
            **base_params,
            loss_function='Logloss',
            eval_metric='AUC'
        )
        
        if multi_label:            
            classifier = OneVsRestClassifier(cat_model)
        else:
            classifier = cat_model

    # Train the classifier
    classifier.fit(features, labels)

    end = time.time()
    print(f"Time taken to train CatBoost classifier: {end - start:.2f} seconds")

    return classifier, scaler

def evaluate_model(classifier, features, labels, dataset=None):
    """
    Evaluates the classifier on the given features and labels.

    Args:
    - classifier: The trained classifier.
    - features: The input features for evaluation.
    - labels: The target labels for evaluation.
    - dataset: The name of the dataset being evaluated.

    Returns:
    - tuple or float: For ImageNet, returns (top1_accuracy, top5_accuracy).
                     For other datasets, returns accuracy.
    """
    print("Evaluating model...")
    if dataset in ['imagenet', 'imagenet-mini']:
        # Get probability predictions
        probs = classifier.predict_proba(features)
        # Get top-k predictions
        top1_preds = np.argmax(probs, axis=1)
        top5_preds = np.argsort(probs, axis=1)[:, -5:]

        # Calculate top-1 accuracy
        top1_acc = accuracy_score(labels, top1_preds)
        
        # Calculate top-5 accuracy
        top5_acc = np.mean([label in pred_top5 for label, pred_top5 in zip(labels, top5_preds)])
        
        return top1_acc, top5_acc
    else:
        # Check if it's a multi-label problem (like ChestMNIST)
        is_multilabel = len(labels.shape) > 1 and labels.shape[1] > 1
        
        if is_multilabel:
            threshold = 0.5
            y_score = np.array(classifier.predict(features))
            y_pred = y_score > threshold
            acc = 0
            for i in range(labels.shape[1]):
                acc += accuracy_score(labels[:, i], y_pred[:, i])
            return acc / labels.shape[1]
        else:
            predictions = classifier.predict(features)

            # Ensure predictions and labels are 1D arrays
            if isinstance(predictions, np.ndarray) and len(predictions.shape) > 1:
                predictions = predictions.squeeze()
            if isinstance(labels, np.ndarray) and len(labels.shape) > 1:
                labels = labels.squeeze()

            return accuracy_score(labels, predictions)

from torchvision.transforms.functional import to_pil_image

def tensor_to_pil_safe(tensor):
    """Convert tensor to PIL Image safely"""
    from PIL import Image
    import numpy as np
    
    # Move to CPU if needed
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Handle different tensor formats
    if tensor.dtype == torch.uint8:
        # Already in 0-255 range
        np_array = tensor.numpy()
    else:
        # Assume 0-1 range, convert to 0-255
        np_array = (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)
    
    # Handle grayscale vs RGB
    if len(np_array.shape) == 3:  # [C, H, W]
        if np_array.shape[0] == 1:  # Grayscale
            np_array = np_array.squeeze(0)  # [H, W]
        else:  # RGB
            np_array = np_array.transpose(1, 2, 0)  # [H, W, C]
    
    return Image.fromarray(np_array)

 # ---------------------------------------------------------------------------
 # MedMNIST‑C robustness evaluation (helper only – not invoked yet)
 # ---------------------------------------------------------------------------
def eval_medmnistc(dataset, models, biofuse_model, classifier, scaler, clean_probs, clean_labels, root):
    """
    Evaluate a model on MedMNIST-C corruptions
    """
    print(f"\nEvaluating MedMNIST-C corruptions for dataset: {dataset}")
    
    # Just use CUDA
    device = torch.device("cuda")
    
    # Get MedMNIST info
    from medmnist import INFO
    info = INFO[dataset]
    task = info['task']
    
    # Get clean test dataset to extract true labels in the expected format
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset_clean = DataClass(split='test', download=False, root=MEDMNIST_ROOT)
    
    # Check the shape of true_labels and reshape if needed
    true_labels = test_dataset_clean.labels
    
    # Ensure true_labels has the correct shape (needs to be 2D)
    if len(true_labels.shape) == 1:
        # Reshape 1D array to 2D
        true_labels = true_labels.reshape(-1, 1)
        
    corruption_types = list(CORRUPTIONS_DS[dataset].keys())
    
    # Initialize the evaluator with properly shaped labels
    evaluator = Evaluator(
        dataset_name=dataset,
        true_labels=true_labels,  # Now properly shaped
        corruption_types=corruption_types,
        output_folder='./',
        architecture=f"BioFuse-{'+'.join(models)}",
        task=task,
        suffix_log=f"fusion-{biofuse_model.fusion_method}"
    )
    
    # Process clean predictions
    if task == "multi-label, binary-class":
        # Multi-label
        y_pred_clean = clean_probs
        # Create a threshold array of 0.5 for each class
        num_classes = true_labels.shape[1]
        threshold = np.array([0.5] * num_classes)
    elif task == "binary-class":
        # Binary
        y_pred_clean = clean_probs[:, 1].reshape(-1, 1)
        threshold = 0.5
    else:
        # Multi-class
        y_pred_clean = clean_probs
        threshold = None
        
    # Evaluate clean performance
    evaluator.evaluate_clean(y_pred_clean, threshold=threshold)

    # Cache extractors once
    extractors = {m: PreTrainedEmbedding(m) for m in models}
    
    # KEY DIFFERENCE: Create SEPARATE preprocessor for EACH model
    preprocessors = {m: MultiModelPreprocessor([m]) for m in models}

    for corr in corruption_types:
        try:
            print(f"\nProcessing corruption: {corr}")
            ds = CorruptedMedMNIST(dataset_name=dataset,
                               corruption=corr,
                               root=root,
                               mmap_mode='r')
            
            #print(f"Corrupted dataset size for {corr}: {len(ds)}")
            
            batch_size = 128 if dataset in ['tissuemnist', 'chestmnist', 'pathmnist', 'octmnist'] else 32
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            
            # Cache embeddings for each model and corruption type
            fused_batches = []
            for imgs, _ in tqdm(loader, leave=False, desc=f'[MedMNIST‑C] {corr}'):
                model_embeddings = []
    
                for model_name in models:
                    processed = preprocessors[model_name].preprocess_tensor(imgs)
                    
                    with torch.no_grad():
                        embedding = extractors[model_name](processed).detach()
                    model_embeddings.append(embedding)

                    # --- CACHE INDIVIDUAL MODEL EMBEDDINGS ---
                    cache_path = os.path.join(CACHE_DIR, f"{dataset}_{model_name}_{corr}.pt")
                    if not os.path.exists(cache_path):
                        torch.save(embedding.cpu(), cache_path)
                        # To load: embedding = torch.load(cache_path)
                        # (This loads the cached embedding tensor for <model> and <corruption_type>)
                
                with torch.no_grad():
                    fused = biofuse_model(model_embeddings).cpu()
                fused_batches.append(fused)

            feats = torch.cat(fused_batches).numpy()
            feats = scaler.transform(feats)
            probs = classifier.predict_proba(feats)

            evaluator.evaluate(probs, corruption_type=corr, threshold=threshold)
        except Exception as e:
            print(f"Error processing corruption {corr}: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    try:
        evaluator.dump_summary()
        met = evaluator.output_log['metrics']
        print(f'\n[MedMNIST‑C] {dataset}: BE={met["be"]:.3f}  rBE={met["rbe"]:.3f}\n')
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        import traceback
        traceback.print_exc()

# method to compute AUC-ROC for binary or multi-class classification
def compute_auc_roc(classifier, features, labels, num_classes, dataset=None):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for the classifier.

    Args:
    - classifier: The trained classifier.
    - features: The input features for evaluation.
    - labels: The target labels for evaluation.
    - dataset: The name of the dataset being evaluated.

    Returns:
    - float or None: The AUC-ROC score of the classifier on the given features and labels,
                    or None if dataset is ImageNet/ImageNet-mini.
    """
    # Skip AUC-ROC calculation for ImageNet datasets
    if dataset in ['imagenet', 'imagenet-mini']:
        return None
        
    print("Computing AUC-ROC...")
    if num_classes == 2:
        predictions = classifier.predict_proba(features)[:, 1]
        return roc_auc_score(labels, predictions)
    else:
        # Check if it's a multi-label problem (like ChestMNIST)
        is_multilabel = len(labels.shape) > 1 and labels.shape[1] > 1
        
        if is_multilabel:
            y_score = np.array(classifier.predict_proba(features))    
            auc_scores = []
            for i in range(labels.shape[1]):
                if len(np.unique(labels[:, i])) > 1:
                    auc = roc_auc_score(labels[:, i], y_score[:, i])
                    auc_scores.append(auc)
            avg_auc = np.mean(auc_scores)
            return avg_auc
        
        # use one-vs-all strategy
        predictions = classifier.predict_proba(features)
        return roc_auc_score(labels, predictions, multi_class='ovr')
    
def standalone_eval(models, biofuse_model, train_embeddings, train_labels, val_embeddings, val_labels, 
                   test_embeddings, test_labels, num_classes, dataset, test_classifier_fn=None):
    biofuse_model.eval()
    device = torch.device("cuda")
    
    multi_label = False
    if dataset == "chestmnist":
        multi_label = True
        # Fix the label shapes for ChestMNIST
        if len(train_labels.shape) == 3:
            train_labels = train_labels.squeeze(1)
        if val_labels is not None and len(val_labels.shape) == 3:
            val_labels = val_labels.squeeze(1)
        if test_labels is not None and len(test_labels.shape) == 3:
            test_labels = test_labels.squeeze(1)

    xgb_train_time = 0  # Initialize training time tracking
    xgb_val_inf_time = 0  # Initialize validation inference time tracking
    start_inf = 0  # Will be set later

    with torch.no_grad():
        # Process train embeddings
        train_fused_embeddings = biofuse_model([train_embeddings[model].to(device) for model in models])
        
        # Only move to CPU if not using neural network classifier
        if test_classifier_fn and 'nn_' in test_classifier_fn.__name__:
            # Clone and enable gradients for neural network training
            train_fused_embeddings_np = train_fused_embeddings.clone().detach().requires_grad_(True)
            train_labels_np = train_labels.to(device)
        else:
            train_fused_embeddings_np = train_fused_embeddings.cpu().numpy()
            train_labels_np = train_labels.cpu().numpy()

        if multi_label:
            train_labels = train_labels.view(-1,14)
            
        # For ImageNet datasets, check if trained model exists
        classifier_path = None
        scaler_path = None
        if dataset in ['imagenet', 'imagenet-mini']:
            model_str = '_'.join(sorted(models))
            classifier_path = f'trained_models/{dataset}_{model_str}_classifier.pkl'
            scaler_path = f'trained_models/{dataset}_{model_str}_scaler.pkl'
            
            if os.path.exists(classifier_path) and os.path.exists(scaler_path):
                print("Loading existing trained model...")
                with open(classifier_path, 'rb') as f:
                    val_classifier = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    val_scaler = pickle.load(f)
                # No training time for cached model
            else:
                # Train a new classifier on fused embeddings
                val_classifier, val_scaler, model_train_time = train_classifier2(train_fused_embeddings_np, train_labels_np, num_classes, multi_label)
                xgb_train_time += model_train_time
                
                # Save the trained model
                print("Saving trained model...")
                os.makedirs('trained_models', exist_ok=True)
                with open(classifier_path, 'wb') as f:
                    pickle.dump(val_classifier, f)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(val_scaler, f)
        else:
            # For non-ImageNet datasets, train classifier as usual
            val_classifier, val_scaler, model_train_time = train_classifier2(train_fused_embeddings_np, train_labels_np, num_classes, multi_label)
            xgb_train_time += model_train_time

        # Train test classifier if a different one is requested
        test_classifier, test_scaler = None, None
        if test_classifier_fn is not None:
            # Simple fix - check if test_classifier_fn is different than default
            if test_classifier_fn is not None and test_classifier_fn.__name__ != train_classifier2.__name__:
                test_classifier, test_scaler, test_train_time = test_classifier_fn(train_fused_embeddings_np, train_labels_np, 
                                                                num_classes, multi_label)
                xgb_train_time += test_train_time
            else:
                test_classifier, test_scaler = val_classifier, val_scaler

        val_accuracy, val_auc_roc = None, None
        if val_embeddings is not None and val_labels is not None:
            val_fused_embeddings = biofuse_model([val_embeddings[model].to(device) for model in models])
            
            # Keep on GPU for neural network classifier
            if test_classifier_fn and 'nn_' in test_classifier_fn.__name__:
                val_fused_embeddings_np = val_fused_embeddings.clone().detach().requires_grad_(True)
                val_labels_np = val_labels.to(device)
            else:
                val_fused_embeddings_np = val_fused_embeddings.cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()
            
            # Scale val embeddings if not using neural network
            if not (test_classifier_fn and 'nn_' in test_classifier_fn.__name__):
                val_fused_embeddings_np = val_scaler.transform(val_fused_embeddings_np)

            # Start timing validation inference
            import time
            start_inf = time.time()
            
            # Save validation predictions for ImageNet datasets
            if dataset in ['imagenet', 'imagenet-mini']:
                val_predictions = val_classifier.predict_proba(val_fused_embeddings_np)
                #save_val_predictions(val_predictions, val_labels, models)

            # Evaluate on validation set
            val_accuracy = evaluate_model(val_classifier, val_fused_embeddings_np, val_labels_np, dataset)
            val_auc_roc = compute_auc_roc(val_classifier, val_fused_embeddings_np, val_labels_np, num_classes, dataset)
            
            # End timing validation inference
            end_inf = time.time()
            xgb_val_inf_time = end_inf - start_inf
            #print(f"Time taken for validation inference: {xgb_val_inf_time:.2f} seconds")

            if dataset in ['imagenet', 'imagenet-mini']:
                val_top1, val_top5 = val_accuracy
                print(f"Validation Top-1 Accuracy: {val_top1:.4f}")
                print(f"Validation Top-5 Accuracy: {val_top5:.4f}")
            else:
                print(f"Validation Accuracy: {val_accuracy:.4f}")
                if val_auc_roc is not None:
                    print(f"Validation AUC-ROC: {val_auc_roc:.4f}")

        test_accuracy, test_auc_roc = 0.0, 0.0
        if test_embeddings is not None: # and False:  # Never execute this block
            test_fused_embeddings = biofuse_model([test_embeddings[model].to(device) for model in models])
            
            # Keep on GPU for neural network classifier
            if test_classifier_fn and 'nn_' in test_classifier_fn.__name__:
                test_fused_embeddings_np = test_fused_embeddings.clone().detach().requires_grad_(True)
                test_labels_np = test_labels.to(device)
            else:
                test_fused_embeddings_np = test_fused_embeddings.cpu().numpy()
                test_labels_np = test_labels.cpu().numpy()

            # Use test classifier and scaler if provided, otherwise use validation ones
            classifier = test_classifier if test_classifier is not None else val_classifier
            scaler = test_scaler if test_scaler is not None else val_scaler

            if dataset in ['imagenet', 'imagenet-mini']:
                # For ImageNet test set, generate predictions and save them
                if not (test_classifier_fn and 'nn_' in test_classifier_fn.__name__):
                    test_fused_embeddings_np = scaler.transform(test_fused_embeddings_np)
                test_predictions = classifier.predict_proba(test_fused_embeddings_np)
                save_test_predictions(test_predictions, test_labels, models)  # test_labels contains filenames
                test_accuracy = None
                test_auc_roc = None
            else:
                if not (test_classifier_fn and 'nn_' in test_classifier_fn.__name__):
                    test_fused_embeddings_np = scaler.transform(test_fused_embeddings_np)
                test_accuracy = evaluate_model(classifier, test_fused_embeddings_np, test_labels_np, dataset)
                test_auc_roc = compute_auc_roc(classifier, test_fused_embeddings_np, test_labels_np, num_classes, dataset)
                print(f"Test Accuracy: {test_accuracy:.4f}")
                if test_auc_roc is not None:
                    print(f"Test AUC-ROC: {test_auc_roc:.4f}")
                    
                # does (global) args have test_noise?
                if hasattr(args, 'test_noise') and args.test_noise:
                    clean_probs = classifier.predict_proba(test_fused_embeddings_np)
                    eval_medmnistc(dataset       = dataset,
                                   models        = models,
                                   biofuse_model = biofuse_model.eval(),
                                   classifier    = classifier,
                                   scaler        = scaler,
                                   clean_probs   = clean_probs,
                                   clean_labels  = test_labels_np,
                                   root          = args.medmnistc_root)

    # Save the models and scalers if they were trained, only for medmnist datasets
    if dataset not in ['imagenet', 'imagenet-mini']:
        model_str = '_'.join(sorted(models))
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
            
        # Only save the trained model and the scalers
        if classifier_path is None:
            classifier_path = f'trained_models/{dataset}_{model_str}_classifier.pkl'
            scaler_path = f'trained_models/{dataset}_{model_str}_scaler.pkl'
        
        with open(classifier_path, 'wb') as f:
            pickle.dump(val_classifier, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(val_scaler, f)
            
        # To load the model later, use:
        # with open(classifier_path, 'rb') as f:
        #     val_classifier = pickle.load(f)
        # with open(scaler_path, 'rb') as f:
        #     val_scaler = pickle.load(f)
            
    # Print total XGBoost training and inference time
    # print(f"Total XGBoost training time: {xgb_train_time:.2f} seconds")
    # print(f"Total XGBoost validation inference time: {xgb_val_inf_time:.2f} seconds")

    # Return timings along with accuracy metrics
    return val_accuracy, val_auc_roc, test_accuracy, test_auc_roc, xgb_train_time, xgb_val_inf_time


def extract_all_embeddings(model_names, dataloaders, dataset, img_size, nocache):
    """
    Extracts and caches embeddings for all models and data splits.

    This function first checks if cached embeddings exist before loading models,
    and only loads a model when necessary to extract embeddings.
    """
    all_embeddings = {'train': {}, 'val': {}, 'test': {}}
    all_labels = {'train': None, 'val': None, 'test': None}

    for model_name in model_names:
        print(f"\nProcessing model: {model_name}")
        
        # Check if cached embeddings exist for all splits before loading the model
        all_cached = True
        if not nocache:
            for split in ['train', 'val', 'test']:
                if dataloaders[split] is None:
                    continue
                    
                cached_data = load_embeddings_from_cache(dataset, model_name, img_size, split)
                if cached_data:
                    print(f"Found cached embeddings for {model_name} ({split})")
                    all_embeddings[split][model_name], all_labels[split] = cached_data
                else:
                    all_cached = False
                    break
        else:
            all_cached = False
            
        # Skip model loading if all embeddings are cached
        if all_cached:
            print(f"Using cached embeddings for all splits of {model_name}")
            continue
            
        # Load model only if needed
        try:
            extractor = PreTrainedEmbedding(model_name)
            preprocessor = MultiModelPreprocessor([model_name])
            vram_allocated = torch.cuda.memory_allocated() / 1024**2
            vram_reserved = torch.cuda.memory_reserved() / 1024**2
        except Exception as e:
            print(f"Failed to load model {model_name}. Skipping. Error: {e}")
            continue

        total_extraction_time = 0
        
        for split in ['train', 'val', 'test']:
            dataloader = dataloaders[split]
            if dataloader is None:
                continue

            # Check cache again (might have loaded some splits above)
            if not nocache:
                cached_data = load_embeddings_from_cache(dataset, model_name, img_size, split)
                if cached_data:
                    print(f"Loaded cached embeddings for {model_name} ({split})")
                    all_embeddings[split][model_name], all_labels[split] = cached_data
                    continue
            
            # If not cached, extract embeddings
            model_embeddings = []
            model_labels = []
            
            start_time = time.time()
            for images, batch_labels in tqdm(dataloader, desc=f"Extracting embeddings for {model_name} ({split})"):
                if not images: continue
                # print images.shape
                #print(f"Processing batch of images with shape: {len(images)}")
                #print(f"Before preprocessing, images type: {type(images[0])}")  # This is PIL.Image
                processed_images = preprocessor.preprocess(images)[0]
                #print type
                #print(f"Processed images type: {type(processed_images)}")
                with torch.no_grad():
                    embeddings = extractor(processed_images)
                model_embeddings.append(embeddings)
                model_labels.append(batch_labels)
            
            total_extraction_time += time.time() - start_time

            if not model_embeddings: continue

            # Store and cache the results
            current_embeddings = torch.cat(model_embeddings)
            all_embeddings[split][model_name] = current_embeddings
            
            if all_labels[split] is None:
                 if split == 'test' and dataset in ['imagenet', 'imagenet-mini']:
                     all_labels[split] = np.concatenate(model_labels)
                 else:
                     all_labels[split] = torch.cat(model_labels)

            if not nocache:
                save_embeddings_to_cache(current_embeddings, all_labels[split], dataset, model_name, img_size, split)
                print(f"Saved embeddings for {model_name} ({split}) to cache")

        # Only log resource usage if model was actually loaded
        log_resource_usage(dataset, model_name, total_extraction_time, vram_allocated, vram_reserved)

        # Clean up memory
        del extractor
        del preprocessor
        torch.cuda.empty_cache()

    return all_embeddings['train'], all_labels['train'], \
           all_embeddings['val'], all_labels['val'], \
           all_embeddings['test'], all_labels['test']

def harmonic_mean(val_acc, val_auc):
    return 2 / ((1 / val_acc) + (1 / val_auc))

def weighted_mean(val_acc, val_auc):
    if val_auc is None:
        return val_acc
    weights = [0.4, 0.6]        
    score = weights[0] * val_acc + weights[1] * val_auc
    return score

def weighted_mean_with_penalty(val_acc, val_auc):
    if val_auc is None:
        return val_acc
        
    weights = [0.4, 0.6]        
    score = weights[0] * val_acc + weights[1] * val_auc
    
    # Define thresholds for penalties
    acc_threshold = 0.90
    auc_threshold = 0.90
    
    # Define penalty factors
    acc_penalty_factor = 0.2
    auc_penalty_factor = 0.2
    
    # Calculate penalties based on exceeding thresholds
    acc_penalty = acc_penalty_factor * max(0, val_acc - acc_threshold)
    auc_penalty = auc_penalty_factor * max(0, val_auc - auc_threshold)
    
    # Apply penalties to the score
    penalized_score = score - acc_penalty - auc_penalty
    
    return penalized_score

def get_configurations(dataset, model_names, file_path, single):
    # if single, there is only a single configuration with all models
    if single:
        return [tuple(model_names)]
    
    # Generate all combinations of pre-trained models
    configurations = []
    for r in range(1, len(model_names) + 1):
       configurations.extend(itertools.combinations(model_names, r))

    # Print the size of the configurations
    print(f"Number of configurations: {len(configurations)}")

    # Check if the results file exists
    if os.path.isfile(file_path):
        # Read all rows, ignore first row
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                # Extract the models and fusion method
                models = row[2].split(',')                
                
                # Remove the models from the configurations
                if tuple(models) in configurations:
                    configurations.remove(tuple(models))

        # Print the new configurations size 
        print(f"Number of configurations after removing existing results: {len(configurations)}")
        # # print configurations 1 per line in file configs.txt
        # with open('configs.txt', 'w') as f:
        #     for config in configurations:
        #         f.write(','.join(config) + '\n')
        # print(f"Configurations saved to configs.txt")
        
    #sys.exit(0)  # Exit after skipping in-progress runs                
    return configurations

def is_config_running(dataset, config):
    """
    Check if a configuration is currently running by looking for its entry in the .current-run directory.
    
    Args:
        dataset (str): The name of the dataset.
        config (tuple): The configuration tuple containing model names.
        
    Returns:
        bool: True if the configuration is running, False otherwise.
    """
    current_run_dir = '.current-run'
    if not os.path.isdir(current_run_dir):
        return False
    
    # Create a unique identifier for the configuration
    config_str = ','.join(config)
    
    # Check if any file in the current run directory contains this configuration
    for run_file in os.listdir(current_run_dir):
        if run_file.startswith(f"{dataset}_") and run_file.endswith('.txt'):
            with open(os.path.join(current_run_dir, run_file), 'r') as f:
                if f.read().strip() == config_str:
                    return True
    return False
        
# Training the model with validation-informed adjustment
def train_model(dataset, model_names, num_epochs, img_size, projection_dims, fusion_methods, single=False, nocache=False, data_root=None, test_classifier='xgb', ood_test_set=None, ood_data_root=None, params=None, batch_size=32):
    set_seed(42)

    file_path = f"results_{dataset}_{img_size}.csv"
    configurations = get_configurations(dataset, model_names, file_path, single)
    #print(configurations)

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_data(dataset, img_size, data_root=data_root)
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    # Extract and cache embeddings
    print("Extracting and caching embeddings...")
    
    train_embeddings_cache, train_labels, val_embeddings_cache, val_labels, test_embeddings_cache, test_labels = extract_all_embeddings(
        model_names, dataloaders, dataset, img_size, nocache
    )

    # Save the test labels to a file
    if dataset in ['imagenet', 'imagenet-mini']:
        with open(f"{dataset}_test_labels.txt", 'w') as f:
            for label in test_labels:
                f.write(f"{label}\n")

    best_config = None
    best_val_acc = 0
    best_test_acc = 0
    best_val_auc_roc = 0
    best_test_auc_roc = 0
    best_harmonic_mean = 0

    # shuffle the configurations
    random.shuffle(configurations)

    # First pass: Evaluate configurations with a single epoch
    print("\nFirst pass: Evaluating model combinations")
    total_xgb_eval_time = 0  # Track total XGBoost training time
    total_xgb_val_inf_time = 0  # Track total validation inference time
    for models in configurations:
        for fusion_method in fusion_methods:
            # Skip if this configuration is already running
            if is_config_running(dataset, models):
                print(f"Skipping configuration {models} as it is already running.")
                continue

            # The configurations have updated, see if this config is in the file_path
            if os.path.isfile(file_path):
                with open(file_path, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    for row in reader:
                        # Extract the models and fusion method
                        existing_models = row[2].split(',')
                        if tuple(existing_models) == models and row[3] == fusion_method:
                            print(f"Skipping configuration {models} with fusion method {fusion_method} as it already exists in results.")
                            continue
            
            print(f"\nEvaluating configuration: Models: {models}, Fusion method: {fusion_method}")
            
            # Write to .current-run directory
            current_run_dir = '.current-run'
            os.makedirs(current_run_dir, exist_ok=True)
            run_file_path = os.path.join(current_run_dir, f"{dataset}_{int(time.time())}.txt")
            with open(run_file_path, 'w') as f:
                f.write(','.join(models))
            print(f"Current run written to: {run_file_path}")

            # Initialize the BioFuse model
            biofuse_model = BioFuseModel(models, fusion_method=fusion_method, projection_dim=0)
            biofuse_model = biofuse_model.to("cuda")

            # Get the train embeddings
            embeddings = [train_embeddings_cache[model].to("cuda") for model in models]
            fused_embeddings = biofuse_model(embeddings)

            # Get the validation embeddings
            val_embeddings = [val_embeddings_cache[model].to("cuda") for model in models]
            val_fused_embeddings = biofuse_model(val_embeddings)

            start = time.time()
            # Compute validation accuracy using standalone_eval
            val_accuracy, val_auc_roc, test_acc, test_auc, xgb_train_time, xgb_val_inf_time = standalone_eval(models, 
                                                                            biofuse_model, 
                                                                            train_embeddings_cache, 
                                                                            train_labels, 
                                                                            val_embeddings_cache, 
                                                                            val_labels, 
                                                                            test_embeddings_cache, 
                                                                            test_labels, 
                                                                            num_classes,
                                                                            dataset,
                                                                            test_classifier_fn=get_classifier_fn(test_classifier))
            end = time.time()
            print(f"Time taken to evaluate configuration: {end - start:.2f} seconds")
            print(f"XGBoost training time for this configuration: {xgb_train_time:.2f} seconds")
            print(f"XGBoost validation inference time for this configuration: {xgb_val_inf_time:.2f} seconds")
            
            # Accumulate total times
            total_xgb_eval_time += xgb_train_time
            total_xgb_val_inf_time += xgb_val_inf_time
            
            # early exit
            #sys.exit(0)
            
            #harmonic_mean_val = weighted_mean_with_penalty(val_accuracy, val_auc_roc)

            # get xgb params
            xgb_params = get_xgb_params(len(train_embeddings_cache[models[0]]))
            n_estimators = xgb_params['n_estimators']

            # Save this result with XGBoost timing metrics
            append_results_to_csv(dataset, img_size, models, fusion_method, 0, 1, val_accuracy, val_auc_roc, test_acc, test_auc, n_estimators, xgb_train_time, xgb_val_inf_time)
            
            # Clean up the current run file
            if os.path.exists(run_file_path):
                os.remove(run_file_path)
                #print(f"Removed current run file: {run_file_path}")

            # if val_accuracy > best_val_acc:            
            #     best_val_acc = val_accuracy
            #     best_config = (models, 0, fusion_method)
            #     best_val_auc_roc = val_auc_roc                

    # Print summary of XGBoost times across all configurations
    print("\n----- XGBoost Timing Summary -----")
    print(f"Total XGBoost training time (xgb_eval_time): {total_xgb_eval_time:.2f} seconds")
    print(f"Total XGBoost validation inference time (xgb_eval_time_inf): {total_xgb_val_inf_time:.2f} seconds")
    
    # Add a check to prevent division by zero
    total_combinations = len(configurations) * len(fusion_methods)
    if total_combinations > 0:
        print(f"Average XGBoost training time per combination: {total_xgb_eval_time / total_combinations:.2f} seconds")
        print(f"Average XGBoost validation inference time per combination: {total_xgb_val_inf_time / total_combinations:.2f} seconds")
    else:
        print("No combinations were evaluated")
    print("----------------------------------")

    # print(f"\nBest configuration from first pass: Models: {best_config[0]}, Fusion method: {best_config[2]}")
    # print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    # print(f"Best Validation AUC-ROC: {best_val_auc_roc:.4f}")

    # Only do the second pass if additional projection dims were passed
    if len(projection_dims) == 1 and projection_dims[0] == 0:
        print("\nNo projection dims provided for the second")
        # save final results
        #append_results_to_csv(dataset, img_size, best_config[0], best_config[2], best_config[1], num_epochs, best_val_acc, best_val_auc, best_test_acc, best_test_auc)
        if ood_test_set:
            print(f"\nEvaluating on OOD test set: {ood_test_set}")
            ood_root = ood_data_root if ood_data_root else data_root
            _, _, ood_dataloader, ood_num_classes = load_data(ood_test_set, img_size, data_root=ood_root)
            
            # Print size of ood_dataloader
            print(f"Number of OOD test samples: {len(ood_dataloader.dataset)}")
            
            # Use extract_all_embeddings() instead of extract_and_cache_embeddings()
            # Create a dataloader dict with only test split
            ood_dataloaders = {'train': None, 'val': None, 'test': ood_dataloader}
            
            # Extract embeddings for OOD test set
            _, _, _, _, ood_embeddings_cache, ood_labels = extract_all_embeddings(
                model_names, 
                ood_dataloaders, 
                ood_test_set, 
                img_size, 
                nocache=True
            )
            
            # Get the best model from the first pass
            # For now, just use the last configuration
            best_models = models
            best_fusion_method = fusion_method
            
            biofuse_model = BioFuseModel(best_models, fusion_method=best_fusion_method, projection_dim=0)
            biofuse_model = biofuse_model.to("cuda")
            
            _, _, ood_test_acc, ood_test_auc, _, _ = standalone_eval(
                best_models,
                biofuse_model,
                train_embeddings_cache,
                train_labels,
                None,
                None,
                ood_embeddings_cache,
                ood_labels,
                ood_num_classes,
                ood_test_set,
                test_classifier_fn=get_classifier_fn(test_classifier)
            )
            
            print(f"OOD Test Accuracy: {ood_test_acc:.4f}")
            print(f"OOD Test AUC-ROC: {ood_test_auc:.4f}")
        return

    # Second pass: Train with learnable layers using the best configuration
    print("\nSecond pass: Training with learnable layers")
    best_models, _, best_fusion_method = best_config
    # remove 0 projection dim from the list
    projection_dims_second_pass = projection_dims[1:]
    
    for projection_dim in projection_dims_second_pass:
        print(f"\nTraining configuration: Models: {best_models}, Projection dim: {projection_dim}, Fusion method: {best_fusion_method}")

        # Initialize the BioFuse model
        biofuse_model = BioFuseModel(best_models, fusion_method=best_fusion_method, projection_dim=projection_dim)
        biofuse_model = biofuse_model.to("cuda")

        # Set up the classifier
        input_dim = projection_dim if fusion_method != 'concat' else projection_dim * len(best_models)
        print(f"Input dim: {input_dim}")
        
        output_dim = 1 if num_classes == 2 else num_classes
        classifier = LogisticRegression2(input_dim, output_dim).to("cuda")

        optimizer = optim.Adam(list(biofuse_model.parameters()) + list(classifier.parameters()), lr=0.004)
        criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

        best_epoch_val_acc = 0.0
        patience = PATIENCE
        patience_counter = 0

        for epoch in range(num_epochs):
            biofuse_model.train()
            classifier.train()
            
            # Train
            optimizer.zero_grad()
            embeddings = [train_embeddings_cache[model].to("cuda") for model in best_models]
            fused_embeddings = biofuse_model(embeddings)
            logits = classifier(fused_embeddings)
            
            labels = train_labels.to("cuda")
            if num_classes == 2:
                loss = criterion(logits, labels.unsqueeze(1).float())
            else:
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            # Validate
            biofuse_model.eval()
            classifier.eval()
            with torch.no_grad():
                val_embeddings = [val_embeddings_cache[model].to("cuda") for model in best_models]
                val_fused_embeddings = biofuse_model(val_embeddings)
                val_logits = classifier(val_fused_embeddings)
                
                val_labels = val_labels.to("cuda")
                if num_classes == 2:
                    val_predictions = (torch.sigmoid(val_logits) > 0.5).float()
                else:
                    val_predictions = torch.argmax(val_logits, dim=1)
                
                val_accuracy = (val_predictions.squeeze() == val_labels).float().mean()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

            if val_accuracy > best_epoch_val_acc:
                best_epoch_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Evaluate on validation set
        biofuse_model.eval()
        classifier.eval()

        # Compute validation accuracy using standalone_eval
        val_accuracy, val_auc_roc, test_acc, test_auc, xgb_train_time, xgb_val_inf_time = standalone_eval(best_models, biofuse_model, train_embeddings_cache, train_labels, val_embeddings_cache, val_labels, test_embeddings_cache, test_labels, num_classes, dataset, test_classifier_fn=get_classifier_fn(test_classifier))
        #harmonic_mean_val = weighted_mean_with_penalty(val_accuracy, val_auc_roc)

        # Save this result
        append_results_to_csv(dataset, img_size, best_models, best_fusion_method, projection_dim, epoch+1, val_accuracy, val_auc_roc, test_acc, test_auc, 0, xgb_train_time, xgb_val_inf_time)#, harmonic_mean_val)

        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_config = (best_models, projection_dim, best_fusion_method)
            best_val_auc_roc = val_auc_roc
           

    print(f"\nBest overall configuration: Models: {best_config[0]}, Projection dim: {best_config[1]}, Fusion method: {best_config[2]}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation AUC-ROC: {best_val_auc_roc:.4f}")

    # Compute test accuracy for the best configuration
    best_biofuse_model = BioFuseModel(best_config[0], fusion_method=best_config[2], projection_dim=best_config[1])
    best_biofuse_model = best_biofuse_model.to("cuda")
    best_val_acc, best_val_auc, best_test_acc, best_test_auc_roc, xgb_train_time, xgb_val_inf_time = standalone_eval(best_config[0], best_biofuse_model, train_embeddings_cache, train_labels, val_embeddings_cache, val_labels, test_embeddings_cache, test_labels, num_classes, dataset, test_classifier_fn=get_classifier_fn(test_classifier))
    #best_harmonic_mean = harmonic_mean(best_val_acc, best_val_auc)

    print(f"Test Accuracy for Best Configuration: {best_test_acc:.4f}")
    print(f"Test AUC-ROC for Best Configuration: {best_test_auc_roc:.4f}")

    # Save final results
    append_results_to_csv(dataset, img_size, best_config[0], best_config[2], best_config[1], num_epochs, best_val_acc, best_val_auc, best_test_acc, best_test_auc_roc, 0, xgb_train_time, xgb_val_inf_time)


def append_results_to_csv(dataset, img_size, model_names, fusion_method, projection_dim, epochs, val_accuracy, val_auc, test_accuracy, test_auc, n_estimators, xgb_train_time=0, xgb_val_inf_time=0):
    """
    Appends the results to a CSV file.

    Args:
    - dataset: The name of the dataset.
    - img_size: The image size.
    - model_names: The list of pre-trained models used in the BioFuse model.
    - fusion_method: The fusion method used in the BioFuse model.
    - projection_dim: The dimension of the projection layer.
    - epochs: The number of training epochs.
    - val_accuracy: The validation accuracy (or tuple of (top1, top5) for ImageNet).
    - val_auc: The validation AUC-ROC score.
    - test_accuracy: The test accuracy (or tuple of (top1, top5) for ImageNet).
    - test_auc: The test AUC-ROC score.
    - n_estimators: Number of estimators used in XGBoost.
    - xgb_train_time: XGBoost model training time in seconds.
    - xgb_val_inf_time: XGBoost validation inference time in seconds.

    Returns:
    - None
    """
    #file_path = f"results_{dataset}_{img_size}_est{n_estimators}.csv"
    file_path = f"results_{dataset}_{img_size}.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            if dataset in ['imagenet', 'imagenet-mini']:
                writer.writerow(['Dataset', 'Image Size', 'Models', 'Fusion Method', 'Projection Dim', 'Epochs', 
                               'Val Top-1 Acc', 'Val Top-5 Acc', 'Test Top-1 Acc', 'Test Top-5 Acc', 'XGBoost Train Time', 'XGBoost Val Inference Time'])
            else:
                writer.writerow(['Dataset', 'Image Size', 'Models', 'Fusion Method', 'Projection Dim', 'Epochs', 
                               'Val Accuracy', 'Val AUC-ROC', 'Test Accuracy', 'Test AUC-ROC', 'XGBoost Train Time', 'XGBoost Val Inference Time'])

        # Handle ImageNet results differently
        if dataset in ['imagenet', 'imagenet-mini']:
            val_top1, val_top5 = val_accuracy if isinstance(val_accuracy, tuple) else (0, 0)
            test_top1, test_top5 = test_accuracy if isinstance(test_accuracy, tuple) else (0, 0)
            writer.writerow([dataset, img_size, ','.join(model_names), fusion_method, projection_dim, epochs,
                           f'{val_top1:.4f}', f'{val_top5:.4f}', f'{test_top1:.4f}', f'{test_top5:.4f}', 
                           f'{xgb_train_time:.2f}', f'{xgb_val_inf_time:.2f}'])
        else:
            # replace Nones with 0 for test or val accuracy
            if val_accuracy is None:
                val_accuracy = 0
                val_auc = 0
            if test_accuracy is None:
                test_accuracy = 0
                test_auc = 0
            
            writer.writerow([dataset, img_size, ','.join(model_names), fusion_method, projection_dim, epochs,
                           f'{val_accuracy:.4f}', f'{val_auc:.4f}', f'{test_accuracy:.4f}', f'{test_auc:.4f}', 
                           f'{xgb_train_time:.2f}', f'{xgb_val_inf_time:.2f}'])

def parse_projections(proj_str):
    if proj_str:
        # add 0 as the first projection dimension
        proj_str = '0,' + proj_str
        return list(map(int, proj_str.split(',')))
    return []

def save_test_predictions(predictions, filenames, models):
    id_mapping = {}
    with open('idx_to_ILSVRC_ID.csv', 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            pred_id, ilsvrc_id = row[0], row[1]
            id_mapping[int(pred_id)] = ilsvrc_id
    
    # Create model name from the list of models
    model_name = '_'.join(models)
    timestamp = datetime.now().strftime('%d%m%y%H%M%S')
    submission_file = f"imagenet-submissions/{model_name}_submission_{timestamp}.txt"    
    
    print(f"Saving predictions to {submission_file}")
    
    with open(submission_file, 'w') as f:
        for pred in predictions:
            # Get top 5 predictions (indices)
            top5_indices = np.argsort(pred)[-5:][::-1]
            
            # Map indices to ILSVRC IDs
            top5_ilsvrc = [id_mapping[idx] for idx in top5_indices]

            # Write to file: space-separated ILSVRC IDs
            f.write(f"{' '.join(top5_ilsvrc)}\n")

    
def save_val_predictions(predictions, labels, models):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_str = '_'.join(models)
    filename = f'predictions_{models_str}_{timestamp}_val.txt'
    
    with open(filename, 'w') as f:
        for i, pred in enumerate(predictions):
            # Get label (assuming it's a filename or ID)
            label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
            
            # Get top 5 class indices
            top5_classes = np.argsort(pred)[-5:][::-1]
            
            # Write as space-separated values without tensor formatting
            f.write(f"{' '.join(map(str, top5_classes))}\n")
            
    print(f"Saved validation predictions to {filename}")

def get_classifier_fn(classifier_name):
    """Maps classifier name to its training function and parameters"""
    classifiers = {
        'xgb': train_classifier2,
        'cat': train_catboost_classifier,
        'nn_mlp': lambda *args: train_nn_classifier(*args, architecture='mlp'),
        'nn_cnn': lambda *args: train_nn_classifier(*args, architecture='cnn'),
        'nn_resnet': lambda *args: train_nn_classifier(*args, architecture='resnet'),
        'nn_cnn_adv': lambda *args: train_nn_classifier(*args, architecture='cnn_adv'),
    }
    return classifiers.get(classifier_name, train_classifier2)

def main():
    model_legend = { 'BC': 'BioMedCLIP',
                    'PC': 'PubMedCLIP',
                    'CO': 'CONCH',
                    'RD': 'rad-dino',
                    'UN': 'UNI',
                    'PG': 'Prov-GigaPath',
                    'HB': 'Hibou-B',
                    'CA': 'CheXagent',
                    'UN2': 'UNI2',}
    parser = argparse.ArgumentParser(description='BioFuse v1.1 (AutoFuse)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--dataset', type=str, default='breastmnist', help='Dataset')
    parser.add_argument('--models', type=str, default='BioMedCLIP', help='List of pre-trained models, delimited by comma')
    parser.add_argument('--projections', type=parse_projections, default=[0], help='List of projection dimensions, delimited by comma')
    parser.add_argument('--fusion_methods', type=str, default='concat', help='Fusion methods separated by comma')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    # add --single
    parser.add_argument('--single', action='store_true', help='Run the model with a single specified configuration')
    parser.add_argument('--nocache', action='store_true', help='Disable use of cached embeddings')
    parser.add_argument('--data_root', type=str, help='Root directory for dataset storage (required for ImageNet)')
    parser.add_argument('--test_classifier', type=str, default='xgb',
                   choices=['xgb', 'cat', 'nn_mlp', 'nn_cnn', 'nn_resnet', 'nn_cnn_adv'],
                   help='Classifier to use for test set')
    parser.add_argument('--ood_test_set', type=str, help='Out-of-distribution test set')
    parser.add_argument('--ood_data_root', type=str, help='Root directory for OOD dataset storage')
    parser.add_argument('--params_json', type=str, help='Path to a JSON file with parameters')
    parser.add_argument('--test-noise', action='store_true', help='Evaluate robustness on MedMNIST-C corruptions')
    parser.add_argument('--medmnistc-root', type=str, default='/data/medmnist-c', help='Root directory containing <dataset>/<corruption>.npz files')
    global args # make args global so it can be accessed in standalone_eval
    args = parser.parse_args()
    

    if args.params_json:
        with open(args.params_json, 'r') as f:
            params = json.load(f)
            # Overwrite args with params from JSON
            for key, value in params.items():
                setattr(args, key, value)
            args.single = True
            if 'models' in params:
                args.models = ','.join([model_legend.get(m, m) for m in params['models'].split(',')])

    # set nocache to True
    #args.nocache = True
    train_model(args.dataset, 
                args.models.split(','), 
                args.num_epochs, 
                args.img_size,
                args.projections,
                args.fusion_methods.split(','),
                args.single,
                args.nocache,
                args.data_root,
                test_classifier=args.test_classifier,
                ood_test_set=args.ood_test_set,
                ood_data_root=args.ood_data_root,
                params=vars(args),
                batch_size=args.batch_size)
    
if __name__ == "__main__":
    main()
