import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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
import random
import argparse
#import ipdb
import itertools
import os
import pickle

# Trainable layer imports
import torch.optim as optim
import torch.nn as nn

# Preprocessor
from biofuse.models.processor import MultiModelPreprocessor

PATIENCE = 25
CACHE_DIR = '/data/biofuse-embedding-cache'

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
    
class MLPClassifier(nn.Module):
    """
    Implements a Multi-Layer Perceptron (MLP) for classification.

    This model consists of a simple feedforward neural network with one
    hidden layer and a ReLU activation function. It can be used for binary
    or multi-class classification tasks.

    Attributes:
        mlp (nn.Sequential): The sequential container of layers forming the MLP.

    Parameters:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Number of neurons in the hidden layer.
        output_dim (int): Number of output classes, typically 1 for binary classification.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1): # Assuming binary classification
        super(MLPClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

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

def load_data(dataset, img_size, train=True):
    """
    Load data for a given dataset.

    Args:
        dataset (str): The name of the dataset.
        img_size (int): The desired image size.
        train (bool, optional): Whether to load the training data. Defaults to True.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, test data loader, and the number of classes.
    """
    print(f"Loading data for {dataset}...")
    
    # Get dataset information
    info = INFO[dataset]
    num_classes = len(info['label'])

    # Get the data class based on the dataset name
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load the datasets
    train_dataset = DataClass(split='train', download=True, size=img_size, root='/data/medmnist')
    val_dataset = DataClass(split='val', download=True, size=img_size, root='/data/medmnist')
    test_dataset = DataClass(split='test', download=True, size=img_size, root='/data/medmnist')
    
    # Set the image paths based on the image size
    if img_size == 28:
        train_images_path = f'/data/medmnist/{dataset}_train/{dataset}'
        val_images_path = f'/data/medmnist/{dataset}_val/{dataset}'
        test_images_path = f'/data/medmnist/{dataset}_test/{dataset}'
    else:
        train_images_path = f'/data/medmnist/{dataset}_train/{dataset}_{img_size}'
        val_images_path = f'/data/medmnist/{dataset}_val/{dataset}_{img_size}'
        test_images_path = f'/data/medmnist/{dataset}_test/{dataset}_{img_size}'

    # Save the datasets if the image paths don't exist
    if not os.path.exists(train_images_path):
        train_dataset.save(f'/data/medmnist/{dataset}_train')
    
    if not os.path.exists(val_images_path):
        val_dataset.save(f'/data/medmnist/{dataset}_val')
    
    if not os.path.exists(test_images_path):
        test_dataset.save(f'/data/medmnist/{dataset}_test')
    
    # Construct image paths by globbing the directory
    train_image_paths = glob.glob(f'{train_images_path}/*.png')
    val_image_paths = glob.glob(f'{val_images_path}/*.png')
    test_image_paths = glob.glob(f'{test_images_path}/*.png')

    if dataset == "chestmnist":
        train_labels = [parse_labels_from_path(path) for path in train_image_paths]
        val_labels = [parse_labels_from_path(path) for path in val_image_paths]
        test_labels = [parse_labels_from_path(path) for path in test_image_paths]        
    else:
        # Labels are just _0.png or _1.png etc
        train_labels = [int(path.split('_')[-1].split('.')[0]) for path in train_image_paths]
        val_labels = [int(path.split('_')[-1].split('.')[0]) for path in val_image_paths]
        test_labels = [int(path.split('_')[-1].split('.')[0]) for path in test_image_paths]

    
    # Construct the datasets
    full_train_dataset = BioFuseImageDataset(train_image_paths, train_labels)
    val_dataset = BioFuseImageDataset(val_image_paths, val_labels)
    test_dataset = BioFuseImageDataset(test_image_paths, test_labels)

    # Function to get a balanced subset of the dataset
    def get_balanced_subset(dataset, num_samples):
        class_counts = {}
        for _, label in dataset:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        samples_per_class = num_samples // len(class_counts)
        balanced_indices = []
        for label in class_counts:
            label_indices = [i for i, (_, l) in enumerate(dataset) if l == label]
            balanced_indices.extend(random.sample(label_indices, min(samples_per_class, len(label_indices))))
        
        return Subset(dataset, balanced_indices)

    # if train and len(full_train_dataset) > 5000:
    #     # Get balanced subsets
    #     train_dataset = get_balanced_subset(full_train_dataset, 5000)
    #     val_dataset = get_balanced_subset(val_dataset, 1000)
    #     test_dataset = get_balanced_subset(test_dataset, 1000)
    # else:
    train_dataset = full_train_dataset
    
    # Use a smaller sample
    # train_dataset = Subset(train_dataset, range(200))
    # val_dataset = Subset(val_dataset, range(50))
    # test_dataset = Subset(test_dataset, range(50))

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
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

def train_classifier2(features, labels, num_classes, multi_label=False):
    """
    Trains an XGBoost classifier using the provided features and labels.

    Args:
        features (array-like): The input features for training the classifier.
        labels (array-like): The corresponding labels for the input features.
        num_classes (int): The number of classes in the classification problem.

    Returns:
        None
    """
    print("Training XGBoost classifier...")

    scaler = StandardScaler()
    
    # Scale features
    features = scaler.fit_transform(features)
    # record time for training
    import time
    start = time.time()

    if num_classes > 2 and not multi_label:
        print("Multi-class classification")
        classifier = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            n_estimators=50,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=8,
            tree_method='hist'           
        )
    else:
        print("Binary classification")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=8,
            tree_method='hist'          
        )

        if multi_label:
            classifier = OneVsRestClassifier(xgb_model)
        else:
            classifier = xgb_model               

    classifier.fit(features, labels)

    end = time.time()
    print(f"Time taken to train XGBoost classifier: {end - start:.2f} seconds")

    return classifier, scaler

def evaluate_model(classifier, features, labels):
    """
    Evaluates the classifier on the given features and labels.

    Args:
    - classifier: The trained classifier.
    - features: The input features for evaluation.
    - labels: The target labels for evaluation.

    Returns:
    - float: The accuracy of the classifier on the given features and labels.
    """
    print("Evaluating model...")
    predictions = classifier.predict(features)
    return accuracy_score(labels, predictions)

# method to compute AUC-ROC for binary or multi-class classification
def compute_auc_roc(classifier, features, labels, num_classes):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for the classifier.

    Args:
    - classifier: The trained classifier.
    - features: The input features for evaluation.
    - labels: The target labels for evaluation.

    Returns:
    - float: The AUC-ROC score of the classifier on the given features and labels.
    """
    print("Computing AUC-ROC...")
    if num_classes == 2:
        predictions = classifier.predict_proba(features)[:, 1]
        return roc_auc_score(labels, predictions)
    else:
        # use one-vs-all strategy
        predictions = classifier.predict_proba(features)
        return roc_auc_score(labels, predictions, multi_class='ovr')
    
def standalone_eval(models, biofuse_model, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels, num_classes, dataset, classifier, scaler=None): 
    """
    Standalone evaluation of the BioFuse model using cached embeddings.

    Args:
    - biofuse_model: The trained BioFuse model.
    - train_embeddings: Cached train embeddings.
    - train_labels: Cached train labels.
    - val_embeddings: Cached validation embeddings (can be None).
    - val_labels: Cached validation labels (can be None).
    - test_embeddings: Cached test embeddings (can be None).
    - test_labels: Cached test labels (can be None).
    - num_classes: Number of classes in the dataset.

    Returns:
    - tuple: (val_accuracy, val_auc_roc, test_accuracy, test_auc_roc)
             Returns None for metrics if corresponding data is not provided.
    """   
    biofuse_model.eval()

    multi_label = False
    if dataset == "chestmnist":
        multi_label = True

    val_accuracy, val_auc_roc = None, None
    new_classifier = classifier
    with torch.no_grad():
        if new_classifier is None:
            # Process train embeddings
            train_fused_embeddings = biofuse_model([train_embeddings[model].to("cuda") for model in models])
            train_fused_embeddings_np = train_fused_embeddings.cpu().numpy()
            if multi_label:
                train_labels = train_labels.view(-1,14)

            train_labels_np = train_labels.cpu().numpy()
                
            # Train a new classifier on fused embeddings
            new_classifier, scaler = train_classifier2(train_fused_embeddings_np, train_labels_np, num_classes, multi_label)

            val_accuracy, val_auc_roc = None, None
            if val_embeddings is not None and val_labels is not None:
                # Process val embeddings
                val_fused_embeddings = biofuse_model([val_embeddings[model].to("cuda") for model in models])
                val_fused_embeddings_np = val_fused_embeddings.cpu().numpy()
                if multi_label:
                    val_labels = val_labels.view(-1,14)
                val_labels_np = val_labels.cpu().numpy()
                
                # Scale val embeddings
                val_fused_embeddings_np = scaler.transform(val_fused_embeddings_np)

                # Evaluate on validation set
                val_accuracy = evaluate_model(new_classifier, val_fused_embeddings_np, val_labels_np)
                val_auc_roc = compute_auc_roc(new_classifier, val_fused_embeddings_np, val_labels_np, num_classes)

                print(f"Validation Accuracy: {val_accuracy:.4f}")
                print(f"Validation AUC-ROC: {val_auc_roc:.4f}")

        test_accuracy, test_auc_roc = None, None
        if test_embeddings is not None and test_labels is not None:
            # Process test embeddings
            test_fused_embeddings = biofuse_model([test_embeddings[model].to("cuda") for model in models])
            test_fused_embeddings_np = test_fused_embeddings.cpu().numpy()
            if multi_label:
                test_labels = test_labels.view(-1,14)
            test_labels_np = test_labels.cpu().numpy()
            
            # Scale test embeddings
            test_fused_embeddings_np = scaler.transform(test_fused_embeddings_np)

            # Evaluate on test set
            test_accuracy = evaluate_model(new_classifier, test_fused_embeddings_np, test_labels_np)
            test_auc_roc = compute_auc_roc(new_classifier, test_fused_embeddings_np, test_labels_np, num_classes)

            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test AUC-ROC: {test_auc_roc:.4f}")

    return val_accuracy, val_auc_roc, test_accuracy, test_auc_roc, new_classifier, scaler

def extract_and_cache_embeddings(dataloader, models, dataset, img_size, split):
    cached_embeddings = {model: [] for model in models}
    labels = []
    
    for model in models:
        cached_data = load_embeddings_from_cache(dataset, model, img_size, split)
        if cached_data is not None:
            cached_embeddings[model], labels = cached_data
            print(f"Loaded cached embeddings for {model} ({split})")
            continue

        extractor = PreTrainedEmbedding(model)
        preprocessor = MultiModelPreprocessor([model])

        model_embeddings = []
        model_labels = []

        for image, label in tqdm(dataloader, desc=f"Extracting embeddings for {model} ({split})"):
            processed_image = preprocessor.preprocess(image[0])[0]
            with torch.no_grad():
                embeddings = extractor(processed_image)
            model_embeddings.append(embeddings.squeeze(0))
            model_labels.append(label)

        cached_embeddings[model] = torch.stack(model_embeddings)
        
        #Only update labels if they haven't been set yet
        if len(labels) == 0:
            if isinstance(model_labels[0], torch.Tensor) and model_labels[0].dim() > 0:
                labels = torch.stack(model_labels)
            else:
                labels = torch.tensor(model_labels)
       

        save_embeddings_to_cache(cached_embeddings[model], labels, dataset, model, img_size, split)
        print(f"Saved embeddings for {model} ({split}) to cache")

    return cached_embeddings, labels

def harmonic_mean(val_acc, val_auc):
    return 2 / ((1 / val_acc) + (1 / val_auc))

def weighted_mean(val_acc, val_auc):
    weights = [0.4, 0.6]        
    score = weights[0] * val_acc + weights[1] * val_auc
    return score

def weighted_mean_with_penalty(val_acc, val_auc):
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

def get_configurations(model_names, file_path, single):
    # if single, there is only a single configuration with all models
    if single:
        return [tuple(model_names)]
    
    # Generate all combinations of pre-trained models
    configurations = []
    for r in range(1, len(model_names) + 1):
        configurations.extend(itertools.combinations(model_names, r))

    # Print the size of the configurations
    print(f"Number of configurations: {len(configurations)}")    

    # # Check if the results file exists
    # if os.path.isfile(file_path):
    #     # Read all rows, ignore first row
    #     with open(file_path, mode='r') as file:
    #         reader = csv.reader(file)
    #         next(reader)
    #         for row in reader:
    #             # Extract the models and fusion method
    #             models = row[2].split(',')                
                
    #             # Remove the models from the configurations
    #             if tuple(models) in configurations:
    #                 configurations.remove(tuple(models))

    #     # Print the new configurations size 
    #     print(f"Number of configurations after removing existing results: {len(configurations)}")

    return configurations
        
# Training the model with validation-informed adjustment
def train_model(dataset, model_names, num_epochs, img_size, projection_dims, fusion_methods, single=False):
    set_seed(42)

    file_path = f"results_{dataset}_{img_size}.csv"
    configurations = get_configurations(model_names, file_path, single)
    print(configurations)

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_data(dataset, img_size)

    # Extract and cache embeddings
    print("Extracting and caching embeddings...")
    train_embeddings_cache, train_labels = extract_and_cache_embeddings(train_dataloader, model_names, dataset, img_size, 'train')
    val_embeddings_cache, val_labels = extract_and_cache_embeddings(val_dataloader, model_names, dataset, img_size, 'val')
    test_embeddings_cache, test_labels = extract_and_cache_embeddings(test_dataloader, model_names, dataset, img_size, 'test')

    best_config = None
    best_val_acc = 0
    best_test_acc = 0
    best_val_auc_roc = 0
    best_test_auc_roc = 0
    best_harmonic_mean = 0
    classifier = None
    scaler = None

    # First pass: Evaluate configurations with a single epoch
    print("\nFirst pass: Evaluating model combinations")
    for models in configurations:
        for fusion_method in fusion_methods:
            print(f"\nEvaluating configuration: Models: {models}, Fusion method: {fusion_method}")

            # Initialize the BioFuse model
            biofuse_model = BioFuseModel(models, fusion_method=fusion_method, projection_dim=0)
            biofuse_model = biofuse_model.to("cuda")

            # Get the train embeddings
            embeddings = [train_embeddings_cache[model].to("cuda") for model in models]
            fused_embeddings = biofuse_model(embeddings)

            # Get the validation embeddings
            val_embeddings = [val_embeddings_cache[model].to("cuda") for model in models]
            val_fused_embeddings = biofuse_model(val_embeddings)

            # Compute validation accuracy using standalone_eval
            val_accuracy, val_auc_roc, _, _, new_classifier, new_scaler = standalone_eval(models, 
                                                                            biofuse_model, 
                                                                            train_embeddings_cache, 
                                                                            train_labels, 
                                                                            val_embeddings_cache, 
                                                                            val_labels, 
                                                                            None, 
                                                                            None, 
                                                                            num_classes,
                                                                            dataset,
                                                                            None)
            harmonic_mean_val = weighted_mean_with_penalty(val_accuracy, val_auc_roc)

            # Save this result
            append_results_to_csv(dataset, img_size, models, fusion_method, 0, 1, val_accuracy, val_auc_roc, 0, 0, harmonic_mean_val)

            #if val_accuracy > best_val_acc:
            if harmonic_mean_val > best_harmonic_mean:
                best_val_acc = val_accuracy
                best_config = (models, 0, fusion_method)
                best_val_auc_roc = val_auc_roc
                best_harmonic_mean = harmonic_mean_val
                classifier = new_classifier        
                scaler = new_scaler      

    
    print(f"\nBest overall configuration: Models: {best_config[0]}, Projection dim: {best_config[1]}, Fusion method: {best_config[2]}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation AUC-ROC: {best_val_auc_roc:.4f}")

    # Compute test accuracy for the best configuration
    best_biofuse_model = BioFuseModel(best_config[0], fusion_method=best_config[2], projection_dim=best_config[1])
    best_biofuse_model = best_biofuse_model.to("cuda")
    _, _, best_test_acc, best_test_auc_roc, _, _ = standalone_eval(best_config[0], best_biofuse_model, train_embeddings_cache, train_labels, val_embeddings_cache, val_labels, test_embeddings_cache, test_labels, num_classes, dataset, classifier, scaler)
    #best_harmonic_mean = harmonic_mean(best_val_acc, best_val_auc)

    print(f"Test Accuracy for Best Configuration: {best_test_acc:.4f}")
    print(f"Test AUC-ROC for Best Configuration: {best_test_auc_roc:.4f}")

    # Save final results
    append_results_to_csv(dataset, img_size, best_config[0], best_config[2], best_config[1], num_epochs, best_val_acc, best_val_auc_roc, best_test_acc, best_test_auc_roc)


def append_results_to_csv(dataset, img_size, model_names, fusion_method, projection_dim, epochs, val_accuracy, val_auc, test_accuracy, test_auc, harmonic_mean_val=0):
    """
    Appends the results to a CSV file.

    Args:
    - dataset: The name of the dataset.
    - img_size: The image size.
    - model_names: The list of pre-trained models used in the BioFuse model.
    - fusion_method: The fusion method used in the BioFuse model.
    - projection_dim: The dimension of the projection layer.
    - epochs: The number of training epochs.
    - val_accuracy: The validation accuracy.
    - val_auc: The validation AUC-ROC score.
    - test_accuracy: The test accuracy.
    - test_auc: The test AUC-ROC score.

    Returns:
    - None
    """
    file_path = f"results_{dataset}_{img_size}.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Dataset', 'Image Size', 'Models', 'Fusion Method', 'Projection Dim', 'Epochs', 'Val Accuracy', 'Val AUC-ROC', 'Test Accuracy', 'Test AUC-ROC'])

        # replace Nones with 0 for test or val accuracy
        if val_accuracy is None:
            val_accuracy = 0
            val_auc = 0
        if test_accuracy is None:
            test_accuracy = 0
            test_auc = 0

        
        #writer.writerow([dataset, img_size, ','.join(model_names), fusion_method, projection_dim, epochs, f'{val_accuracy:.3f}', f'{val_auc:.3f}', f'{test_accuracy:.3f}', f'{test_auc:.3f}', f'{harmonic_mean_val:.3f}'])
        # write the results as XX.YY insead of 0.XYZ
        writer.writerow([dataset, img_size, ','.join(model_names), fusion_method, projection_dim, epochs, f'{val_accuracy:.4f}', f'{val_auc:.4f}', f'{test_accuracy:.4f}', f'{test_auc:.4f}', f'{harmonic_mean_val:.4f}'])

def parse_projections(proj_str):
    if proj_str:
        # add 0 as the first projection dimension
        proj_str = '0,' + proj_str
        return list(map(int, proj_str.split(',')))
    return []

def main():
    parser = argparse.ArgumentParser(description='BioFuse v1.1 (AutoFuse)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--dataset', type=str, default='breastmnist', help='Dataset')
    parser.add_argument('--models', type=str, default='BioMedCLIP', help='List of pre-trained models, delimited by comma')
    parser.add_argument('--projections', type=parse_projections, default=[0], help='List of projection dimensions, delimited by comma')
    parser.add_argument('--fusion_methods', type=str, default='concat', help='Fusion methods separated by comma')
    # add --single
    parser.add_argument('--single', action='store_true', help='Run the model with a single specified configuration')
    args = parser.parse_args()

    train_model(args.dataset, 
                args.models.split(','), 
                args.num_epochs, 
                args.img_size,
                args.projections,
                args.fusion_methods.split(','),
                args.single)
    
if __name__ == "__main__":
    main()
