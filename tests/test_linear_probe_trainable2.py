import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
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

# Trainable layer imports
import torch.optim as optim
import torch.nn as nn

PATIENCE = 25

def set_seed(seed: int = 42) -> None:
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
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression2, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        #outputs = torch.sigmoid(self.linear(x))
        return self.linear(x)
    
class MLPClassifier(nn.Module):
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
    # Filter out None values
    batch = [(img, label) for img, label in batch if img is not None]
    
    if len(batch) == 0:
        return [], torch.tensor([])

    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def load_data(dataset, img_size, train=True):
    print(f"Loading data for {dataset}...")
    
    info = INFO[dataset]
    num_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', download=False, size=img_size, root='/data/medmnist')
    val_dataset = DataClass(split='val', download=False, size=img_size, root='/data/medmnist')
    test_dataset = DataClass(split='test', download=False, size=img_size, root='/data/medmnist')
    
    if img_size == 28:
        train_images_path = f'/data/medmnist/{dataset}_train/{dataset}'
        val_images_path = f'/data/medmnist/{dataset}_val/{dataset}'
        test_images_path = f'/data/medmnist/{dataset}_test/{dataset}'
    else:
        train_images_path = f'/data/medmnist/{dataset}_train/{dataset}_{img_size}'
        val_images_path = f'/data/medmnist/{dataset}_val/{dataset}_{img_size}'
        test_images_path = f'/data/medmnist/{dataset}_test/{dataset}_{img_size}'

    if not os.path.exists(train_images_path):
        train_dataset.save(f'/data/medmnist/{dataset}_train')
    
    if not os.path.exists(val_images_path):
        val_dataset.save(f'/data/medmnist/{dataset}_val')
    
    if not os.path.exists(test_images_path):
        test_dataset.save(f'/data/medmnist/{dataset}_test')
    
    # Construct image paths, glob directory
    train_image_paths = glob.glob(f'{train_images_path}/*.png')
    print(train_image_paths[0])
    val_image_paths = glob.glob(f'{val_images_path}/*.png')
    test_image_paths = glob.glob(f'{test_images_path}/*.png')

    if dataset == "chestmnist":
        # labels are multi-label test20092_0_0_0_0_1_0_0_0_0_0_0_0_0_0.png, remove the dirnames
        train_labels = [list(map(int, path.split('_')[2:-1])) for path in train_image_paths]
        train_labels.append([int(path.split('_')[-1].split('.')[0]) for path in train_image_paths])
        
        val_labels = [list(map(int, path.split('_')[2:-1])) for path in val_image_paths]
        val_labels.append([int(path.split('_')[-1].split('.')[0]) for path in val_image_paths])

        test_labels = [list(map(int, path.split('_')[2:-1])) for path in test_image_paths]
        test_labels.append([int(path.split('_')[-1].split('.')[0]) for path in test_image_paths])
    else:
        # Labels are just _0.png or _1.png etc
        train_labels = [int(path.split('_')[-1].split('.')[0]) for path in train_image_paths]
        val_labels = [int(path.split('_')[-1].split('.')[0]) for path in val_image_paths]
        test_labels = [int(path.split('_')[-1].split('.')[0]) for path in test_image_paths]

    #print(train_labels[0])  
    # Construct the datasets
    full_train_dataset = BioFuseImageDataset(train_image_paths, train_labels)
    val_dataset = BioFuseImageDataset(val_image_paths, val_labels)
    test_dataset = BioFuseImageDataset(test_image_paths, test_labels)

    # Function to get balanced subset
    def get_balanced_subset(dataset, num_samples):
        if isinstance(dataset[0][1], list):  # Multi-label case
            label_counts = {i: 0 for i in range(len(dataset[0][1]))}
            for _, labels in dataset:
                for i, label in enumerate(labels):
                    if label == 1:
                        label_counts[i] += 1
            
            samples_per_class = num_samples // len(label_counts)
            balanced_indices = []
            for label in label_counts:
                label_indices = [i for i, (_, labels) in enumerate(dataset) if labels[label] == 1]
                balanced_indices.extend(random.sample(label_indices, min(samples_per_class, len(label_indices))))
        else:  # Single-label case
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

    train = True
    if train and len(full_train_dataset) > 5000:
        # Get balanced subsets
        train_dataset = get_balanced_subset(full_train_dataset, 500)
        val_dataset = get_balanced_subset(val_dataset, 100)
        test_dataset = get_balanced_subset(test_dataset, 100)
    else:
        train_dataset = full_train_dataset

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader, test_loader, num_classes

def extract_features(dataloader, biofuse_model):
    print("Extracting features...")
    features = []
    labels = []    
    # use progress bar
    for image, label in tqdm(dataloader):
        embedding = biofuse_model(image)
        #features.append(embedding.squeeze(0).numpy())
        features.append(embedding.squeeze(0).detach().numpy())
        labels.append(label.numpy())
   
    return np.array(features), np.array(labels)
    
def print_cuda_mem_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.0f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.0f} MB")

def generate_embeddings(dataloader, biofuse_model, cache_raw_embeddings=False, is_training=True, is_test=False, progress_bar=False):
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
        print(f'label: {label}')
        labels.append(label)
    
    # Embeddings is a list of tensors, stack them and remove the batch dimension
    embeddings_tensor = torch.stack(embeddings).squeeze(1)
    labels_tensor = torch.tensor(labels)        
    
    return embeddings_tensor, labels_tensor

def print_trainable_parameters(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():   
        if param.requires_grad:
            print(name, param.shape)
            print(name, param.numel())

def log_projection_layer_weights(model, epoch, stage):
    for i, layer in enumerate(model.projection_layers):
        print(f"Epoch [{epoch}] - {stage} - Projection Layer {i} Weights:")
        for name, param in layer.named_parameters():  # Iterate through MLP parameters
            weights = param.data
            print(f"  - {name}: {weights.mean().item():.6f} ± {weights.std().item():.6f}")

def log_projection_layer_gradients(model, epoch, stage):
    for i, layer in enumerate(model.projection_layers):
        print(f"Epoch [{epoch}] - {stage} - Projection Layer {i} Gradients:")
        for name, param in layer.named_parameters():  # Iterate through MLP parameters
            if param.grad is not None:
                grad = param.grad.data
                print(f"  - {name}: {grad.mean().item():.6f} ± {grad.std().item():.6f}")
            else:
                print(f"  - {name}: None")

def train_classifier(features, labels, num_classes):
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

def train_classifier2(features, labels, num_classes):
    print("Training XGBoost classifier...")

    scaler = StandardScaler()
    
    # Scale features
    features = scaler.fit_transform(features)

    if num_classes > 2:
        classifier = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    else:
        classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    classifier.fit(features, labels)
    return classifier, scaler

def train_classifier3(features, labels, num_classes):
    print("Training ordinal regression classifier...")

    scaler = StandardScaler()
    
    # Scale features
    features = scaler.fit_transform(features)

    # Encode labels for ordinal regression
    encoder = OrdinalEncoder()
    labels = encoder.fit_transform(labels.reshape(-1, 1)).ravel()

    # multi-class classification using xgboost
    classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(encoder.categories_[0]),
        n_estimators=100,
        learning_rate=0.1,        
        eval_metric='mlogloss'
    )

    classifier.fit(features, labels)
    return classifier, scaler

def train_multi_label_classifier(features, labels):
    print("Training multi-label classifier...")
    scaler = StandardScaler()
    
    # Scale features
    features = scaler.fit_transform(features)

    classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    classifier.fit(features, labels)
    return classifier, scaler

def evaluate_model(classifier, features, labels, is_chestmnist=False):
    print("Evaluating model...")
    print(f"Labels 0: {labels[0]}")

    if is_chestmnist:
        # ChestMNIST case: multi-label binary classification
        y_score = classifier.predict_proba(features)
        y_pred = (y_score > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Calculate per-class accuracy
        per_class_accuracy = np.mean(labels == y_pred, axis=0)
        for i, acc in enumerate(per_class_accuracy):
            print(f"Accuracy for label {i}: {acc:.4f}")

    else:
        # Single-label case for other datasets
        y_pred = classifier.predict(features)
        avg_accuracy = accuracy_score(labels, y_pred)
        print(f"Accuracy: {avg_accuracy:.4f}")

    return avg_accuracy

# method to compute AUC-ROC for binary or multi-class classification
def compute_auc_roc(classifier, features, labels, num_classes, is_chestmnist=False):
    print("Computing AUC-ROC...")
    
    if is_chestmnist:
        # ChestMNIST case: multi-label binary classification
        y_score = classifier.predict_proba(features)
        
        # Compute AUC-ROC for each label
        auc_scores = []
        for i in range(labels.shape[1]):
            auc = roc_auc_score(labels[:, i], y_score[:, i])
            auc_scores.append(auc)
        
        avg_auc = np.mean(auc_scores)
        print(f"Average AUC-ROC: {avg_auc:.4f}")
        
        for i, auc in enumerate(auc_scores):
            print(f"AUC-ROC for label {i}: {auc:.4f}")

    else:
        # Single-label case for other datasets
        if labels.ndim == 1:  # Binary classification
            y_score = classifier.predict_proba(features)[:, 1]
            avg_auc = roc_auc_score(labels, y_score)
        else:  # Multi-class classification
            y_score = classifier.predict_proba(features)
            avg_auc = roc_auc_score(labels, y_score, multi_class='ovr')
        
        print(f"AUC-ROC: {avg_auc:.4f}")
    
    return avg_auc


def standalone_eval(dataset, img_size, biofuse, models, fusion_method, projection_dim):    
    # Load the data
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_data(dataset, img_size, train=False)

    # Extract features from the training set
    embeddings_tensor, labels_tensor = generate_embeddings(train_dataloader, biofuse, progress_bar=True)

    # convert to numpy
    embeddings_np = embeddings_tensor.cpu().detach().numpy()
    labels_np = labels_tensor.cpu().detach().numpy()

    if dataset == "chestmnist":
        classifier, scaler = train_classifier(embeddings_np, labels_np, num_classes)
    else:
        classifier, scaler = train_classifier2(embeddings_np, labels_np, num_classes)

    # Validation set evaluation
    val_embeddings_tensor, val_labels_tensor = generate_embeddings(val_dataloader, biofuse, progress_bar=True, is_training=False)
    val_embeddings_np = val_embeddings_tensor.cpu().detach().numpy()
    val_labels_np = val_labels_tensor.cpu().detach().numpy()
    val_embeddings_np = scaler.transform(val_embeddings_np)

    chestmnist = False
    if dataset == "chestmnist": chestmnist = True
        
    val_accuracy = evaluate_model(classifier, val_embeddings_np, val_labels_np, is_chestmnist=chestmnist)
    val_auc_roc = compute_auc_roc(classifier, val_embeddings_np, val_labels_np, num_classes, is_chestmnist=chestmnist)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC-ROC: {val_auc_roc:.4f}")

    # Test set evaluation
    test_embeddings_tensor, test_labels_tensor = generate_embeddings(test_dataloader, biofuse, progress_bar=True, is_test=True)
    test_embeddings_np = test_embeddings_tensor.cpu().detach().numpy()
    test_labels_np = test_labels_tensor.cpu().detach().numpy()
    test_embeddings_np = scaler.transform(test_embeddings_np)

    test_accuracy = evaluate_model(classifier, test_embeddings_np, test_labels_np, is_chestmnist=chestmnist)
    test_auc_roc = compute_auc_roc(classifier, test_embeddings_np, test_labels_np, num_classes, is_chestmnist=chestmnist)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC-ROC: {test_auc_roc:.4f}")

    return val_accuracy, val_auc_roc, test_accuracy, test_auc_roc

        
# Training the model with validation-informed adjustment
def train_model(dataset, model_names, num_epochs, img_size, projection_dim, fusion_method, fast_run):
    set_seed(42)

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_data(dataset, img_size)
    #sys.exit(0)

    #model_names = ["CheXagent"] # CheXagent needs a bigger GPU :/
    #model_names =  ["rad-dino"] 
    #model_names =  ["BioMedCLIP"]
    #model_names = ["PubMedCLIP"]
    #model_names = ["BioMedCLIP", "PubMedCLIP", "rad-dino", "CheXagent"]
    #model_names = ["BioMedCLIP", "rad-dino"]
    #model_names = ["UNI"]
    print("Model names: ", model_names)
    print("Fusion method: ", fusion_method)
    print("Projection dim: ", projection_dim)
    print("Number of epochs: ", num_epochs)
    print("img_size: ", img_size)
    fusion_method = fusion_method
    projection_dim = projection_dim
    biofuse_model = BioFuseModel(model_names, fusion_method=fusion_method, projection_dim=projection_dim)
    # Switch to half-precision
    #biofuse_model = biofuse_model.half()
    # Move to GPU
    biofuse_model = biofuse_model.to("cuda")
    
    # Show me the trainable layers
    # print_trainable_parameters(biofuse_model)

    print("Extracting features from the training set...")
    # Extract features from the training set
    embeddings_np, labels_np = generate_embeddings(train_dataloader, biofuse_model, cache_raw_embeddings=True, progress_bar=True)
    
    # Extract features from the validation set
    print("Extracting features from the validation set...")
    val_embeddings_np, val_labels_np = generate_embeddings(val_dataloader, biofuse_model, cache_raw_embeddings=True, is_training=False, progress_bar=True)

    print_cuda_mem_stats()
    #ipdb.set_trace()

    # Set up the classifier
    input_dim = embeddings_np.shape[1]
    output_dim = 1 # binary classification   
    if num_classes > 2:
        output_dim = num_classes    

    print("Output dim: ", output_dim)

    classifier = LogisticRegression2(input_dim, output_dim)
    # Switch to half-precision
    #classifier = classifier.half()
    classifier = classifier.to("cuda")
    #classifier = MLPClassifier(input_dim, hidden_dim=64, output_dim=output_dim)

    optimizer = optim.Adam(list(biofuse_model.parameters()) + list(classifier.parameters()), lr=0.004)
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    

    best_model = None
    best_val_acc = 0.0
    best_loss = float('inf')
    patience = PATIENCE
    patience_counter = 0

    print("Training model...")
    for epoch in tqdm(range(num_epochs)):
        #print(f"Epoch [{epoch+1}/{num_epochs}]..")
        biofuse_model.train()
        classifier.train()
        optimizer.zero_grad()

        # Compute embeddings and labels
        embeddings_tensor, labels_tensor = generate_embeddings(train_dataloader, biofuse_model)
        labels_tensor = labels_tensor.to("cuda")
       
        # Train classifier
        logits = classifier(embeddings_tensor)        
        if num_classes == 2:
            loss = criterion(logits, labels_tensor.unsqueeze(1).float())
        else:
            loss = criterion(logits, labels_tensor)
        loss.backward()            
        optimizer.step()  

        # Log the projection layer weights and gradients
        # log_projection_layer_weights(biofuse_model, epoch, "Train")        
        # log_projection_layer_gradients(biofuse_model, epoch, "Train")
        
        # Evaluate on validation set
        biofuse_model.eval()
        classifier.eval()

        # Features for the validation set
        val_embeddings_tensor, val_labels_tensor = generate_embeddings(val_dataloader, biofuse_model, is_training=False)
        val_labels_tensor = val_labels_tensor.to("cuda")

        with torch.no_grad():
            val_logits = classifier(val_embeddings_tensor)
            
            if num_classes == 2:
                val_loss = criterion(val_logits, val_labels_tensor.unsqueeze(1).float())
                val_predictions = (torch.sigmoid(val_logits) > 0.5).float()  # Apply sigmoid for probability, then threshold
            else:
                val_loss = criterion(val_logits, val_labels_tensor)
                val_predictions = torch.argmax(val_logits, dim=1)

            # Calculate Validation Accuracy
            #val_predictions = (torch.sigmoid(val_logits) > 0.5).float()  # Apply sigmoid for probability, then threshold
            # use argmax
            #val_predictions = torch.argmax(val_logits, dim=1)
            val_accuracy = (val_predictions.squeeze() == val_labels_tensor).float().mean()        
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_loss = val_loss
                #best_model = copy.deepcopy(biofuse_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

        # Reclaim memory from the GPU for embeddings and labels
        del embeddings_tensor, labels_tensor
        del val_embeddings_tensor, val_labels_tensor
        torch.cuda.empty_cache()    
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}') 
        #print("-"*80)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break        

    print("Training completed.")
    # clear cache
    #biofuse_model.clear_cached_embeddings()

    # Print the best validation accuracy and loss 
    print(f"Best Validation Accuracy: {best_val_acc:.4f}, Best Validation Loss: {best_loss.item():.4f}")       

    # Evaluate on validation and test sets
    val_accuracy, val_auc, test_accuracy, test_auc = standalone_eval(dataset, img_size, biofuse_model, model_names, fusion_method, projection_dim)

    append_results_to_csv(dataset, img_size, model_names, fusion_method, projection_dim, epoch + 1, val_accuracy, val_auc, test_accuracy, test_auc)


def append_results_to_csv(dataset, img_size, model_names, fusion_method, projection_dim, epochs, val_accuracy, val_auc, test_accuracy, test_auc):
    file_path = f"results_{dataset}_{img_size}.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Dataset', 'Image Size', 'Models', 'Fusion Method', 'Projection Dim', 'Epochs', 'Val Accuracy', 'Val AUC-ROC', 'Test Accuracy', 'Test AUC-ROC'])
        writer.writerow([dataset, img_size, ','.join(model_names), fusion_method, projection_dim, epochs, f'{val_accuracy:.3f}', f'{val_auc:.3f}', f'{test_accuracy:.3f}', f'{test_auc:.3f}'])


def main():
    parser = argparse.ArgumentParser(description='BioFuse v0.1')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--projection_dim', type=int, default=512, help='Projection dimension')
    parser.add_argument('--fusion_method', type=str, default='concat', help='Fusion method')
    parser.add_argument('--fast_run', type=bool, default=False, help='Fast run')
    parser.add_argument('--dataset', type=str, default='breastmnist', help='Dataset')
    parser.add_argument('--models', type=str, default='BioMedCLIP', help='List of pre-trained models, delimited by comma')
    args = parser.parse_args()

    train_model(args.dataset, 
                args.models.split(','), 
                args.num_epochs, 
                args.img_size, 
                args.projection_dim, 
                args.fusion_method, 
                args.fast_run)
    
if __name__ == "__main__":
    main()
