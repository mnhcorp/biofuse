import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from medmnist import BreastMNIST
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

# Trainable layer imports
import torch.optim as optim
import torch.nn as nn

PATIENCE = 8

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

def load_data(dataset, img_size, fast_run):
    print(f"Loading data for {dataset}...")
    
    info = INFO[dataset]
    num_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', download=True, size=img_size, root='/data/medmnist')
    val_dataset = DataClass(split='val', download=True, size=img_size, root='/data/medmnist')
    test_dataset = DataClass(split='test', download=True, size=img_size, root='/data/medmnist')

    # force save
    # train_dataset.save(f'/tmp/{dataset}_train')
    # val_dataset.save(f'/tmp/{dataset}_val')
    # test_dataset.save(f'/tmp/{dataset}_test')
    
    # Save the images to disk if not already done
    if not os.path.exists(f'/tmp/{dataset}_train'):
        train_dataset.save(f'/tmp/{dataset}_train')
    
    if not os.path.exists(f'/tmp/{dataset}_val'):
        val_dataset.save(f'/tmp/{dataset}_val')

    if not os.path.exists(f'/tmp/{dataset}_test'):
        test_dataset.save(f'/tmp/{dataset}_test')
    
    if img_size == 28:
        train_images_path = f'/tmp/{dataset}_train/{dataset}'
        val_images_path = f'/tmp/{dataset}_val/{dataset}'
        test_images_path = f'/tmp/{dataset}_test/{dataset}'
    else:
        train_images_path = f'/tmp/{dataset}_train/{dataset}_{img_size}'
        val_images_path = f'/tmp/{dataset}_val/{dataset}_{img_size}'
        test_images_path = f'/tmp/{dataset}_test/{dataset}_{img_size}'
    
    # Construct image paths, glob directory
    train_image_paths = glob.glob(f'{train_images_path}/*.png')
    val_image_paths = glob.glob(f'{val_images_path}/*.png')
    test_image_paths = glob.glob(f'{test_images_path}/*.png')

    print(f"Number of training images: {len(train_image_paths)}")
    print(f"Number of validation images: {len(val_image_paths)}")
    print(f"Number of test images: {len(test_image_paths)}")
    
    # if FAST_RUN:
    #     train_image_paths = train_image_paths[:2500]
    #     val_image_paths = val_image_paths[:10000]
    #     test_image_paths = test_image_paths[:2500]
    
    # Labels are just _0.png or _1.png etc
    train_labels = [int(path.split('_')[-1].split('.')[0]) for path in train_image_paths]
    val_labels = [int(path.split('_')[-1].split('.')[0]) for path in val_image_paths]
    test_labels = [int(path.split('_')[-1].split('.')[0]) for path in test_image_paths]
    
    # Construct the datasets
    train_dataset = BioFuseImageDataset(train_image_paths, train_labels)
    val_dataset = BioFuseImageDataset(val_image_paths, val_labels)
    test_dataset = BioFuseImageDataset(test_image_paths, test_labels)

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
        # generate a random tensor for now
        #embedding = torch.randn(512)
        embeddings.append(embedding)
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

def train_classifier(features, labels, scaler=None):
    print("Training classifier...")

    if scaler is None:        
        scaler = StandardScaler()
    
    # Scale features
    features = scaler.fit_transform(features)

    # Train a simple linear classifier
    classifier = LogisticRegression(max_iter=1000, solver='liblinear')
    classifier.fit(features, labels)
    return classifier, scaler

def evaluate_model(classifier, features, labels):
    print("Evaluating model...")
    predictions = classifier.predict(features)
    return accuracy_score(labels, predictions)

def standalone_eval(train_dataloader, val_dataloader, test_dataloader, model_path, models, fusion_method, projection_dim):    
    biofuse = BioFuseModel(models, fusion_method, projection_dim=projection_dim)

    # Load the state dictionary
    state_dict = torch.load(model_path)
    biofuse.load_state_dict(state_dict)
    biofuse = biofuse.to("cuda")

    # Extract features from the training set
    embeddings_tensor, labels_tensor = generate_embeddings(train_dataloader, biofuse, progress_bar=True, is_test=True)

    # convert to numpy
    embeddings_np = embeddings_tensor.cpu().detach().numpy()
    labels_np = labels_tensor.cpu().detach().numpy()

    # Train a simple linear classifier
    classifier, scaler = train_classifier(embeddings_np, labels_np)

    # Extract features from the validation set
    val_embeddings_tensor, val_labels_tensor = generate_embeddings(val_dataloader, biofuse, progress_bar=True, is_test=True)

    # convert to numpy
    val_embeddings_np = val_embeddings_tensor.cpu().detach().numpy()
    val_labels_np = val_labels_tensor.cpu().detach().numpy()

    # Scale features
    val_embeddings_np = scaler.transform(val_embeddings_np)

    # Evaluate the model
    val_accuracy = evaluate_model(classifier, val_embeddings_np, val_labels_np)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return val_accuracy

    # # on the test set
    # test_embeddings_tensor, test_labels_tensor = generate_embeddings(test_dataloader, biofuse, progress_bar=True, is_test=True)

    # # convert to numpy
    # test_embeddings_np = test_embeddings_tensor.cpu().detach().numpy()
    # test_labels_np = test_labels_tensor.cpu().detach().numpy()

    # # Scale features
    # test_embeddings_np = scaler.transform(test_embeddings_np)

    # test_accuracy = evaluate_model(classifier, test_embeddings_np, test_labels_np)

    # print(f"Test Accuracy: {test_accuracy:.4f}")
    # return val_accuracy, test_accuracy

        
# Training the model with validation-informed adjustment
def train_model(dataset, model_names, num_epochs, img_size, projection_dim, fusion_method, fast_run):
    set_seed(42)

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_data(dataset, img_size, fast_run)
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

    optimizer = optim.Adam(list(biofuse_model.parameters()) + list(classifier.parameters()), lr=0.005)
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
                best_model = copy.deepcopy(biofuse_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}') 
        #print("-"*80)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break        

    print("Training completed.")
    # clear cache
    biofuse_model.clear_cached_embeddings()

    # Print the best validation accuracy and loss 
    print(f"Best Validation Accuracy: {best_val_acc:.4f}, Best Validation Loss: {best_loss.item():.4f}")       

    # save the best model
    print("Saving the best model...")
    model_path = f"./models/biofuse_{fusion_method}.pt"    
    torch.save(best_model, model_path)

    # print(f"Test Accuracy: {test_accuracy:.4f}")
    val_accuracy = standalone_eval(train_dataloader, val_dataloader, test_dataloader, model_path, model_names, fusion_method, projection_dim)

    append_results_to_csv(dataset, img_size, model_names, fusion_method, projection_dim, epoch + 1, val_accuracy)


def append_results_to_csv(dataset, img_size, model_names, fusion_method, projection_dim, epochs, val_accuracy):
    file_path = f"results_{dataset}.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dataset, img_size, ','.join(model_names), fusion_method, projection_dim, epochs, f'{val_accuracy:.3f}'])


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