import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from medmnist import BreastMNIST
# progressbar
from tqdm import tqdm
import sys, os, glob
from biofuse.models.image_dataset import BioFuseImageDataset
import numpy as np

# Trainable layer imports
import torch.optim as optim
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        #outputs = torch.sigmoid(self.linear(x))
        return self.linear(x)

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def load_data():
    print("Loading data...")
    train_dataset = BreastMNIST(split='train', size=224, download=True)
    val_dataset = BreastMNIST(split='val', size=224, download=True)

    # # Use only a subset of the data for faster training
    # train_dataset = Subset(train_dataset, range(25))
    # val_dataset = Subset(val_dataset, range(10))

    # Save the images to disk if not already done
    if not os.path.exists('/tmp/breastmnist_train'):
        train_dataset.save('/tmp/breastmnist_train')
        val_dataset.save('/tmp/breastmnist_val')

    train_images_path = '/tmp/breastmnist_train/breastmnist_224'
    val_images_path = '/tmp/breastmnist_val/breastmnist_224'

    # Construct image paths, glob directory
    train_image_paths = glob.glob(f'{train_images_path}/*.png')[:100]
    val_image_paths = glob.glob(f'{val_images_path}/*.png')[:20]
    # labels are just _0.png or _1.png etc
    train_labels = [int(path.split('_')[-1].split('.')[0]) for path in train_image_paths]
    val_labels = [int(path.split('_')[-1].split('.')[0]) for path in val_image_paths]

    # Construct the dataste now
    train_dataset = BioFuseImageDataset(train_image_paths, train_labels)
    val_dataset = BioFuseImageDataset(val_image_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader

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
    

def generate_embeddings(dataloader, biofuse_model, cache_raw_embeddings=False):
    embeddings = []
    labels = []
    
    for image, label in tqdm(dataloader):
        embedding = biofuse_model(image, cache_raw_embeddings=cache_raw_embeddings)
        embeddings.append(embedding)
        labels.append(label)
    
    # Embeddings is a list of tensors, stack them and remove the batch dimension
    embeddings_tensor = torch.stack(embeddings).squeeze(1)
    labels_tensor = torch.tensor(labels)        
    
    return embeddings_tensor, labels_tensor

def print_trainable_parameters(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        # numel
        
        if param.requires_grad:
            print(name, param.shape)
            print(name, param.numel())

def log_projection_layer_weights(model, epoch, stage):
    for i, layer in enumerate(model.projection_layers):
        weights = layer.weight.data
        print(f"Epoch [{epoch}] - {stage} - Projection Layer {i} Weights: {weights.mean().item():.6f} ± {weights.std().item():.6f}")

def log_projection_layer_gradients(model, epoch, stage):
    for i, layer in enumerate(model.projection_layers):
        if layer.weight.grad is not None:
            grad = layer.weight.grad.data
            print(f"Epoch [{epoch}] - {stage} - Projection Layer {i} Gradients: {grad.mean().item():.6f} ± {grad.std().item():.6f}")
        else:
            print(f"Epoch [{epoch}] - {stage} - Projection Layer {i} Gradients: None")

# Training the model with validation-informed adjustment
def train_model():
    train_dataloader, val_dataloader = load_data()

    model_names = ["BioMedCLIP"] #, "rad-dino"]
    fusion_method = "mean"
    biofuse_model = BioFuseModel(model_names, fusion_method=fusion_method, projection_dim=512)
    
    # Show me the trainable layers
    print_trainable_parameters(biofuse_model)

    print("Extracting features...")
    # Extract features from the training set
    embeddings_np, labels_np = generate_embeddings(train_dataloader, biofuse_model, cache_raw_embeddings=True)
    
    # Extract features from the validation set
    val_embeddings_np, val_labels_np = generate_embeddings(val_dataloader, biofuse_model, cache_raw_embeddings=True)

    # Set up the classifier
    input_dim = embeddings_np.shape[1]
    output_dim = 1 # binary classification

    print("Setting up classifier...")
    print("Input dimension:", input_dim)
    print("Output dimension:", output_dim)

    classifier = LogisticRegression(input_dim, output_dim)

    optimizer = optim.Adam(list(biofuse_model.parameters()) + list(classifier.parameters()), lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 25
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]..")
        biofuse_model.train()
        classifier.train()
        optimizer.zero_grad()

        # Compute embeddings and labels
        embeddings_tensor, labels_tensor = generate_embeddings(train_dataloader, biofuse_model)

        # Train classifier on embeddings
        # embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32, requires_grad=True)
        # labels_tensor = torch.tensor(labels_np, dtype=torch.long)

        # print("Logits shape:", classifier(embeddings_tensor).shape)
        # print("Labels shape:", labels_tensor.shape)
        # print("Labels:", labels_tensor)

        # Train classifier
        logits = classifier(embeddings_tensor)
        loss = criterion(logits, labels_tensor.unsqueeze(1).float())
        loss.backward()            
        optimizer.step()  

        # Log the projection layer weights and gradients
        log_projection_layer_weights(biofuse_model, epoch, "Train")        
        log_projection_layer_gradients(biofuse_model, epoch, "Train")
        
        # Evaluate on validation set
        biofuse_model.eval()
        classifier.eval()

        # Features for the validation set
        val_embeddings_tensor, val_labels_tensor = generate_embeddings(val_dataloader, biofuse_model)

        with torch.no_grad():
            # val_embeddings_tensor = torch.tensor(val_embeddings_np, dtype=torch.float32, requires_grad=True)
            # val_labels_tensor = torch.tensor(val_labels_np, dtype=torch.long)
            val_logits = classifier(val_embeddings_tensor)

            val_loss = criterion(val_logits, val_labels_tensor.unsqueeze(1).float())
            # Calculate Validation Accuracy
            val_predictions = (torch.sigmoid(val_logits) > 0.5).float()  # Apply sigmoid for probability, then threshold
            val_accuracy = (val_predictions.squeeze() == val_labels_tensor).float().mean()
            print(f"Validation Accuracy: {val_accuracy.item():.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')        

        #PATIENCE=5
        # stop when validation loss increases or stops decreasing after PATIENCE epochs
        #if val_loss > best_val_loss and 

               
    print("Training completed.")

if __name__ == "__main__":
    train_model()