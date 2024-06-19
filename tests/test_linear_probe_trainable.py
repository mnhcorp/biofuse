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
import sys, os, glob
from biofuse.models.image_dataset import BioFuseImageDataset
import numpy as np
import copy

# Trainable layer imports
import torch.optim as optim
import torch.nn as nn

FAST_RUN = False
NUM_EPOCHS = 100

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
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def load_data():
    print("Loading data...")
    train_dataset = BreastMNIST(split='train', size=224, download=True)
    val_dataset = BreastMNIST(split='val', size=224, download=True)
    test_dataset = BreastMNIST(split='test', size=224, download=True)

    # # Use only a subset of the data for faster training
    # train_dataset = Subset(train_dataset, range(25))
    # val_dataset = Subset(val_dataset, range(10))

    # Save the images to disk if not already done
    if not os.path.exists('/tmp/breastmnist_train'):
        train_dataset.save('/tmp/breastmnist_train')
        val_dataset.save('/tmp/breastmnist_val')
        test_dataset.save('/tmp/breastmnist_test')

    train_images_path = '/tmp/breastmnist_train/breastmnist_224'
    val_images_path = '/tmp/breastmnist_val/breastmnist_224'
    test_images_path = '/tmp/breastmnist_test/breastmnist_224'

    # Construct image paths, glob directory
    train_image_paths = glob.glob(f'{train_images_path}/*.png')
    val_image_paths = glob.glob(f'{val_images_path}/*.png')
    test_image_paths = glob.glob(f'{test_images_path}/*.png')

    if FAST_RUN:
        train_image_paths = train_image_paths[:100]
        val_image_paths = val_image_paths[:25]
        test_image_paths = test_image_paths[:25]
    
    # labels are just _0.png or _1.png etc
    train_labels = [int(path.split('_')[-1].split('.')[0]) for path in train_image_paths]
    val_labels = [int(path.split('_')[-1].split('.')[0]) for path in val_image_paths]
    test_labels = [int(path.split('_')[-1].split('.')[0]) for path in test_image_paths]

    # Construct the dataste now
    train_dataset = BioFuseImageDataset(train_image_paths, train_labels)
    val_dataset = BioFuseImageDataset(val_image_paths, val_labels)
    test_dataset = BioFuseImageDataset(test_image_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader, test_loader

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

def standalone_eval(train_dataloader, val_dataloader, model_path, models, fusion_method, projection_dim):    
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

        
# Training the model with validation-informed adjustment
def train_model():
    train_dataloader, val_dataloader, test_dataloader = load_data()

    #model_names =  ["rad-dino"] 
    #model_names =  ["BioMedCLIP"]
    #model_names = ["PubMedCLIP"]
    model_names = ["BioMedCLIP", "PubMedCLIP", "rad-dino"]
    #model_names = ["BioMedCLIP", "rad-dino"]
    fusion_method = "mean"
    projection_dim = 512
    biofuse_model = BioFuseModel(model_names, fusion_method=fusion_method, projection_dim=projection_dim)
    # Move to GPU
    biofuse_model = biofuse_model.to("cuda")
    
    # Show me the trainable layers
    # print_trainable_parameters(biofuse_model)

    print("Extracting features...")
    # Extract features from the training set
    embeddings_np, labels_np = generate_embeddings(train_dataloader, biofuse_model, cache_raw_embeddings=True, progress_bar=True)
    
    # Extract features from the validation set
    val_embeddings_np, val_labels_np = generate_embeddings(val_dataloader, biofuse_model, cache_raw_embeddings=True, is_training=False, progress_bar=True)

    # Set up the classifier
    input_dim = embeddings_np.shape[1]
    output_dim = 1 # binary classification   

    classifier = LogisticRegression2(input_dim, output_dim)
    classifier = classifier.to("cuda")
    #classifier = MLPClassifier(input_dim, hidden_dim=64, output_dim=output_dim)

    optimizer = optim.Adam(list(biofuse_model.parameters()) + list(classifier.parameters()), lr=0.005)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    best_model = None
    best_val_acc = 0.0
    best_loss = float('inf')

    print("Training model...")
    for epoch in tqdm(range(NUM_EPOCHS)):
        #print(f"Epoch [{epoch+1}/{num_epochs}]..")
        biofuse_model.train()
        classifier.train()
        optimizer.zero_grad()

        # Compute embeddings and labels
        embeddings_tensor, labels_tensor = generate_embeddings(train_dataloader, biofuse_model)
        labels_tensor = labels_tensor.to("cuda")
       
        # Train classifier
        logits = classifier(embeddings_tensor)
        loss = criterion(logits, labels_tensor.unsqueeze(1).float())
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
            val_loss = criterion(val_logits, val_labels_tensor.unsqueeze(1).float())

            # Calculate Validation Accuracy
            val_predictions = (torch.sigmoid(val_logits) > 0.5).float()  # Apply sigmoid for probability, then threshold
            val_accuracy = (val_predictions.squeeze() == val_labels_tensor).float().mean()        

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_loss = val_loss
                best_model = copy.deepcopy(biofuse_model.state_dict())
            
            
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}') 
        #print("-"*80)

        #PATIENCE=5
        # stop when validation loss increases or stops decreasing after PATIENCE epochs
        #if val_loss > best_val_loss and 

    print("Training completed.")
    # clear cache
    biofuse_model.clear_cached_embeddings()

    # Print the best validation accuracy and loss 
    print(f"Best Validation Accuracy: {best_val_acc:.4f}, Best Validation Loss: {best_loss.item():.4f}")       

    # save the best model
    print("Saving the best model...")
    model_path = f"./models/biofuse_{fusion_method}.pt"    
    torch.save(best_model, model_path)

    # print("Evaluating on test set...")    
    # biofuse_model.load_state_dict(best_model)
    # biofuse_model.eval()

    # print("Extracting features...")
    # test_embeddings_tensor, test_labels_tensor = generate_embeddings(test_dataloader, biofuse_model, is_Test=True)
    # test_labels_tensor = test_labels_tensor.to("cuda")
        
    # with torch.no_grad():
    #     test_logits = classifier(test_embeddings_tensor)
    #     test_logits = test_logits.to("cuda")
    #     test_predictions = (torch.sigmoid(test_logits) > 0.5).float()
    #     test_accuracy = (test_predictions.squeeze() == test_labels_tensor).float().mean()

    # print(f"Test Accuracy: {test_accuracy:.4f}")
    standalone_eval(train_dataloader, val_dataloader, model_path, model_names, fusion_method, projection_dim)

if __name__ == "__main__":
    train_model()