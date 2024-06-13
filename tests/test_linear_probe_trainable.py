import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from medmnist import BreastMNIST
# progressbar
from tqdm import tqdm
import sys, os, glob
from biofuse.models.image_dataset import BioFuseImageDataset

# Trainable layer imports
import torch.optim as optim
import torch.nn as nn

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
    train_image_paths = glob.glob(f'{train_images_path}/*.png')
    val_image_paths = glob.glob(f'{val_images_path}/*.png')
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
        features.append(embedding.squeeze(0).numpy())
        labels.append(label.numpy())

    # stack
    # features = torch.cat(features, dim=0)
    # labels = torch.cat(labels, dim=0)
    
    # convert to numpy and return
    # return features.numpy(), labels.numpy()
    return features, labels

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

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
def train_knn(features, labels):
    print("Training classifier...")
    # Scale
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train a simple linear classifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(features, labels)
    return classifier

def evaluate_model(classifier, features, labels):
    print("Evaluating model...")
    predictions = classifier.predict(features)
    return accuracy_score(labels, predictions)
        
def main():
    # Load data
    train_loader, val_loader = load_data()

    # Initialize BioFuse model
    model_names = ["BioMedCLIP", "rad-dino"] #"PubMedCLIP"] #, "rad-dino"]
    fusion_method = "concat"
    biofuse_model = BioFuseModel(model_names, fusion_method)

    # Extract features
    train_features, train_labels = extract_features(train_loader, biofuse_model)

    # Train a classifier
    classifier, scaler = train_classifier(train_features, train_labels)
    
    # get validation features
    val_features, val_labels = extract_features(val_loader, biofuse_model)    

    # Evaluate the model
    accuracy = evaluate_model(classifier, scaler.transform(val_features), val_labels)
    print("Accuracy: ", accuracy)

# Training the model with validation-informed adjustment
def train_model():
    train_dataloader, val_dataloader = load_data()

    model_names = ["BioMedCLIP"]
    fusion_method = "mean"
    biofuse_model = BioFuseModel(model_names, fusion_method=fusion_method, projection_dim=512)

    optimizer = optim.Adam(biofuse_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        biofuse_model.train()
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            embeddings = [biofuse_model(image) for image in images]
            embeddings_tensor = torch.stack(embeddings).squeeze(1)
            labels_tensor = torch.tensor(labels).squeeze(0)

            # Convert embeddings and labels to numpy
            embeddings_np = embeddings_tensor.detach().cpu().numpy()
            labels_np = labels_tensor.detach().cpu().numpy()

            # Train classifier on embeddings
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(embeddings_np, labels_np)
            
            # Evaluate on validation set
            biofuse_model.eval()
            val_features, val_labels = extract_features(val_dataloader, biofuse_model)
            val_features_np = scaler.transform(val_features)
            val_predictions = classifier.predict(val_features_np)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')

            # Compute validation loss
            val_probs = classifier.predict_proba(val_features_np)
            val_loss = criterion(torch.tensor(val_probs, requires_grad=True), torch.tensor(val_labels))
            
            # Backpropagate validation loss
            optimizer.zero_grad()
            val_loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {val_loss.item():.4f}')

    print("Training completed.")

if __name__ == "__main__":
    train_model()