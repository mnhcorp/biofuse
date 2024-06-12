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
import sys

def load_data():
    print("Loading data...")
    train_dataset = BreastMNIST(split='train', size=224, download=True)
    val_dataset = BreastMNIST(split='val', size=224, download=True)

    # Use only a subset of the data for faster training
    # train_dataset = Subset(train_dataset, range(25))
    # val_dataset = Subset(val_dataset, range(10))
    
    return train_dataset, val_dataset

def extract_features(dataset, biofuse_model):
    print("Extracting features...")
    features = []
    labels = []    
    # use progress bar
    for image, label in tqdm(dataset):    
        embedding = biofuse_model(image)
        features.append(embedding.squeeze(0).numpy())
        labels.append(label)

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
    classifier = LogisticRegression(max_iter=1000, n_jobs=-1, solver='liblinear')
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
    train_dataset, val_dataset = load_data()    

    # Initialize BioFuse model
    model_names = ["BioMedCLIP"] #, "rad-dino"]
    fusion_method = "concat"
    biofuse_model = BioFuseModel(model_names, fusion_method)

    # Extract features
    train_features, train_labels = extract_features(train_dataset, biofuse_model)

    # print("Train features shape: ", train_features.shape)
    # print("Train labels shape: ", train_labels.shape)
    # print(type(train_features))
    # print(len(train_features))
    # print(type(train_features[0]))
    # print(train_features[0].shape)
    # print(type(train_labels))
    # print(len(train_labels))
    # #print(train_labels[0].shape)
    # print(train_labels[0])

    # Train a classifier
    classifier, scaler = train_classifier(train_features, train_labels)
    
    # get validation features
    val_features, val_labels = extract_features(val_dataset, biofuse_model)    

    # Evaluate the model
    accuracy = evaluate_model(classifier, scaler.transform(val_features), val_labels)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()