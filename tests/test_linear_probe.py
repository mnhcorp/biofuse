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
        features.append(embedding.squeeze(0).cpu().numpy())
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
    model_names = ["BioMedCLIP", "rad-dino", "PubMedCLIP"] #, "rad-dino"]
    fusion_method = "concat"
    biofuse_model = BioFuseModel(model_names, fusion_method)

    # Extract features
    train_features, train_labels = extract_features(train_loader, biofuse_model)
  
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
    val_features, val_labels = extract_features(val_loader, biofuse_model)    

    # Evaluate the model
    accuracy = evaluate_model(classifier, scaler.transform(val_features), val_labels)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()