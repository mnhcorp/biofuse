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
import random
import numpy as np

IMG_SIZE = 28

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

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def load_data():
    print("Loading data...")
    train_dataset = BreastMNIST(split='train', download=True, size=IMG_SIZE, root='/data/medmnist')
    val_dataset = BreastMNIST(split='val', download=True, size=IMG_SIZE, root='/data/medmnist')
    test_dataset = BreastMNIST(split='test', download=True, size=IMG_SIZE, root='/data/medmnist')

    # # Use only a subset of the data for faster training
    # train_dataset = Subset(train_dataset, range(25))
    # val_dataset = Subset(val_dataset, range(10))

    # Save the images to disk if not already done
    if not os.path.exists('/tmp/breastmnist_train'):
        train_dataset.save('/tmp/breastmnist_train')
        val_dataset.save('/tmp/breastmnist_val')
        test_dataset.save('/tmp/breastmnist_test')

    if IMG_SIZE != 28:
        train_images_path = '/tmp/breastmnist_train/breastmnist_{IMG_SIZE}'
        val_images_path = '/tmp/breastmnist_val/breastmnist_{IMG_SIZE}'
        test_images_path = '/tmp/breastmnist_test/breastmnist_{IMG_SIZE}'
    else:
        train_images_path = '/tmp/breastmnist_train/breastmnist'
        val_images_path = '/tmp/breastmnist_val/breastmnist'
        test_images_path = '/tmp/breastmnist_test/breastmnist'

    # print lens
    print("Train dataset size: ", len(train_dataset))
    print("Val dataset size: ", len(val_dataset))
    print("Test dataset size: ", len(test_dataset))

    # Construct image paths, glob directory
    train_image_paths = glob.glob(f'{train_images_path}/*.png')
    val_image_paths = glob.glob(f'{val_images_path}/*.png')
    test_image_paths = glob.glob(f'{test_images_path}/*.png')

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
        features.append(embedding.squeeze(0).detach().cpu().numpy())
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
    # set seed
    set_seed(42)

    # Load data
    train_loader, val_loader, test_loader = load_data()

    # Initialize BioFuse model
    model_names = ["BioMedCLIP"] #, "rad-dino", "PubMedCLIP"] #, "rad-dino"]
    fusion_method = "concat"
    biofuse_model = BioFuseModel(model_names, fusion_method)
    biofuse_model.to("cuda")

    # Extract features
    train_features, train_labels = extract_features(train_loader, biofuse_model)

    # Train a classifier
    classifier, scaler = train_classifier(train_features, train_labels)
    
    # get validation features
    val_features, val_labels = extract_features(val_loader, biofuse_model)    

    # Evaluate the model
    accuracy = evaluate_model(classifier, scaler.transform(val_features), val_labels)
    print("Val Accuracy: ", accuracy)

    # get test features
    test_features, test_labels = extract_features(test_loader, biofuse_model)

    # Evaluate the model
    accuracy = evaluate_model(classifier, scaler.transform(test_features), test_labels)
    print("Test Accuracy: ", accuracy)

if __name__ == "__main__":
    main()