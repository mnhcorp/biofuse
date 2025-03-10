from biofuse import BioFuse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Example 1: Using MedMNIST dataset
print("Example 1: MedMNIST dataset")
biofuse = BioFuse(models=['BioMedCLIP', 'rad-dino', 'UNI'], fusion_method='concat')

# Generate embeddings for MedMNIST dataset
train_embeddings, train_labels, val_embeddings, val_labels, biofuse_model = biofuse.generate_embeddings(
    train_data=None,  # Not needed for MedMNIST
    val_data=None,    # Not needed for MedMNIST
    dataset_type=BioFuse.MEDMNIST,
    dataset_name='pathmnist',  # Specify which MedMNIST dataset to use
    img_size=224,
    task_type=BioFuse.MULTICLASS
)

# Train a downstream model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_embeddings, train_labels)

# Evaluate on validation data
val_preds = classifier.predict(val_embeddings)
accuracy = accuracy_score(val_labels, val_preds)
print(f"MedMNIST Validation Accuracy: {accuracy:.4f}")

# Example 2: Using ImageNet dataset
print("\nExample 2: ImageNet dataset")
biofuse_imagenet = BioFuse(models=['BioMedCLIP', 'rad-dino'], fusion_method='mean')

# Generate embeddings for ImageNet dataset
train_embeddings, train_labels, val_embeddings, val_labels, biofuse_model = biofuse_imagenet.generate_embeddings(
    train_data=None,  # Not needed for ImageNet
    val_data=None,    # Not needed for ImageNet
    dataset_type=BioFuse.IMAGENET,
    root='/path/to/imagenet',  # Specify ImageNet root directory
    task_type=BioFuse.MULTICLASS
)

# Train a downstream model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_embeddings, train_labels)

# Evaluate on validation data
val_preds = classifier.predict(val_embeddings)
accuracy = accuracy_score(val_labels, val_preds)
print(f"ImageNet Validation Accuracy: {accuracy:.4f}")

# Example 3: Using custom dataset from directory
print("\nExample 3: Custom dataset")
biofuse_custom = BioFuse(models=['BioMedCLIP', 'UNI'], fusion_method='concat')

# Generate embeddings for custom dataset
train_embeddings, train_labels, val_embeddings, val_labels, biofuse_model = biofuse_custom.generate_embeddings(
    train_data='path/to/train',
    val_data='path/to/val',
    dataset_type=BioFuse.CUSTOM,  # Default value, can be omitted
    task_type=BioFuse.BINARY
)

# Train a downstream model
classifier = LogisticRegression()
classifier.fit(train_embeddings, train_labels)

# Evaluate on validation data
val_preds = classifier.predict(val_embeddings)
val_probs = classifier.predict_proba(val_embeddings)[:, 1]  # For binary classification
accuracy = accuracy_score(val_labels, val_preds)
auc = roc_auc_score(val_labels, val_probs)
print(f"Custom Dataset Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# Generate test embeddings and evaluate
test_embeddings, test_labels = biofuse_custom.embed(
    data='path/to/test',
    dataset_type=BioFuse.CUSTOM
)
test_preds = classifier.predict(test_embeddings)
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Custom Dataset Test Accuracy: {test_accuracy:.4f}")

# Save the BioFuse model for later use
biofuse_custom.save('biofuse_model.pt')
