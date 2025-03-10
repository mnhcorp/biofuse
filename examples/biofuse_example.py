from biofuse import BioFuse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Step 1: Initialize BioFuse
biofuse = BioFuse(models=['BioMedCLIP', 'rad-dino', 'UNI'], fusion_method='concat')

# Step 2: Generate BioFuse embeddings
train_embeddings, train_labels, val_embeddings, val_labels, biofuse_model = biofuse.generate_embeddings(
    train_data='path/to/train', 
    val_data='path/to/val',
    task_type=BioFuse.BINARY
)

# Step 3: Train a downstream model
classifier = LogisticRegression()
classifier.fit(train_embeddings, train_labels)

# Step 4: Evaluate on validation data
val_preds = classifier.predict(val_embeddings)
val_probs = classifier.predict_proba(val_embeddings)[:, 1]  # For binary classification
accuracy = accuracy_score(val_labels, val_preds)
auc = roc_auc_score(val_labels, val_probs)
print(f"Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# Step 5: Generate test embeddings and evaluate
test_embeddings, test_labels = biofuse.embed('path/to/test')
test_preds = classifier.predict(test_embeddings)
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the BioFuse model for later use
biofuse.save('biofuse_model.pt')
