# Migration Guide: BioFuse v0.1 → v2.0

This guide helps you migrate from BioFuse v0.1 to v2.0.

## Overview of Changes

v2.0 is a complete refactoring with:
- Modern CLI replacing monolithic scripts
- Configuration system (YAML/JSON)
- Modular architecture
- Proper Python packaging
- Unified classifier interface

## Quick Migration

### Command Line Usage

**v0.1:**
```bash
python tests/test_linear_probe_trainable2.py \
  --dataset pathmnist \
  --models BC,CO,RD \
  --img_size 224 \
  --fusion_methods concat \
  --projections 0 \
  --batch_size 32 \
  --test_classifier xgb
```

**v2.0:**
```bash
biofuse train \
  --dataset pathmnist \
  --models BioMedCLIP,CONCH,rad-dino \
  --fusion-method concat \
  --projection-dim 0 \
  --batch-size 32 \
  --classifier xgboost
```

### Model Name Changes

v0.1 used abbreviations, v2.0 uses full names:

| v0.1 | v2.0 |
|------|------|
| BC | BioMedCLIP |
| PC | PubMedCLIP |
| CO | CONCH |
| RD | rad-dino |
| UN | UNI |
| UN2 | UNI2 |
| PG | Prov-GigaPath |
| HB | Hibou-B |
| CA | CheXagent |

### Python API Changes

**v0.1:**
```python
from biofuse.biofuse import BioFuse

biofuse = BioFuse(models=['BioMedCLIP', 'CONCH'])
# ... complex setup required
```

**v2.0:**
```python
from biofuse import BioFuse, load_medmnist, get_classifier

# Cleaner imports and API
train_data, num_classes = load_medmnist('pathmnist', split='train')
biofuse = BioFuse(models=['BioMedCLIP', 'CONCH'])
classifier = get_classifier('xgboost')
```

## Detailed Changes

### 1. Configuration Files (New in v2.0)

v2.0 supports configuration files for reproducibility:

```yaml
# experiment.yaml
name: my_experiment
data:
  dataset: pathmnist
  img_size: 224
model:
  models: [BioMedCLIP, CONCH]
  fusion_method: concat
classifier:
  type: xgboost
```

Run with:
```bash
biofuse train --config experiment.yaml
```

### 2. Cache Management

**v0.1:**
- Hardcoded cache directory: `/data/biofuse-embedding-cache3`
- No cache versioning
- Manual cache clearing

**v2.0:**
- Configurable: `BIOFUSE_CACHE_DIR` environment variable
- Cache versioning for invalidation
- CLI commands:
  ```bash
  biofuse cache list
  biofuse cache clear --dataset pathmnist
  biofuse cache info
  ```

### 3. Classifier Interface

**v0.1:**
```python
# Different functions for different classifiers
train_classifier2(features, labels, num_classes)  # XGBoost
train_catboost_classifier(features, labels, num_classes)  # CatBoost
train_nn_classifier(features, labels, num_classes)  # Neural nets
```

**v2.0:**
```python
# Unified interface
from biofuse.classifiers import get_classifier

classifier = get_classifier('xgboost')  # or 'catboost', 'nn_mlp', etc.
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

### 4. Data Loading

**v0.1:**
```python
from biofuse.models.data_adapter import DataAdapter

dataset, num_classes = DataAdapter.from_medmnist('pathmnist', 'train', 224)
```

**v2.0:**
```python
from biofuse.data import load_medmnist

dataset, num_classes = load_medmnist('pathmnist', split='train', img_size=224)
```

### 5. Evaluation

**v0.1:**
```python
# Manual evaluation
accuracy = evaluate_model(classifier, features, labels, dataset)
auc = compute_auc_roc(classifier, features, labels, num_classes, dataset)
```

**v2.0:**
```python
from biofuse.evaluation import Evaluator

evaluator = Evaluator(verbose=True)
results = evaluator.evaluate_classifier(
    classifier, X_val, y_val, X_test, y_test, dataset='pathmnist'
)
# Returns: {'validation': {'accuracy': ..., 'auc_roc': ...}, 
#           'test': {'accuracy': ..., 'auc_roc': ...}}
```

## Breaking Changes

### Removed Features
- ❌ Shell script automation (`autofuse*.sh`) → Use `biofuse` CLI
- ❌ Model abbreviations (BC, CO, etc.) → Use full names
- ❌ Direct script execution → Use CLI or Python API

### Deprecated (Still Available)
- ⚠️ `tests/test_linear_probe_trainable2.py` → Migrate to CLI
- ⚠️ Old data adapter → Use new `biofuse.data` module

## Migration Checklist

- [ ] Install v2.0: `pip install -e .`
- [ ] Update scripts to use `biofuse` CLI
- [ ] Convert model abbreviations to full names
- [ ] (Optional) Create config files for experiments
- [ ] Update Python imports if using API
- [ ] Test your workflow with new commands
- [ ] Update environment variables for paths

## Getting Help

If you encounter issues during migration:
1. Check the [README.md](README.md) for examples
2. See example configs in `examples/configs/`
3. Run `biofuse --help` or `biofuse train --help`
4. Open an issue on GitHub

## Example Migration

**Before (v0.1):**
```bash
#!/bin/bash
python tests/test_linear_probe_trainable2.py \
  --dataset chestmnist \
  --models BC,RD,CA \
  --img_size 224 \
  --fusion_methods concat \
  --projections 512 \
  --test_classifier xgb \
  --batch_size 64 \
  --num_epochs 100
```

**After (v2.0 - Option 1: CLI)**
```bash
#!/bin/bash
biofuse train \
  --dataset chestmnist \
  --models BioMedCLIP,rad-dino,CheXagent \
  --fusion-method concat \
  --projection-dim 512 \
  --classifier xgboost \
  --batch-size 64
```

**After (v2.0 - Option 2: Config)**
```yaml
# chestmnist_config.yaml
name: chestmnist_experiment
data:
  dataset: chestmnist
  img_size: 224
  batch_size: 64
model:
  models: [BioMedCLIP, rad-dino, CheXagent]
  fusion_method: concat
  projection_dim: 512
classifier:
  type: xgboost
  n_estimators: 250
```

```bash
biofuse train --config chestmnist_config.yaml
```
