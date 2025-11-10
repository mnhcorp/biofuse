# BioFuse v2.0 - Multi-Modal Fusion Framework for Biomedical Foundation Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](setup.py)

BioFuse enables combining embeddings from multiple pre-trained foundation models to improve performance on biomedical imaging tasks. Version 2.0 is a complete refactoring into a production-ready framework with modern CLI, configuration management, and modular architecture.

## ✨ What's New in v2.0

- **🎯 Modern CLI**: Replace monolithic scripts with clean commands (`biofuse train`, `biofuse cache`, etc.)
- **⚙️ Configuration System**: YAML/JSON-based experiment configs with validation
- **🏗️ Modular Architecture**: Clean separation into core, data, classifiers, evaluation, and utils
- **📦 Proper Packaging**: Install via pip with console scripts
- **🚀 Smart Caching**: Configurable embedding cache with versioning
- **🧪 Unified Classifiers**: Factory pattern for LogisticRegression, XGBoost, CatBoost, Neural Nets
- **📊 Comprehensive Evaluation**: Built-in metrics, cross-validation, robustness testing

## 🚀 Quick Start

### Installation

\`\`\`bash
# Clone repository
git clone https://github.com/mnhcorp/biofuse.git
cd biofuse

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
\`\`\`

### Basic Usage

**Option 1: Configuration File (Recommended)**

Create `experiment.yaml`:
\`\`\`yaml
name: pathmnist_experiment
data:
  dataset: pathmnist
  img_size: 224
model:
  models: [BioMedCLIP, CONCH]
  fusion_method: concat
classifier:
  type: xgboost
\`\`\`

Run:
\`\`\`bash
biofuse train --config experiment.yaml
\`\`\`

**Option 2: CLI Arguments**

\`\`\`bash
biofuse train --dataset pathmnist --models BioMedCLIP,CONCH --classifier xgboost
\`\`\`

**Option 3: Python API**

\`\`\`python
from biofuse import BioFuse, load_medmnist, get_classifier

# Load data
train_data, num_classes = load_medmnist('pathmnist', split='train')

# Generate embeddings
biofuse = BioFuse(models=['BioMedCLIP', 'CONCH'])
train_emb, train_labels, _, _, _ = biofuse.generate_embeddings(
    train_data=None,
    dataset_type='medmnist',
    dataset_name='pathmnist'
)

# Train & evaluate
classifier = get_classifier('xgboost')
classifier.fit(train_emb, train_labels)
\`\`\`

## 📚 Supported Models & Datasets

### Models (12+)
BioMedCLIP, CONCH, UNI/UNI2, rad-dino, Prov-GigaPath, PubMedCLIP, Hibou-B, CheXagent, BioMistral, LLama-3-Aloe, CLIP

### Datasets  
- **MedMNIST**: All 12 variants (PathMNIST, ChestMNIST, etc.)
- **ImageNet-1K**, **BUSI**, **Custom directories**

### Classifiers
`logistic`, `xgboost`, `catboost`, `nn_mlp`, `nn_cnn`, `nn_resnet`

## 🎮 CLI Commands

\`\`\`bash
# Train
biofuse train --config experiment.yaml
biofuse train -d pathmnist -m BioMedCLIP,CONCH

# Cache management  
biofuse cache list
biofuse cache clear --dataset pathmnist

# System info
biofuse info
\`\`\`

## 📖 Documentation

See full documentation:
- **Configuration**: See `examples/` for config templates
- **Python API**: See docstrings in `biofuse/`
- **Migration Guide**: Upgrade from v0.1 in `MIGRATION.md`

## 📊 Pre-computed Embeddings

- **MedMNIST**: [10.5281/zenodo.13952293](https://doi.org/10.5281/zenodo.13952293)
- **ImageNet-1K**: [10.5281/zenodo.14930584](https://doi.org/10.5281/zenodo.14930584)

## 🔄 Migration from v0.1

Old (v0.1):
\`\`\`bash
python tests/test_linear_probe_trainable2.py --dataset pathmnist --models BC,CO
\`\`\`

New (v2.0):
\`\`\`bash
biofuse train --dataset pathmnist --models BioMedCLIP,CONCH
\`\`\`

## 🏗️ Architecture

\`\`\`
biofuse/
├── core/           # Cache, utilities
├── models/         # Embedding extractors, fusion
├── data/           # Dataset loaders
├── classifiers/    # Unified classifier interface
├── evaluation/     # Metrics, evaluators
├── config/         # Configuration management
├── cli/            # Command-line interface
└── utils/          # Logging, paths, reproducibility
\`\`\`

## 📧 Contact

**Mirza Hossain** - mnh3@st-andrews.ac.uk  
GitHub: [@mnhcorp](https://github.com/mnhcorp/biofuse)

## 📜 License

MIT License - see [LICENSE](LICENSE)
