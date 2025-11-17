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

**Option 1: Install directly from GitHub (Recommended)**

```bash
# Install latest from main branch
pip install git+https://github.com/mnhcorp/biofuse.git

# Or install from specific branch
pip install git+https://github.com/mnhcorp/biofuse.git@claude/v2.0-011CUzcaafUUp1M9VqiHkhj9
```

**Option 2: Clone and install locally**

```bash
# Clone repository
git clone https://github.com/mnhcorp/biofuse.git
cd biofuse

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### 🔑 HuggingFace Authentication (Required for Gated Models)

Many foundation models (UNI, CONCH, CheXagent, etc.) are **gated** and require HuggingFace authentication.

**Setup your HuggingFace token (choose one method):**

**Method 1: HuggingFace CLI (Recommended)**
```bash
# Install HF CLI if not already installed
pip install huggingface_hub

# Login interactively
huggingface-cli login
```

**Method 2: Environment Variable**
```bash
# Add to your ~/.bashrc or ~/.zshrc
export HF_TOKEN="hf_your_token_here"

# Or set for current session
export HF_TOKEN="hf_your_token_here"
```

**Method 3: .env File**
```bash
# Copy example file
cp .env.example .env

# Edit .env and add your token
nano .env
# Add: HF_TOKEN=hf_your_token_here
```

**Get your HuggingFace token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Accept model access agreements for gated models you want to use:
   - [UNI](https://huggingface.co/MahmoodLab/UNI)
   - [CONCH](https://huggingface.co/MahmoodLab/CONCH)
   - [CheXagent](https://huggingface.co/StanfordAIMI/CheXagent-8b)

**Token Priority (BioFuse checks in this order):**
1. `HF_TOKEN` environment variable
2. `HUGGINGFACE_TOKEN` environment variable
3. `~/.huggingface/token` (from `huggingface-cli login`)
4. `.env` file in project root

### ⚙️ Configuration (Optional)

**Cache Directories:**

BioFuse uses two types of caches:

1. **HuggingFace Model Cache** (for downloaded models)
   - Default: `~/.cache/huggingface` or `/data/hf-hub` (if writable)
   - Override: Set `HF_HOME` environment variable

2. **BioFuse Embedding Cache** (for pre-computed embeddings)
   - Default: `/data/biofuse-embedding-cache` or `~/biofuse-cache`
   - Override: Set `BIOFUSE_CACHE_DIR` environment variable

```bash
# Example: Custom cache directories
export HF_HOME="/mnt/storage/hf-models"
export BIOFUSE_CACHE_DIR="/mnt/storage/biofuse-cache"
```

**All configuration can be set in `.env` file:**
```bash
cp .env.example .env
# Edit .env with your settings
```

### Basic Usage

**Option 1: Configuration File (Recommended)**

Create `experiment.yaml`:
```yaml
name: pathmnist_experiment
data:
  dataset: pathmnist
  img_size: 224
model:
  models: [BioMedCLIP, CONCH]
  fusion_method: concat
classifier:
  type: xgboost
```

Run:
```bash
biofuse train --config experiment.yaml
```

**Option 2: CLI Arguments**

```bash
biofuse train --dataset pathmnist --models BioMedCLIP,CONCH --classifier xgboost
```

**Option 3: Python API**

```python
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
```

## 📚 Supported Models & Datasets

### Models (12+)
BioMedCLIP, CONCH, UNI/UNI2, rad-dino, Prov-GigaPath, PubMedCLIP, Hibou-B, CheXagent, BioMistral, LLama-3-Aloe, CLIP

### Datasets  
- **MedMNIST**: All 12 variants (PathMNIST, ChestMNIST, etc.)
- **ImageNet-1K**, **BUSI**, **Custom directories**

### Classifiers
`logistic`, `xgboost`, `catboost`, `nn_mlp`, `nn_cnn`, `nn_resnet`

## 🎮 CLI Commands

```bash
# Train
biofuse train --config experiment.yaml
biofuse train -d pathmnist -m BioMedCLIP,CONCH

# Cache management  
biofuse cache list
biofuse cache clear --dataset pathmnist

# System info
biofuse info
```

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
```bash
python tests/test_linear_probe_trainable2.py --dataset pathmnist --models BC,CO
```

New (v2.0):
```bash
biofuse train --dataset pathmnist --models BioMedCLIP,CONCH
```

## 🏗️ Architecture

```
biofuse/
├── core/           # Cache, utilities
├── models/         # Embedding extractors, fusion
├── data/           # Dataset loaders
├── classifiers/    # Unified classifier interface
├── evaluation/     # Metrics, evaluators
├── config/         # Configuration management
├── cli/            # Command-line interface
└── utils/          # Logging, paths, reproducibility
```

## 📧 Contact

**Mirza Hossain** - mnh3@st-andrews.ac.uk  
GitHub: [@mnhcorp](https://github.com/mnhcorp/biofuse)

## 📜 License

MIT License - see [LICENSE](LICENSE)
