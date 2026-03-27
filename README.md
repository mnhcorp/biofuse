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

### New User Path

If you are new to BioFuse, use this order:

1. Run a tiny Docker smoke test
2. Run a small MedMNIST smoke test
3. Run a real training job

**Step 1: Build the image**

```bash
git clone https://github.com/mnhcorp/biofuse.git
cd biofuse
docker build -t biofuse .
```

**Step 2: Run the smallest possible end-to-end check**

```bash
docker run --rm --gpus all biofuse
```

This runs:

```bash
biofuse smoke --preset custom --device cuda
```

It uses the repository sample images and verifies that BioFuse can install, launch the CLI, load a model, extract embeddings, train a classifier, and write results.

**Step 3: Run a small MedMNIST smoke test**

```bash
docker run --rm --gpus all \
  -v "$PWD/.cache/medmnist:/data/medmnist" \
  -v "$PWD/results:/workspace/results" \
  biofuse \
  biofuse smoke \
    --preset medmnist \
    --dataset pathmnist \
    --models CLIP \
    --device cuda \
    --max-train-samples 64 \
    --max-val-samples 32 \
    --max-test-samples 32 \
    --output-dir /workspace/results
```

This is the recommended first real dataset run:
- dataset: `PathMNIST`
- encoder: `CLIP`
- classifier: `logistic`
- download behavior: automatic if missing
- runtime: bounded to a small subset

**Step 4: Run a normal training job**

```bash
docker run --rm --gpus all \
  -v "$PWD/.cache/medmnist:/data/medmnist" \
  -v "$PWD/results:/workspace/results" \
  biofuse \
  biofuse train \
    --dataset pathmnist \
    --models CLIP \
    --classifier logistic \
    --device cuda \
    --data-root /data/medmnist \
    --output-dir /workspace/results
```

### Local Installation

If you do not want Docker, install locally:

```bash
git clone https://github.com/mnhcorp/biofuse.git
cd biofuse
pip install -e ".[dev]"
```

Then inspect the CLI:

```bash
biofuse --help
biofuse smoke --help
biofuse train --help
biofuse info
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

**Option 1: Smoke Test Through The CLI**

```bash
biofuse smoke --preset custom
biofuse smoke --preset medmnist --dataset pathmnist --models CLIP --max-train-samples 64
```

**Option 2: Train From CLI Arguments**

```bash
biofuse train --dataset pathmnist --models CLIP --classifier logistic
biofuse train --dataset pathmnist --models BioMedCLIP --classifier xgboost
```

**Option 3: Configuration File**

Create `experiment.yaml`:

```yaml
name: pathmnist_experiment
data:
  dataset: pathmnist
  img_size: 224
model:
  models: [BioMedCLIP]
  fusion_method: concat
classifier:
  type: xgboost
```

Run:

```bash
biofuse train --config experiment.yaml
```

**Option 4: Python API**

```python
from biofuse import BioFuse

biofuse = BioFuse(models=['CLIP'], device='cuda')
train_emb, train_labels, _, _, _ = biofuse.generate_embeddings(
    train_data=None,
    dataset_type='medmnist',
    dataset_name='pathmnist'
)
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
# Smoke tests
biofuse smoke --preset custom
biofuse smoke --preset medmnist --dataset pathmnist --models CLIP

# Train
biofuse train --config experiment.yaml
biofuse train -d pathmnist -m BioMedCLIP
biofuse train -d pathmnist -m CLIP --max-train-samples 64 --max-val-samples 32

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
