# BioFuse: An Embedding Fusion Framework for Biomedical Foundation Models

BioFuse is a multi-modal fusion framework designed specifically for biomedical images. It enables the combination of embeddings from multiple foundation models to improve performance on downstream tasks.

## Overview

BioFuse allows researchers and practitioners to:

1. Extract embeddings from multiple pre-trained foundation models
2. Fuse these embeddings using various fusion methods
3. Apply the fused embeddings to downstream tasks like classification
4. Evaluate performance across different model combinations

The framework is particularly effective for biomedical imaging tasks where different foundation models may capture complementary aspects of the data.

## Key Features

- **Fusion Methods**: Officially supports concatenation fusion, with experimental support for other methods
- **Projection Layers**: Optional learnable projection layers to transform embeddings before fusion
- **Caching System**: Efficient embedding caching to speed up experiments
- **Comprehensive Evaluation**: Tools for evaluating model performance with metrics like accuracy and AUC-ROC
- **Support for Various Datasets**: Works with MedMNIST datasets and custom datasets

### Evaluation

The repository includes evaluation scripts in the `tests/` directory:

- `test_linear_probe_trainable2.py`: The main evaluation script that supports multiple classifier options and datasets

## Usage

```bash
python tests/test_linear_probe_trainable2.py --dataset chestmnist --img_size 224 --models BioMedCLIP,CONCH --fusion_methods concat --projections 512
```

## Supported Models

BioFuse supports a variety of foundation models, including:

- BioMedCLIP (512-dim)
- BioMistral (4096-dim)
- CheXagent (1408-dim)
- CONCH (512-dim)
- LLama-3-Aloe (4096-dim)
- Prov-GigaPath (1536-dim)
- PubMedCLIP (512-dim)
- rad-dino (768-dim)
- UNI (1024-dim)
- UNI2 (1536-dim)
- Hibou-B (768-dim)
- CLIP (512-dim)

## Pre-computed Embeddings

To accelerate experimentation, we provide pre-computed embeddings for various datasets:

- **MedMNIST Embeddings**: [https://doi.org/10.5281/zenodo.13952293](https://doi.org/10.5281/zenodo.13952293)
- **ImageNet-1K Embeddings**: [https://doi.org/10.5281/zenodo.14930584](https://doi.org/10.5281/zenodo.14930584)

These can be downloaded and placed in the `/data/biofuse-embedding-cache` directory to avoid recomputing embeddings.

## Supported Datasets

The framework has been tested on various biomedical datasets:

- MedMNIST datasets (PathMNIST, DermaMNIST, BreastMNIST, ChestMNIST, etc.)
- ImageNet-1K

## Installation

TBD.

## High-level API

TBD.
