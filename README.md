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

## Usage

### Basic Example

