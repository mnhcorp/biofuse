"""
BioFuse: Multi-modal Fusion Framework for Biomedical Foundation Models.

BioFuse enables combining embeddings from multiple pre-trained foundation models
to improve performance on biomedical imaging tasks.
"""

__version__ = '0.2.0'

# Core API
from .biofuse import BioFuse

# Utilities
from .utils import set_seed, get_device, PathManager, ExperimentLogger

# Classifiers
from .classifiers import get_classifier, BaseClassifier

# Evaluation
from .evaluation import Evaluator, compute_metrics

# Data
from .data import load_medmnist, load_imagenet, BioFuseImageDataset

# Cache
from .core import EmbeddingCache, get_cache

__all__ = [
    # Main API
    'BioFuse',
    # Utilities
    'set_seed',
    'get_device',
    'PathManager',
    'ExperimentLogger',
    # Classifiers
    'get_classifier',
    'BaseClassifier',
    # Evaluation
    'Evaluator',
    'compute_metrics',
    # Data
    'load_medmnist',
    'load_imagenet',
    'BioFuseImageDataset',
    # Cache
    'EmbeddingCache',
    'get_cache',
]
