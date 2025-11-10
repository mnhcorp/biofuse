"""
BioFuse: Multi-modal Fusion Framework for Biomedical Foundation Models.

BioFuse enables combining embeddings from multiple pre-trained foundation models
to improve performance on biomedical imaging tasks.
"""

__version__ = '0.2.0'

# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name):
    """Lazy import for module-level attributes."""
    # Core API
    if name == 'BioFuse':
        from .biofuse import BioFuse
        return BioFuse

    # Utilities
    if name == 'set_seed':
        from .utils import set_seed
        return set_seed
    if name == 'get_device':
        from .utils import get_device
        return get_device
    if name == 'PathManager':
        from .utils import PathManager
        return PathManager
    if name == 'ExperimentLogger':
        from .utils import ExperimentLogger
        return ExperimentLogger

    # Classifiers
    if name == 'get_classifier':
        from .classifiers import get_classifier
        return get_classifier
    if name == 'BaseClassifier':
        from .classifiers import BaseClassifier
        return BaseClassifier

    # Evaluation
    if name == 'Evaluator':
        from .evaluation import Evaluator
        return Evaluator
    if name == 'compute_metrics':
        from .evaluation import compute_metrics
        return compute_metrics

    # Data
    if name == 'load_medmnist':
        from .data import load_medmnist
        return load_medmnist
    if name == 'load_imagenet':
        from .data import load_imagenet
        return load_imagenet
    if name == 'BioFuseImageDataset':
        from .data import BioFuseImageDataset
        return BioFuseImageDataset

    # Cache
    if name == 'EmbeddingCache':
        from .core import EmbeddingCache
        return EmbeddingCache
    if name == 'get_cache':
        from .core import get_cache
        return get_cache

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
