"""
Classifier modules for BioFuse.

Imports heavy classifier backends lazily so the CLI can start without every
optional ML package installed.
"""

from .base import BaseClassifier, ClassifierFactory


def _ensure_registry_loaded():
    """Load classifier implementations on demand."""
    from . import neural  # noqa: F401
    from . import sklearn_classifiers  # noqa: F401


def get_classifier(name: str, **kwargs):
    """Create a classifier by name, loading backends lazily."""
    _ensure_registry_loaded()
    return ClassifierFactory.create(name, **kwargs)


def __getattr__(name):
    if name in {
        'LogisticRegression',
        'XGBoostClassifier',
        'CatBoostClassifierWrapper',
    }:
        from . import sklearn_classifiers as sklearn_module
        return getattr(sklearn_module, name)

    if name == 'NeuralNetClassifier':
        from . import neural as neural_module
        return getattr(neural_module, name)

    if name in {'BaseClassifier', 'ClassifierFactory', 'get_classifier'}:
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'BaseClassifier',
    'ClassifierFactory',
    'get_classifier',
    'LogisticRegression',
    'XGBoostClassifier',
    'CatBoostClassifierWrapper',
    'NeuralNetClassifier',
]
