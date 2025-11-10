"""
Classifier modules for BioFuse.

Provides sklearn-based and neural network-based classifiers with a unified
interface for training and evaluation.
"""

from .base import BaseClassifier, ClassifierFactory, get_classifier
from .sklearn_classifiers import (
    LogisticRegression,
    XGBoostClassifier,
    CatBoostClassifierWrapper
)
from .neural import NeuralNetClassifier

__all__ = [
    'BaseClassifier',
    'ClassifierFactory',
    'get_classifier',
    'LogisticRegression',
    'XGBoostClassifier',
    'CatBoostClassifierWrapper',
    'NeuralNetClassifier',
]
