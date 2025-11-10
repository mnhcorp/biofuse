"""
Base classifier interface for BioFuse.

Provides abstract base class for all classifiers to ensure consistent API.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers in BioFuse.

    All classifiers (sklearn-based and neural network-based) should
    inherit from this class and implement the required methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the classifier.

        Args:
            **kwargs: Classifier-specific configuration parameters
        """
        self.is_fitted = False
        self.num_classes = None
        self.multi_label = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'BaseClassifier':
        """
        Train the classifier on the given data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,) or (n_samples, n_classes)
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,) or (n_samples, n_classes)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X: Test features
            y: True labels

        Returns:
            Mean accuracy
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self) -> Dict[str, Any]:
        """
        Get classifier parameters.

        Returns:
            Dictionary of parameter names and values
        """
        return {}

    def set_params(self, **params) -> 'BaseClassifier':
        """
        Set classifier parameters.

        Args:
            **params: Parameter names and values

        Returns:
            Self (for method chaining)
        """
        return self

    def __repr__(self) -> str:
        """String representation of the classifier."""
        class_name = self.__class__.__name__
        params = self.get_params()
        if params:
            params_str = ', '.join(f"{k}={v}" for k, v in params.items())
            return f"{class_name}({params_str})"
        return f"{class_name}()"


class ClassifierFactory:
    """
    Factory for creating classifiers by name.

    Provides a registry of available classifiers and convenience
    methods for instantiation.
    """

    _registry = {}

    @classmethod
    def register(cls, name: str, classifier_class: type):
        """
        Register a classifier class.

        Args:
            name: Name to register the classifier under
            classifier_class: Classifier class to register
        """
        cls._registry[name] = classifier_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseClassifier:
        """
        Create a classifier instance by name.

        Args:
            name: Registered classifier name
            **kwargs: Parameters to pass to classifier constructor

        Returns:
            Classifier instance

        Raises:
            ValueError: If classifier name is not registered
        """
        if name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unknown classifier '{name}'. "
                f"Available classifiers: {available}"
            )

        classifier_class = cls._registry[name]
        return classifier_class(**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """
        List all available classifier names.

        Returns:
            List of registered classifier names
        """
        return list(cls._registry.keys())


def get_classifier(name: str, **kwargs) -> BaseClassifier:
    """
    Convenience function to create a classifier by name.

    Args:
        name: Classifier name
        **kwargs: Parameters to pass to classifier constructor

    Returns:
        Classifier instance

    Example:
        >>> classifier = get_classifier('xgboost', n_estimators=100)
        >>> classifier.fit(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
    """
    return ClassifierFactory.create(name, **kwargs)
