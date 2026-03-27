"""
Scikit-learn based classifiers for BioFuse.

Includes logistic regression, XGBoost, and CatBoost classifiers with
sensible defaults for biomedical imaging tasks.
"""

import inspect
import time
from typing import Optional, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

from .base import BaseClassifier, ClassifierFactory


def _supports_kwarg(callable_obj, parameter_name: str) -> bool:
    """Check whether a backend constructor still accepts a specific kwarg."""
    return parameter_name in inspect.signature(callable_obj).parameters


class LogisticRegression(BaseClassifier):
    """
    Logistic Regression classifier wrapper.

    Uses sklearn's LogisticRegression with sensible defaults for
    high-dimensional features from foundation models.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        multi_class: str = 'auto',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Logistic Regression classifier.

        Args:
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            multi_class: Multi-class strategy ('ovr', 'multinomial', 'auto')
            random_state: Random seed
            **kwargs: Additional parameters passed to sklearn LogisticRegression
        """
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.random_state = random_state
        self.kwargs = kwargs

        self.scaler = StandardScaler()
        self.classifier = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'LogisticRegression':
        """Train the logistic regression classifier."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Determine number of classes
        self.num_classes = len(np.unique(y))
        self.multi_label = len(y.shape) > 1 and y.shape[1] > 1

        # Create classifier
        classifier_kwargs = dict(self.kwargs)
        if self.multi_class is not None and _supports_kwarg(SklearnLogisticRegression, 'multi_class'):
            classifier_kwargs['multi_class'] = self.multi_class

        self.classifier = SklearnLogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **classifier_kwargs
        )

        # Train
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'multi_class': self.multi_class,
            'random_state': self.random_state,
            **self.kwargs
        }


class XGBoostClassifier(BaseClassifier):
    """
    XGBoost classifier wrapper with auto-tuning based on dataset size.

    Automatically configures parameters based on the number of samples
    and supports binary, multi-class, and multi-label classification.
    """

    def __init__(
        self,
        n_estimators: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
        tree_method: str = 'auto',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds (auto-tuned if None)
            learning_rate: Learning rate (auto-tuned if None)
            max_depth: Maximum tree depth (auto-tuned if None)
            n_jobs: Number of parallel threads (-1 for all cores)
            tree_method: Tree construction algorithm ('auto', 'gpu_hist', etc.)
            random_state: Random seed
            **kwargs: Additional XGBoost parameters
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.random_state = random_state
        self.kwargs = kwargs

        self.scaler = StandardScaler()
        self.classifier = None
        self.training_time = 0

    def _get_auto_params(self, n_samples: int) -> Dict[str, Any]:
        """
        Get auto-tuned parameters based on dataset size.

        Args:
            n_samples: Number of training samples

        Returns:
            Dictionary of XGBoost parameters
        """
        # Use provided params if available
        if all(p is not None for p in [self.n_estimators, self.learning_rate, self.max_depth]):
            return {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth
            }

        # Default configuration optimized for biomedical datasets
        params = {
            'n_estimators': self.n_estimators or 250,
            'learning_rate': self.learning_rate or 0.1,
            'max_depth': self.max_depth or 6
        }

        return params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'XGBoostClassifier':
        """Train the XGBoost classifier."""
        start_time = time.time()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Determine task type
        self.num_classes = len(np.unique(y))
        self.multi_label = len(y.shape) > 1 and y.shape[1] > 1

        # Get auto-tuned parameters
        auto_params = self._get_auto_params(len(X))

        # Configure for task type
        if self.num_classes > 2 and not self.multi_label:
            # Multi-class
            self.classifier = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=self.num_classes,
                eval_metric='mlogloss',
                n_estimators=auto_params['n_estimators'],
                learning_rate=auto_params['learning_rate'],
                max_depth=auto_params['max_depth'],
                n_jobs=self.n_jobs,
                tree_method=self.tree_method,
                random_state=self.random_state,
                use_label_encoder=False,
                **self.kwargs
            )
        else:
            # Binary
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=auto_params['n_estimators'],
                learning_rate=auto_params['learning_rate'],
                max_depth=auto_params['max_depth'],
                n_jobs=self.n_jobs,
                tree_method=self.tree_method,
                random_state=self.random_state,
                use_label_encoder=False,
                **self.kwargs
            )

            if self.multi_label:
                self.classifier = OneVsRestClassifier(xgb_model)
            else:
                self.classifier = xgb_model

        # Train
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True

        self.training_time = time.time() - start_time

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'n_jobs': self.n_jobs,
            'tree_method': self.tree_method,
            'random_state': self.random_state,
            **self.kwargs
        }


class CatBoostClassifierWrapper(BaseClassifier):
    """
    CatBoost classifier wrapper.

    Provides a unified interface for CatBoost with sensible defaults
    for biomedical tasks.
    """

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.1,
        depth: int = 6,
        task_type: str = 'CPU',
        devices: str = '0',
        verbose: int = 100,
        thread_count: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize CatBoost classifier.

        Args:
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Tree depth
            task_type: 'CPU' or 'GPU'
            devices: GPU device IDs (for GPU mode)
            verbose: Verbosity frequency
            thread_count: Number of threads (-1 for auto)
            random_state: Random seed
            **kwargs: Additional CatBoost parameters
        """
        super().__init__()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.task_type = task_type
        self.devices = devices
        self.verbose = verbose
        self.thread_count = thread_count
        self.random_state = random_state
        self.kwargs = kwargs

        self.scaler = StandardScaler()
        self.classifier = None
        self.training_time = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'CatBoostClassifierWrapper':
        """Train the CatBoost classifier."""
        start_time = time.time()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Determine task type
        self.num_classes = len(np.unique(y))
        self.multi_label = len(y.shape) > 1 and y.shape[1] > 1

        # Base parameters
        base_params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'task_type': self.task_type,
            'verbose': self.verbose,
            'random_state': self.random_state,
            **self.kwargs
        }

        if self.task_type == 'GPU':
            base_params['devices'] = self.devices

        if self.thread_count > 0:
            base_params['thread_count'] = self.thread_count

        # Configure for task type
        if self.num_classes > 2 and not self.multi_label:
            # Multi-class
            self.classifier = CatBoostClassifier(
                **base_params,
                loss_function='MultiClass',
                classes_count=self.num_classes,
                eval_metric='MultiClass'
            )
        else:
            # Binary
            cat_model = CatBoostClassifier(
                **base_params,
                loss_function='Logloss',
                eval_metric='AUC'
            )

            if self.multi_label:
                self.classifier = OneVsRestClassifier(cat_model)
            else:
                self.classifier = cat_model

        # Train
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True

        self.training_time = time.time() - start_time

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'task_type': self.task_type,
            'devices': self.devices,
            'verbose': self.verbose,
            'thread_count': self.thread_count,
            'random_state': self.random_state,
            **self.kwargs
        }


# Register classifiers
ClassifierFactory.register('logistic', LogisticRegression)
ClassifierFactory.register('logreg', LogisticRegression)
ClassifierFactory.register('xgboost', XGBoostClassifier)
ClassifierFactory.register('xgb', XGBoostClassifier)
ClassifierFactory.register('catboost', CatBoostClassifierWrapper)
ClassifierFactory.register('cat', CatBoostClassifierWrapper)
