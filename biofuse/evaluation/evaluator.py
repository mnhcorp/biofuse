"""
Evaluation orchestration for BioFuse.

Provides high-level evaluation functions that combine data loading,
model prediction, and metric computation.
"""

from typing import Dict, Any, Optional, Tuple
import time
import numpy as np
from ..utils.logging import ExperimentLogger
from .metrics import compute_metrics, evaluate_classifier


class Evaluator:
    """
    Orchestrates evaluation of fusion models and classifiers.

    Handles the complete evaluation pipeline including metric computation,
    logging, and result aggregation.
    """

    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        verbose: bool = True
    ):
        """
        Initialize evaluator.

        Args:
            logger: Optional experiment logger
            verbose: Whether to print progress
        """
        self.logger = logger
        self.verbose = verbose

    def evaluate_classifier(
        self,
        classifier,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        dataset: Optional[str] = None,
        split_names: Tuple[str, str] = ('validation', 'test')
    ) -> Dict[str, Any]:
        """
        Evaluate a classifier on validation and/or test data.

        Args:
            classifier: Trained classifier
            X_val: Validation features
            y_val: Validation labels
            X_test: Optional test features
            y_test: Optional test labels
            dataset: Dataset name
            split_names: Names for the splits (for logging)

        Returns:
            Dictionary of results for each split

        Example:
            >>> evaluator = Evaluator()
            >>> results = evaluator.evaluate_classifier(
            ...     classifier, X_val, y_val, X_test, y_test, 'pathmnist'
            ... )
            >>> print(results['validation']['accuracy'])
        """
        results = {}

        # Evaluate on validation set
        if self.verbose:
            print(f"\nEvaluating on {split_names[0]} set...")

        val_start = time.time()
        val_metrics = compute_metrics(classifier, X_val, y_val, dataset)
        val_time = time.time() - val_start

        results[split_names[0]] = {
            **val_metrics,
            'inference_time': val_time,
            'num_samples': len(X_val)
        }

        if self.verbose:
            self._print_metrics(split_names[0], val_metrics)

        if self.logger:
            self.logger.log_metrics(
                {f'{split_names[0]}_{k}': v for k, v in val_metrics.items()}
            )

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            if self.verbose:
                print(f"\nEvaluating on {split_names[1]} set...")

            test_start = time.time()
            test_metrics = compute_metrics(classifier, X_test, y_test, dataset)
            test_time = time.time() - test_start

            results[split_names[1]] = {
                **test_metrics,
                'inference_time': test_time,
                'num_samples': len(X_test)
            }

            if self.verbose:
                self._print_metrics(split_names[1], test_metrics)

            if self.logger:
                self.logger.log_metrics(
                    {f'{split_names[1]}_{k}': v for k, v in test_metrics.items()}
                )

        return results

    def evaluate_fusion(
        self,
        biofuse_model,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: np.ndarray,
        val_labels: np.ndarray,
        test_embeddings: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        classifier_fn=None,
        dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a complete BioFuse pipeline: embeddings → fusion → classifier.

        Args:
            biofuse_model: Fusion model
            train_embeddings: Training embeddings
            train_labels: Training labels
            val_embeddings: Validation embeddings
            val_labels: Validation labels
            test_embeddings: Optional test embeddings
            test_labels: Optional test labels
            classifier_fn: Function that takes (X_train, y_train) and returns
                          a trained classifier
            dataset: Dataset name

        Returns:
            Dictionary with training info and evaluation results

        Example:
            >>> def train_xgb(X, y):
            ...     from biofuse.classifiers import XGBoostClassifier
            ...     clf = XGBoostClassifier()
            ...     return clf.fit(X, y)
            >>> results = evaluator.evaluate_fusion(
            ...     model, X_train, y_train, X_val, y_val,
            ...     classifier_fn=train_xgb, dataset='pathmnist'
            ... )
        """
        results = {
            'dataset': dataset,
            'num_train_samples': len(train_embeddings),
            'num_val_samples': len(val_embeddings),
            'embedding_dim': train_embeddings.shape[1]
        }

        # Train classifier
        if self.verbose:
            print("\nTraining classifier on fused embeddings...")

        train_start = time.time()

        if classifier_fn is None:
            # Default: use logistic regression
            from ..classifiers import get_classifier
            classifier = get_classifier('logistic')
            classifier.fit(train_embeddings, train_labels)
        else:
            classifier = classifier_fn(train_embeddings, train_labels)

        train_time = time.time() - train_start

        results['training_time'] = train_time
        results['classifier_type'] = type(classifier).__name__

        if self.verbose:
            print(f"Classifier training time: {train_time:.2f}s")

        # Evaluate classifier
        eval_results = self.evaluate_classifier(
            classifier,
            val_embeddings,
            val_labels,
            test_embeddings,
            test_labels,
            dataset
        )

        results['evaluation'] = eval_results

        return results

    def _print_metrics(self, split_name: str, metrics: Dict[str, Any]):
        """
        Print metrics in a formatted way.

        Args:
            split_name: Name of the data split
            metrics: Dictionary of metrics
        """
        print(f"\n{split_name.capitalize()} Results:")
        print("-" * 40)

        for metric_name, value in metrics.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"{metric_name}: {value}")


class CrossValidator:
    """
    Handles cross-validation for BioFuse models.

    Provides k-fold and stratified k-fold cross-validation.
    """

    def __init__(
        self,
        n_folds: int = 5,
        stratified: bool = True,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize cross-validator.

        Args:
            n_folds: Number of folds
            stratified: Whether to use stratified folds
            random_state: Random seed
            verbose: Whether to print progress
        """
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_state = random_state
        self.verbose = verbose

    def cross_validate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        classifier_fn,
        dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            embeddings: Input embeddings
            labels: Labels
            classifier_fn: Function that takes (X_train, y_train) and returns
                          a trained classifier
            dataset: Dataset name

        Returns:
            Dictionary with cross-validation results

        Example:
            >>> cv = CrossValidator(n_folds=5)
            >>> results = cv.cross_validate(embeddings, labels, train_xgb)
            >>> print(results['mean_accuracy'])
        """
        from sklearn.model_selection import StratifiedKFold, KFold

        # Create folder
        if self.stratified:
            kfold = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            kfold = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(embeddings, labels)):
            if self.verbose:
                print(f"\n{'=' * 50}")
                print(f"Fold {fold + 1}/{self.n_folds}")
                print(f"{'=' * 50}")

            # Split data
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Train classifier
            classifier = classifier_fn(X_train, y_train)

            # Evaluate
            evaluator = Evaluator(verbose=self.verbose)
            results = evaluator.evaluate_classifier(
                classifier, X_val, y_val, dataset=dataset, split_names=('validation',)
            )

            fold_results.append(results['validation'])

        # Aggregate results
        aggregated = self._aggregate_fold_results(fold_results)

        return aggregated

    def _aggregate_fold_results(self, fold_results: list) -> Dict[str, Any]:
        """
        Aggregate results across folds.

        Args:
            fold_results: List of results from each fold

        Returns:
            Dictionary with mean and std of metrics
        """
        # Extract metrics from all folds
        metric_names = fold_results[0].keys()
        aggregated = {}

        for metric in metric_names:
            values = [fold[metric] for fold in fold_results
                     if fold[metric] is not None]

            if values:
                aggregated[f'mean_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)

        aggregated['fold_results'] = fold_results

        return aggregated
