"""
Evaluation metrics for BioFuse.

Provides functions to compute accuracy, AUC-ROC, and other metrics for
binary, multi-class, and multi-label classification tasks.
"""

from typing import Tuple, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multi_label: bool = False
) -> float:
    """
    Compute accuracy for predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        multi_label: Whether this is a multi-label problem

    Returns:
        Accuracy score

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> compute_accuracy(y_true, y_pred)
        0.75
    """
    # Ensure arrays are 1D for single-label tasks
    if not multi_label:
        if len(y_true.shape) > 1:
            y_true = y_true.squeeze()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.squeeze()

        return accuracy_score(y_true, y_pred)

    # Multi-label: compute per-label accuracy and average
    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        # Not actually multi-label
        return accuracy_score(y_true, y_pred)

    accuracies = []
    for i in range(y_true.shape[1]):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies.append(acc)

    return np.mean(accuracies)


def compute_auc_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    multi_class: str = 'ovr',
    multi_label: bool = False
) -> float:
    """
    Compute AUC-ROC score.

    Args:
        y_true: True labels
        y_score: Predicted probabilities
        multi_class: Strategy for multi-class ('ovr' or 'ovo')
        multi_label: Whether this is a multi-label problem

    Returns:
        AUC-ROC score

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_score = np.array([0.1, 0.9, 0.8, 0.2])
        >>> compute_auc_roc(y_true, y_score)
        1.0
    """
    # Determine number of classes
    num_classes = len(np.unique(y_true))

    if num_classes == 2 and not multi_label:
        # Binary classification
        if len(y_score.shape) > 1 and y_score.shape[1] == 2:
            # Take probability of positive class
            y_score = y_score[:, 1]
        return roc_auc_score(y_true, y_score)

    # Multi-label classification
    if multi_label or (len(y_true.shape) > 1 and y_true.shape[1] > 1):
        auc_scores = []
        for i in range(y_true.shape[1]):
            # Only compute AUC if there are both positive and negative examples
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
                auc_scores.append(auc)

        return np.mean(auc_scores) if auc_scores else 0.0

    # Multi-class classification
    return roc_auc_score(y_true, y_score, multi_class=multi_class)


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy (useful for ImageNet-like datasets).

    Args:
        y_true: True labels
        y_score: Predicted probabilities
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy

    Example:
        >>> y_true = np.array([2, 1, 0])
        >>> y_score = np.array([
        ...     [0.1, 0.2, 0.3, 0.4],  # Top-2: [3, 2] -> Correct (2 in top-2)
        ...     [0.4, 0.3, 0.2, 0.1],  # Top-2: [0, 1] -> Correct (1 in top-2)
        ...     [0.1, 0.2, 0.3, 0.4],  # Top-2: [3, 2] -> Wrong (0 not in top-2)
        ... ])
        >>> compute_top_k_accuracy(y_true, y_score, k=2)
        0.6666...
    """
    # Get top-k predictions
    top_k_preds = np.argsort(y_score, axis=1)[:, -k:]

    # Check if true label is in top-k predictions
    correct = [true_label in pred_top_k
               for true_label, pred_top_k in zip(y_true, top_k_preds)]

    return np.mean(correct)


def evaluate_classifier(
    classifier,
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: Optional[str] = None,
    return_proba: bool = False
) -> Union[float, Tuple[float, float], dict]:
    """
    Comprehensive evaluation of a classifier.

    Args:
        classifier: Trained classifier with predict and predict_proba methods
        X: Input features
        y_true: True labels
        dataset: Dataset name (special handling for 'imagenet')
        return_proba: Whether to return predicted probabilities

    Returns:
        For ImageNet: (top1_accuracy, top5_accuracy)
        For others: accuracy score
        If return_proba=True: dict with metrics and probabilities

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> clf = LogisticRegression().fit(X, y)
        >>> accuracy = evaluate_classifier(clf, X, y)
    """
    # Special handling for ImageNet
    if dataset in ['imagenet', 'imagenet-mini']:
        y_score = classifier.predict_proba(X)

        top1_acc = accuracy_score(y_true, np.argmax(y_score, axis=1))
        top5_acc = compute_top_k_accuracy(y_true, y_score, k=5)

        if return_proba:
            return {
                'top1_accuracy': top1_acc,
                'top5_accuracy': top5_acc,
                'probabilities': y_score
            }
        return top1_acc, top5_acc

    # Determine if multi-label
    multi_label = len(y_true.shape) > 1 and y_true.shape[1] > 1

    # Get predictions
    y_pred = classifier.predict(X)

    # Compute accuracy
    accuracy = compute_accuracy(y_true, y_pred, multi_label=multi_label)

    if return_proba:
        y_score = classifier.predict_proba(X)
        return {
            'accuracy': accuracy,
            'probabilities': y_score,
            'predictions': y_pred
        }

    return accuracy


def compute_metrics(
    classifier,
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: Optional[str] = None
) -> dict:
    """
    Compute comprehensive metrics for a classifier.

    Args:
        classifier: Trained classifier
        X: Input features
        y_true: True labels
        dataset: Dataset name

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = compute_metrics(classifier, X_test, y_test, 'pathmnist')
        >>> print(metrics['accuracy'], metrics['auc_roc'])
    """
    metrics = {}

    # Special handling for ImageNet (no AUC-ROC)
    if dataset in ['imagenet', 'imagenet-mini']:
        y_score = classifier.predict_proba(X)
        metrics['top1_accuracy'] = accuracy_score(y_true, np.argmax(y_score, axis=1))
        metrics['top5_accuracy'] = compute_top_k_accuracy(y_true, y_score, k=5)
        return metrics

    # Determine task type
    num_classes = len(np.unique(y_true))
    multi_label = len(y_true.shape) > 1 and y_true.shape[1] > 1

    # Predictions
    y_pred = classifier.predict(X)
    y_score = classifier.predict_proba(X)

    # Accuracy
    metrics['accuracy'] = compute_accuracy(y_true, y_pred, multi_label=multi_label)

    # AUC-ROC
    try:
        metrics['auc_roc'] = compute_auc_roc(
            y_true,
            y_score,
            multi_class='ovr',
            multi_label=multi_label
        )
    except Exception as e:
        print(f"Warning: Could not compute AUC-ROC: {e}")
        metrics['auc_roc'] = None

    return metrics
