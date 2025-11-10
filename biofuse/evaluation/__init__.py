"""
Evaluation modules for BioFuse.

Provides metrics computation, evaluation orchestration, and robustness testing.
"""

from .metrics import (
    compute_accuracy,
    compute_auc_roc,
    compute_top_k_accuracy,
    evaluate_classifier,
    compute_metrics
)
from .evaluator import Evaluator, CrossValidator
from .robustness import RobustnessEvaluator, compute_robustness_metrics

__all__ = [
    # Metrics
    'compute_accuracy',
    'compute_auc_roc',
    'compute_top_k_accuracy',
    'evaluate_classifier',
    'compute_metrics',
    # Evaluator
    'Evaluator',
    'CrossValidator',
    # Robustness
    'RobustnessEvaluator',
    'compute_robustness_metrics',
]
