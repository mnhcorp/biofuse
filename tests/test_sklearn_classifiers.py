import importlib
import sys
import types

import numpy as np


def test_logistic_regression_omits_multi_class_when_backend_signature_does_not_support_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBClassifier=object))
    monkeypatch.setitem(sys.modules, "catboost", types.SimpleNamespace(CatBoostClassifier=object))

    sklearn_module = importlib.import_module("biofuse.classifiers.sklearn_classifiers")

    created = {}

    class FakeLogisticRegression:
        def __init__(self, C=1.0, max_iter=1000, random_state=42, solver="lbfgs"):
            created["C"] = C
            created["max_iter"] = max_iter
            created["random_state"] = random_state
            created["solver"] = solver

        def fit(self, X, y):
            created["fit_shape"] = X.shape
            created["labels_shape"] = y.shape
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([1.0, 0.0], (len(X), 1))

    monkeypatch.setattr(sklearn_module, "SklearnLogisticRegression", FakeLogisticRegression)

    classifier = sklearn_module.LogisticRegression(
        max_iter=123,
        multi_class="ovr",
        solver="liblinear",
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([0, 1, 1, 0])

    classifier.fit(X, y)

    assert created["max_iter"] == 123
    assert created["solver"] == "liblinear"
    assert created["fit_shape"] == (4, 2)
    assert classifier.is_fitted is True
