"""Model helpers for the baseline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

from creditrisk.config import ModelConfig, TrainingConfig


def build_classifier(model_cfg: ModelConfig, training_cfg: TrainingConfig):
    """Return an instantiated sklearn-compatible classifier."""
    model_type = model_cfg.type.lower()
    params = {**model_cfg.params}

    if model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Please `pip install xgboost` or switch model.type."
            )
        params.setdefault("n_jobs", training_cfg.n_jobs)
        params.setdefault("random_state", training_cfg.random_state)
        params.setdefault("tree_method", "hist" if not training_cfg.use_gpu else "gpu_hist")
        if training_cfg.use_gpu:
            params.setdefault("predictor", "gpu_predictor")
        return XGBClassifier(**params)

    if model_type == "logistic_regression":
        params.setdefault("max_iter", 500)
        params.setdefault("n_jobs", training_cfg.n_jobs)
        return LogisticRegression(class_weight=training_cfg.class_weight, **params)

    if model_type == "random_forest":
        params.setdefault("n_estimators", 300)
        params.setdefault("n_jobs", training_cfg.n_jobs)
        return RandomForestClassifier(
            class_weight=training_cfg.class_weight,
            random_state=training_cfg.random_state,
            **params,
        )

    raise ValueError(f"Unsupported model type '{model_cfg.type}'.")


def evaluate_classifier(estimator, X_test, y_test) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    y_pred = estimator.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(estimator, "predict_proba"):
        y_prob = estimator.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        metrics["roc_auc"] = roc_auc_score(y_test, scores)

    return metrics


def save_model(estimator, path: Path) -> None:
    """Persist the trained estimator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
