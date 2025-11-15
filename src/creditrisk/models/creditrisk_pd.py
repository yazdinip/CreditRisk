"""Model helpers for the CreditRisk PD pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - optional dependency
    SMOTE = None  # type: ignore

from creditrisk.config import ModelConfig, TrainingConfig
from creditrisk.features.preprocess import ColumnSubsetter


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


def build_training_pipeline(
    selected_columns: Iterable[str],
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
) -> Pipeline:
    """Create the sklearn pipeline (column subset -> scaler -> classifier)."""
    classifier = build_classifier(model_cfg, training_cfg)
    return Pipeline(
        steps=[
            ("select", ColumnSubsetter(selected_columns)),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def rebalance_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    training_cfg: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE and/or downsampling the way the original notebook did."""
    X_bal = X.copy()
    y_bal = y.copy()
    y_name = y_bal.name or "target"

    if training_cfg.use_smote:
        if SMOTE is None:
            raise ImportError(
                "imbalanced-learn is required for SMOTE. Install it via `pip install imbalanced-learn`."
            )
        sampler = SMOTE(
            random_state=training_cfg.random_state,
            sampling_strategy=training_cfg.smote_sampling_strategy,
        )
        X_res, y_res = sampler.fit_resample(X_bal, y_bal)
        X_bal = pd.DataFrame(X_res, columns=X_bal.columns)
        y_bal = pd.Series(y_res, name=y_name)

    if training_cfg.downsample_majority:
        X_bal, y_bal = _downsample_majority_class(
            X_bal, y_bal, training_cfg.random_state
        )

    return X_bal, y_bal


def _downsample_majority_class(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    combined = X.copy()
    target_name = y.name or "target"
    combined[target_name] = y.values

    class_counts = combined[target_name].value_counts()
    min_count = class_counts.min()

    balanced_frames = []
    for label, _ in class_counts.items():
        subset = combined[combined[target_name] == label]
        if len(subset) > min_count:
            subset = subset.sample(n=min_count, random_state=random_state)
        balanced_frames.append(subset)

    balanced = (
        pd.concat(balanced_frames)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    X_bal = balanced.drop(columns=target_name)
    y_bal = balanced[target_name]
    return X_bal, y_bal


def _normalized_gini_from_auc(roc_auc: float | None) -> float | None:
    """Return the normalized Gini coefficient given an ROC-AUC value."""
    if roc_auc is None:
        return None
    return max(min(2 * roc_auc - 1, 1.0), -1.0)


def evaluate_classifier(estimator, X_test, y_test) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    y_pred = estimator.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    roc_auc = None
    if hasattr(estimator, "predict_proba"):
        y_prob = estimator.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, scores)

    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
        normalized_gini = _normalized_gini_from_auc(roc_auc)
        if normalized_gini is not None:
            metrics["normalized_gini"] = normalized_gini

    return metrics


def save_model(estimator, path: Path) -> None:
    """Persist the trained estimator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
