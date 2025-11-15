"""Governance utilities for inference-time auditing."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from creditrisk.config import InferenceConfig, TrackingConfig

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore


def _band_from_probability(prob: float) -> str:
    if prob >= 0.7:
        return "high_risk"
    if prob >= 0.4:
        return "medium_risk"
    return "low_risk"


@dataclass
class InferenceAuditLogger:
    """Publishes inference-time metrics/tags to MLflow for governance."""

    tracking_cfg: TrackingConfig
    inference_cfg: InferenceConfig

    def __post_init__(self) -> None:
        self.enabled = bool(self.tracking_cfg.enabled and mlflow is not None)
        self.tracking_uri = self.tracking_cfg.tracking_uri
        self.experiment = (
            self.inference_cfg.mlflow_inference_experiment
            or f"{self.tracking_cfg.experiment_name}_inference"
        )
        self.base_tags = {
            "service": "creditrisk-inference",
            **self.tracking_cfg.tags,
            **self.inference_cfg.governance_tags,
        }
        self.deployment_stage = os.getenv("DEPLOYMENT_STAGE", "production")

    def log_request(
        self,
        *,
        request_id: str,
        threshold: float,
        payload_df: pd.DataFrame,
        result_df: pd.DataFrame,
        probabilities: Optional[np.ndarray],
        entity_column: Optional[str],
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        if not self.enabled:
            return

        assert mlflow is not None  # for type checkers
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

        entity_count = len(result_df)
        positive = int(result_df["prediction"].sum()) if "prediction" in result_df else 0
        high_risk_rate = positive / entity_count if entity_count else 0.0
        avg_probability = float(np.mean(probabilities)) if probabilities is not None else None

        band_counts: Dict[str, int] = {"high_risk": 0, "medium_risk": 0, "low_risk": 0}
        if probabilities is not None:
            for prob in probabilities:
                band_counts[_band_from_probability(float(prob))] += 1
        else:
            for decision in result_df.get("prediction", []):
                band = "high_risk" if decision == 1 else "low_risk"
                band_counts[band] += 1

        unique_entities = (
            payload_df[entity_column].nunique() if entity_column and entity_column in payload_df.columns else entity_count
        )

        dynamic_tags = {
            "request_id": request_id,
            "deployment_stage": self.deployment_stage,
            "decision_threshold": str(threshold),
            "entity_count": str(entity_count),
            **(extra_tags or {}),
        }

        with mlflow.start_run(run_name=f"inference-{request_id}", nested=True):
            mlflow.set_tags({**self.base_tags, **dynamic_tags})
            metrics = {
                "entity_count": entity_count,
                "alert_rate": high_risk_rate,
                "positive_decisions": positive,
                "unique_entities": unique_entities,
                "band_high_risk": band_counts["high_risk"],
                "band_medium_risk": band_counts["medium_risk"],
                "band_low_risk": band_counts["low_risk"],
            }
            if avg_probability is not None:
                metrics["probability_mean"] = avg_probability
            mlflow.log_metrics(metrics)
            if "probability" in result_df:
                snapshot = result_df.head(50).to_dict(orient="list")
                mlflow.log_dict(snapshot, artifact_file="sample_predictions.json")
