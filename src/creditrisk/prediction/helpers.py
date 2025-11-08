"""Shared helpers for turning model outputs into consumable artifacts."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def predict_with_threshold(
    pipeline,
    features: pd.DataFrame,
    threshold: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return binary decisions and optional probabilities for a dataframe."""
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(features)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs

    LOGGER.warning(
        "Estimator does not expose predict_proba; using estimator.predict output exclusively.",
    )
    preds = pipeline.predict(features)
    return preds, None


def build_output_frame(
    raw_df: pd.DataFrame,
    decisions: np.ndarray,
    probabilities: Optional[np.ndarray],
    entity_column: Optional[str],
) -> pd.DataFrame:
    """Compose a dataframe with identifiers + decision outputs."""
    output = pd.DataFrame()
    if entity_column and entity_column in raw_df.columns:
        output[entity_column] = raw_df[entity_column].values
    else:
        output["row_id"] = np.arange(len(raw_df))
    output["prediction"] = decisions.astype(int)
    if probabilities is not None:
        output["probability"] = probabilities
    return output
