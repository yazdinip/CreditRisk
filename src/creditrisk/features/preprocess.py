"""Feature engineering utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creditrisk.config import FeaturesConfig


def _infer_columns(
    df: pd.DataFrame,
    target_column: str,
    config: FeaturesConfig,
    global_drop: List[str],
) -> Tuple[List[str], List[str]]:
    drop_cols = set(global_drop or []) | set(config.drop_columns or [])
    candidate_df = df.drop(columns=[target_column], errors="ignore")
    candidate_df = candidate_df.drop(columns=list(drop_cols), errors="ignore")

    if config.categorical:
        categorical_cols = config.categorical
    else:
        categorical_cols = candidate_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if config.numerical:
        numerical_cols = config.numerical
    else:
        numerical_cols = candidate_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    # Remove overlap
    categorical_cols = [col for col in categorical_cols if col not in numerical_cols]
    numerical_cols = [col for col in numerical_cols if col not in categorical_cols]
    return numerical_cols, categorical_cols


def build_preprocessor(
    df: pd.DataFrame,
    target_column: str,
    feature_config: FeaturesConfig,
    global_drop: List[str],
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create a ColumnTransformer that handles numerical and categorical columns."""
    numerical_cols, categorical_cols = _infer_columns(df, target_column, feature_config, global_drop)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numerical_cols, categorical_cols
