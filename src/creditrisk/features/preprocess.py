"""Preprocessing utilities that mirror the original notebook workflow."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from creditrisk.config import FeaturesConfig

EPS_DEFAULT = 1e-6


class ColumnSubsetter(BaseEstimator, TransformerMixin):
    """Sklearn transformer that selects a fixed list of columns from a DataFrame."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # type: ignore[override]
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        self._validate_columns(X)
        return X.loc[:, self.columns]

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing = [col for col in self.columns if col not in X.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {', '.join(missing)}"
            )


def _drop_sparse_columns(df: pd.DataFrame, missing_threshold: float) -> pd.DataFrame:
    if missing_threshold <= 0:
        return df
    keep_ratio = max(0.0, 1 - missing_threshold)
    min_non_null = int(len(df) * keep_ratio)
    if min_non_null <= 0:
        return df
    return df.dropna(axis=1, thresh=min_non_null)


def preprocess_application_dataframe(
    df: pd.DataFrame,
    target_column: str,
    feature_config: FeaturesConfig,
    drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply the notebook-style preprocessing steps to the raw application data."""

    processed = df.copy()

    drop_cols = [col for col in (drop_columns or []) if col != target_column]
    if drop_cols:
        processed = processed.drop(columns=drop_cols, errors="ignore")

    eps = feature_config.ratio_feature_eps or EPS_DEFAULT

    if "DAYS_BIRTH" in processed.columns:
        processed["AGE_YEARS"] = -processed["DAYS_BIRTH"] / (365 + eps)

    if (
        feature_config.add_days_employed_anomaly
        and "DAYS_EMPLOYED" in processed.columns
    ):
        anomaly_val = feature_config.days_employed_anomaly_value
        processed["DAYS_EMPLOYED_ANOM"] = (
            processed["DAYS_EMPLOYED"] == anomaly_val
        ).astype(int)
        replaced = processed["DAYS_EMPLOYED"].replace({anomaly_val: np.nan})
        processed["DAYS_EMPLOYED_REPLACED"] = replaced
        processed["EMPLOYED_YEARS"] = -replaced / (365 + eps)
        if "DAYS_BIRTH" in processed.columns:
            processed["EMPLOYMENT_YEARS_TO_AGE"] = (-replaced) / (
                -processed["DAYS_BIRTH"] + eps
            )

    for col in feature_config.missing_indicator_columns:
        if col in processed.columns:
            processed[f"{col}_IS_MISSING"] = processed[col].isna().astype(int)

    if feature_config.add_missing_count:
        missing_count = processed.isna().sum(axis=1)
        processed["missing_count"] = missing_count
        processed["TOT_MISSING_COUNT"] = missing_count

    if feature_config.add_ratio_features:
        _maybe_assign_ratio(processed, "AMT_ANNUITY", "AMT_CREDIT", "PAYMENT_RATE", eps)
        _maybe_assign_ratio(processed, "AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_TO_INCOME", eps)
        _maybe_assign_ratio(processed, "AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_TO_INCOME", eps)
        _maybe_assign_ratio(processed, "AMT_GOODS_PRICE", "AMT_CREDIT", "GOODS_TO_CREDIT", eps)
        _maybe_assign_ratio(processed, "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "INCOME_PER_PERSON", eps)
        _maybe_assign_ratio(processed, "CNT_CHILDREN", "CNT_FAM_MEMBERS", "CHILDREN_RATIO", eps)

    if feature_config.add_count_features:
        _maybe_sum_columns(
            processed,
            [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
            "DOC_COUNT",
        )
        _maybe_sum_columns(
            processed,
            ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"],
            "CONTACT_COUNT",
        )
        _maybe_sum_columns(
            processed,
            [
                "REG_REGION_NOT_LIVE_REGION",
                "REG_REGION_NOT_WORK_REGION",
                "LIVE_REGION_NOT_WORK_REGION",
                "REG_CITY_NOT_LIVE_CITY",
                "REG_CITY_NOT_WORK_CITY",
                "LIVE_CITY_NOT_WORK_CITY",
            ],
            "ADDR_MISMATCH_SUM",
        )

    processed = _drop_sparse_columns(processed, feature_config.missing_threshold)

    drop_feature_cols = [
        col
        for col in feature_config.drop_columns
        if col in processed.columns and col != target_column
    ]
    if drop_feature_cols:
        processed = processed.drop(columns=drop_feature_cols)

    categorical_drop = [
        col
        for col in feature_config.categorical_drop
        if col in processed.columns and col != target_column
    ]
    if categorical_drop:
        processed = processed.drop(columns=categorical_drop)

    categorical_cols = [
        col
        for col in processed.select_dtypes(include=["object"]).columns
        if col != target_column
    ]
    if categorical_cols:
        processed = pd.get_dummies(processed, columns=categorical_cols, drop_first=False)

    processed = processed.fillna(0)

    return processed


def _maybe_assign_ratio(df: pd.DataFrame, numerator: str, denominator: str, out_col: str, eps: float) -> None:
    if numerator in df.columns and denominator in df.columns:
        df[out_col] = df[numerator] / (df[denominator] + eps)


def _maybe_sum_columns(df: pd.DataFrame, columns: List[str], out_col: str) -> None:
    existing = [col for col in columns if col in df.columns]
    if existing:
        df[out_col] = df[existing].fillna(0).sum(axis=1)


def resolve_selected_columns(
    df: pd.DataFrame,
    target_column: str,
    feature_config: FeaturesConfig,
) -> List[str]:
    """Return the ordered list of feature columns to feed the scaler/model."""

    if feature_config.selected_columns:
        missing = [
            col for col in feature_config.selected_columns if col not in df.columns
        ]
        if missing:
            raise ValueError(
                "Configured selected_columns are missing after preprocessing: "
                + ", ".join(missing)
            )
        return feature_config.selected_columns

    return [col for col in df.columns if col != target_column]
