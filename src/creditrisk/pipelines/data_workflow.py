"""Shared helpers for the modular DVC pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from creditrisk.config import Config
from creditrisk.data.datasets import (
    load_dataset,
    load_optional_dataset,
    train_test_split_df,
)
from creditrisk.features.feature_store import (
    SqlFeatureStoreInputs,
    build_feature_store_via_sql,
)
from creditrisk.features.preprocess import EPS_DEFAULT

LOGGER = logging.getLogger(__name__)


def _load_required_dataset(path: Path | str | None, label: str) -> pd.DataFrame:
    df = load_optional_dataset(path)
    if df is None:
        raise ValueError(f"{label} is required to build the feature store.")
    return df


def build_feature_store_frame(config: Config) -> pd.DataFrame:
    """Run the SQL feature store builder and apply project-specific post-processing."""
    LOGGER.info("Building feature store from %s", config.data.raw_path)
    application_df = load_dataset(
        config.data.raw_path,
        sample_rows=config.data.sample_rows,
    )
    application_df["TOT_MISSING_COUNT"] = application_df.isna().sum(axis=1)

    sql_inputs = SqlFeatureStoreInputs(
        application_df=application_df,
        bureau_df=_load_required_dataset(config.data.bureau_path, "bureau_path"),
        bureau_balance_df=_load_required_dataset(
            config.data.bureau_balance_path,
            "bureau_balance_path",
        ),
        prev_application_df=_load_required_dataset(
            config.data.previous_application_path,
            "previous_application_path",
        ),
        installments_payments_df=_load_required_dataset(
            config.data.installments_payments_path,
            "installments_payments_path",
        ),
        credit_card_balance_df=_load_required_dataset(
            config.data.credit_card_balance_path,
            "credit_card_balance_path",
        ),
        pos_cash_balance_df=_load_required_dataset(
            config.data.pos_cash_balance_path,
            "pos_cash_balance_path",
        ),
    )

    ratio_eps = config.features.ratio_feature_eps or EPS_DEFAULT
    feature_store_df = build_feature_store_via_sql(sql_inputs, eps=ratio_eps)
    feature_store_df = _postprocess_feature_store(feature_store_df, config)
    LOGGER.info(
        "Feature store ready with shape (rows=%d, cols=%d)",
        feature_store_df.shape[0],
        feature_store_df.shape[1],
    )
    return feature_store_df


def _postprocess_feature_store(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    feature_store_df = df.copy()
    if "DAYS_EMPLOYED_ANOMALY" in feature_store_df.columns:
        feature_store_df = feature_store_df.rename(
            columns={"DAYS_EMPLOYED_ANOMALY": "DAYS_EMPLOYED_ANOM"}
        )

    if "DAYS_EMPLOYED" in feature_store_df.columns:
        anomaly_value = config.features.days_employed_anomaly_value
        feature_store_df["DAYS_EMPLOYED_REPLACED"] = feature_store_df["DAYS_EMPLOYED"].replace(
            {anomaly_value: np.nan}
        )

    if "TOT_MISSING_COUNT" in feature_store_df.columns:
        feature_store_df["missing_count"] = feature_store_df["TOT_MISSING_COUNT"]
    else:
        feature_store_df["missing_count"] = feature_store_df.isna().sum(axis=1)

    drop_cols = [
        col for col in config.data.drop_columns if col != config.data.target_column
    ]
    feature_store_df = feature_store_df.drop(columns=drop_cols, errors="ignore")

    feature_store_df = feature_store_df.fillna(0)
    return feature_store_df


def split_feature_store(
    feature_store_df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the engineered features into train and test frames."""
    train_df, test_df = train_test_split_df(
        feature_store_df,
        target_column=config.data.target_column,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=config.data.stratify,
    )
    return train_df, test_df


def save_dataframe(df: pd.DataFrame, path: Path | str) -> Path:
    """Persist a dataframe to parquet, returning the resolved path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info(
        "Saved dataframe with shape (rows=%d, cols=%d) to %s",
        df.shape[0],
        df.shape[1],
        output_path,
    )
    return output_path


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Load a parquet dataframe from disk."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Could not find dataframe at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    LOGGER.info(
        "Loaded dataframe from %s with shape (rows=%d, cols=%d)",
        parquet_path,
        df.shape[0],
        df.shape[1],
    )
    return df
