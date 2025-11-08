"""Data ingestion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: Path | str, sample_rows: Optional[int] = None) -> pd.DataFrame:
    """Load a CSV dataset with optional sampling."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate dataset at {csv_path}")

    df = pd.read_csv(csv_path)
    if sample_rows:
        df = df.sample(n=sample_rows, random_state=42)
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features (X) and target (y)."""
    drop_columns = drop_columns or []
    feature_df = df.drop(columns=[target_column] + drop_columns, errors="ignore")
    target = df[target_column]
    return feature_df, target


def train_test_split_df(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper around sklearn train/test split with optional stratification."""
    stratify_values = df[target_column] if stratify else None
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
