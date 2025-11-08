"""Validation runner that executes Pandera contracts and lightweight checks."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import pandera as pa
from pandera import errors as pa_errors

from creditrisk.config import ValidationConfig
from creditrisk.validation import contracts

LOGGER = logging.getLogger(__name__)


class ValidationRunner:
    """Centralized helper to run data contracts throughout the pipeline."""

    def __init__(self, cfg: Optional[ValidationConfig] = None):
        self.cfg = cfg or ValidationConfig()

    # --------------------------------------------------------------------- #
    # Raw layer validators
    # --------------------------------------------------------------------- #

    def validate_raw_table(self, table_name: str, df: pd.DataFrame) -> None:
        if not self._should_run(self.cfg.enforce_raw_contracts):
            return
        schema = contracts.RAW_SCHEMAS.get(table_name)
        if schema is None:
            raise ValueError(f"No raw schema registered for table '{table_name}'.")
        self._run_schema(schema, df, f"{table_name}_raw")

    def validate_application(self, df: pd.DataFrame) -> None:
        self.validate_raw_table("application", df)

    def validate_bureau(self, df: pd.DataFrame) -> None:
        self.validate_raw_table("bureau", df)

    def validate_bureau_balance(self, df: pd.DataFrame) -> None:
        self.validate_raw_table("bureau_balance", df)

    def validate_previous_applications(self, df: pd.DataFrame) -> None:
        self.validate_raw_table("previous_application", df)

    def validate_installments(self, df: pd.DataFrame) -> None:
        self.validate_raw_table("installments", df)

    def validate_credit_card(self, df: pd.DataFrame) -> None:
        self.validate_raw_table("credit_card", df)

    def validate_pos_cash(self, df: pd.DataFrame) -> None:
        self.validate_raw_table("pos_cash", df)

    # --------------------------------------------------------------------- #
    # Feature-store / split validators
    # --------------------------------------------------------------------- #

    def validate_feature_store(
        self,
        df: pd.DataFrame,
        target_column: str,
        required_feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        if not self._should_run(self.cfg.enforce_feature_store_contract):
            return
        schema = contracts.build_feature_store_schema(target_column)
        self._run_schema(schema, df, "feature_store")
        if required_feature_columns:
            missing = sorted(set(required_feature_columns) - set(df.columns))
            if missing:
                raise ValueError(
                    f"Feature store is missing expected engineered features: {', '.join(missing[:10])}"
                )

    def validate_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        entity_column: str,
        split_name: str,
    ) -> None:
        if not self._should_run(self.cfg.enforce_split_contracts):
            return
        schema = contracts.build_split_schema(target_column)
        self._run_schema(schema, df, f"{split_name}_split")
        duplicates = df[entity_column][df[entity_column].duplicated()]
        if not duplicates.empty:
            raise ValueError(
                f"Duplicate {entity_column} values detected inside the {split_name} split: "
                f"{duplicates.head().tolist()}"
            )

    def validate_split_pair(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        entity_column: str,
    ) -> None:
        if not self._should_run(self.cfg.enforce_split_contracts):
            return
        overlap = np.intersect1d(train_df[entity_column].values, test_df[entity_column].values)
        if overlap.size > 0:
            raise ValueError(
                f"Entity values overlap between train/test splits ({overlap[:5].tolist()} ...)."
            )

    # --------------------------------------------------------------------- #
    # Model IO validators
    # --------------------------------------------------------------------- #

    def validate_feature_matrix(
        self,
        feature_df: pd.DataFrame,
        expected_columns: Sequence[str],
        stage_name: str,
    ) -> None:
        if not self._should_run(self.cfg.enforce_model_io_contracts):
            return
        missing = sorted(set(expected_columns) - set(feature_df.columns))
        if missing:
            raise ValueError(
                f"{stage_name} matrix is missing columns required by the model: {', '.join(missing[:10])}"
            )
        view = feature_df[expected_columns]
        if view.isna().any().any():
            raise ValueError(f"{stage_name} matrix contains NaNs after preprocessing.")
        if np.isinf(view.to_numpy()).any():
            raise ValueError(f"{stage_name} matrix contains +/-inf values.")

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _should_run(self, flag: bool) -> bool:
        return bool(self.cfg.enabled and flag)

    def _run_schema(self, schema: pa.DataFrameSchema, df: pd.DataFrame, label: str) -> None:
        try:
            schema.validate(df, lazy=True)
            LOGGER.debug("Validation for %s passed (rows=%d, cols=%d).", label, len(df), len(df.columns))
        except pa_errors.SchemaErrors as exc:
            sample = exc.failure_cases.head(10)
            raise ValueError(
                f"Data contract validation failed for '{label}'. Sample failures:\n{sample}"
            ) from exc
