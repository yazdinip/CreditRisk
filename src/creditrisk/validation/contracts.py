"""Pandera-based data contracts for the CreditRisk pipeline."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import pandera as pa
from pandera import Check, Column

ID_COLUMN = "SK_ID_CURR"

UNIQUE_SERIES_CHECK = Check(
    lambda s: s.is_unique,
    element_wise=False,
    error="Column must contain unique values.",
)
NON_NEGATIVE = Check.ge(0)
NON_POSITIVE = Check.le(0)


def _allow_special_days_employed(series: pd.Series) -> bool:
    """Allow special sentinel value (365243) used in the Kaggle dataset."""
    mask = (series <= 0) | (series == 365243) | series.isna()
    return bool(mask.all())


APPLICATION_SCHEMA = pa.DataFrameSchema(
    {
        ID_COLUMN: Column(
            pa.Float64,
            nullable=False,
            checks=[NON_NEGATIVE, UNIQUE_SERIES_CHECK],
        ),
        "TARGET": Column(
            pa.Float64,
            nullable=False,
            checks=[Check.isin([0, 1])],
        ),
        "AMT_INCOME_TOTAL": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "AMT_CREDIT": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "CNT_CHILDREN": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
        "DAYS_BIRTH": Column(pa.Float64, nullable=False, checks=[NON_POSITIVE]),
        "DAYS_EMPLOYED": Column(
            pa.Float64,
            nullable=True,
            checks=[Check(_allow_special_days_employed, element_wise=False)],
        ),
        "EXT_SOURCE_1": Column(pa.Float64, nullable=True),
        "EXT_SOURCE_2": Column(pa.Float64, nullable=True),
        "EXT_SOURCE_3": Column(pa.Float64, nullable=True),
    },
    strict=False,
    coerce=True,
)

BUREAU_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_BUREAU": Column(
            pa.Float64,
            nullable=False,
            checks=[NON_NEGATIVE, UNIQUE_SERIES_CHECK],
        ),
        ID_COLUMN: Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "AMT_CREDIT_SUM": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
        "CREDIT_DAY_OVERDUE": Column(pa.Float64, nullable=True),
        "CREDIT_ACTIVE": Column(pa.String, nullable=True),
    },
    strict=False,
    coerce=True,
)

BUREAU_BALANCE_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_BUREAU": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "MONTHS_BALANCE": Column(pa.Float64, nullable=False),
        "STATUS": Column(pa.String, nullable=False),
    },
    strict=False,
    coerce=True,
)

PREVIOUS_APPLICATION_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_PREV": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        ID_COLUMN: Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "AMT_CREDIT": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
        "AMT_ANNUITY": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
        "NAME_CONTRACT_STATUS": Column(pa.String, nullable=False),
    },
    strict=False,
    coerce=True,
)

INSTALLMENTS_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_PREV": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        ID_COLUMN: Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "NUM_INSTALMENT_NUMBER": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "NUM_INSTALMENT_VERSION": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
        "AMT_INSTALMENT": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
        "AMT_PAYMENT": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
    },
    strict=False,
    coerce=True,
)

CREDIT_CARD_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_PREV": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        ID_COLUMN: Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "MONTHS_BALANCE": Column(pa.Float64, nullable=False),
        "AMT_BALANCE": Column(pa.Float64, nullable=True),
        "AMT_CREDIT_LIMIT_ACTUAL": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
    },
    strict=False,
    coerce=True,
)

POS_CASH_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_PREV": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        ID_COLUMN: Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        "MONTHS_BALANCE": Column(pa.Float64, nullable=False),
        "CNT_INSTALMENT": Column(pa.Float64, nullable=True, checks=[NON_NEGATIVE]),
        "CNT_INSTALMENT_FUTURE": Column(pa.Float64, nullable=True),
    },
    strict=False,
    coerce=True,
)

RAW_SCHEMAS: Dict[str, pa.DataFrameSchema] = {
    "application": APPLICATION_SCHEMA,
    "bureau": BUREAU_SCHEMA,
    "bureau_balance": BUREAU_BALANCE_SCHEMA,
    "previous_application": PREVIOUS_APPLICATION_SCHEMA,
    "installments": INSTALLMENTS_SCHEMA,
    "credit_card": CREDIT_CARD_SCHEMA,
    "pos_cash": POS_CASH_SCHEMA,
}


def build_feature_store_schema(target_column: str) -> pa.DataFrameSchema:
    """Return a schema that validates the engineered feature store."""
    return pa.DataFrameSchema(
        {
            ID_COLUMN: Column(
                pa.Float64,
                nullable=False,
                checks=[NON_NEGATIVE, UNIQUE_SERIES_CHECK],
            ),
            target_column: Column(
                pa.Float64,
                nullable=False,
                checks=[Check.isin([0, 1])],
            ),
            "missing_count": Column(pa.Float64, nullable=False, checks=[NON_NEGATIVE]),
        },
        strict=False,
        coerce=True,
    )


def build_split_schema(target_column: str) -> pa.DataFrameSchema:
    """Basic schema for persisted train/test splits."""
    return pa.DataFrameSchema(
        {
            ID_COLUMN: Column(
                pa.Float64,
                nullable=False,
                checks=[NON_NEGATIVE],
            ),
            target_column: Column(
                pa.Float64,
                nullable=False,
                checks=[Check.isin([0, 1])],
            ),
        },
        strict=False,
        coerce=True,
    )
