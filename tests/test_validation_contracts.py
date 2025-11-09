import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from creditrisk.validation import contracts


def test_raw_schemas_cover_expected_tables():
    expected = {
        "application",
        "bureau",
        "bureau_balance",
        "previous_application",
        "installments",
        "credit_card",
        "pos_cash",
    }
    assert set(contracts.RAW_SCHEMAS.keys()) == expected


def test_feature_store_schema_requires_missing_count():
    schema = contracts.build_feature_store_schema("TARGET")
    valid = pd.DataFrame(
        {
            "SK_ID_CURR": [1.0],
            "TARGET": [0.0],
            "missing_count": [10.0],
        }
    )
    schema.validate(valid)

    missing_col = pd.DataFrame({"SK_ID_CURR": [1.0], "TARGET": [0.0]})
    with pytest.raises((SchemaErrors, SchemaError)):
        schema.validate(missing_col)


def test_application_schema_blocks_invalid_days_employed():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1.0],
            "TARGET": [0.0],
            "AMT_INCOME_TOTAL": [1000.0],
            "AMT_CREDIT": [500.0],
            "CNT_CHILDREN": [0.0],
            "DAYS_BIRTH": [-10000.0],
            "DAYS_EMPLOYED": [100.0],  # invalid (positive)
            "EXT_SOURCE_1": [0.1],
            "EXT_SOURCE_2": [0.2],
            "EXT_SOURCE_3": [0.3],
        }
    )
    with pytest.raises((SchemaErrors, SchemaError)):
        contracts.APPLICATION_SCHEMA.validate(df)
