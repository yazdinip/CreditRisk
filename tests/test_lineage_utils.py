import json

import pandas as pd

from creditrisk.utils.lineage import record_data_lineage


def test_record_data_lineage_writes_snapshot(tmp_path):
    application_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "AMT_INCOME_TOTAL": [1000, 2000],
        }
    )
    feature_store_df = application_df.copy()

    source_path = tmp_path / "application.csv"
    application_df.to_csv(source_path, index=False)

    lineage_path = tmp_path / "reports" / "data_lineage.json"
    record_data_lineage(
        raw_tables={"application": (source_path, application_df)},
        feature_store_df=feature_store_df,
        output_path=lineage_path,
    )

    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert "sources" in payload
    assert "application" in payload["sources"]
    assert payload["sources"]["application"]["rows"] == 2
    assert payload["feature_store"]["rows"] == 2
