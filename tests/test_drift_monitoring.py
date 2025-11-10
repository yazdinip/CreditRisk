from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from creditrisk.config import MonitoringConfig
from creditrisk.monitoring.drift import generate_drift_report
from tests.utils import build_test_config


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_generate_drift_report_creates_artifacts(tmp_path):
    monitoring_cfg = MonitoringConfig(
        enabled=True,
        drift_enabled=True,
        reference_sample_size=None,
        current_sample_size=None,
        stat_test="ks",
        stat_test_threshold=0.05,
    )
    config = build_test_config(
        tmp_path,
        selected_columns=["TARGET", "SK_ID_CURR", "FEATURE_A", "FEATURE_B"],
        monitoring_config=monitoring_cfg,
    )

    train_path = tmp_path / "data" / "train.parquet"
    test_path = tmp_path / "data" / "test.parquet"
    config.paths.train_set_path = train_path
    config.paths.test_set_path = test_path

    ref_df = pd.DataFrame(
        {
            "SK_ID_CURR": range(100),
            "TARGET": [0] * 100,
            "FEATURE_A": [i % 5 for i in range(100)],
            "FEATURE_B": [0.1 * i for i in range(100)],
        }
    )
    cur_df = pd.DataFrame(
        {
            "SK_ID_CURR": range(200, 300),
            "TARGET": [0] * 100,
            "FEATURE_A": [2 for _ in range(100)],
            "FEATURE_B": [5 + 0.1 * i for i in range(100)],
        }
    )
    _write_parquet(train_path, ref_df)
    _write_parquet(test_path, cur_df)

    summary = generate_drift_report(config)

    assert config.paths.drift_report.exists()
    assert config.paths.drift_dashboard.exists()
    assert isinstance(summary.share_drifted_columns, float)


def test_generate_drift_report_handles_disabled_monitoring(tmp_path):
    monitoring_cfg = MonitoringConfig(enabled=False, drift_enabled=False)
    config = build_test_config(
        tmp_path,
        selected_columns=["TARGET", "SK_ID_CURR", "FEATURE"],
        monitoring_config=monitoring_cfg,
    )
    report_path = config.paths.drift_report
    html_path = config.paths.drift_dashboard

    summary = generate_drift_report(config)

    assert report_path.exists()
    assert html_path.exists()
    assert summary.dataset_drift is False


def test_generate_drift_report_uses_fallback_backend(tmp_path):
    monitoring_cfg = MonitoringConfig(
        enabled=True,
        drift_enabled=True,
        backend="ks",
        feature_list=["FEATURE_A", "FEATURE_B"],
    )
    config = build_test_config(
        tmp_path,
        selected_columns=["TARGET", "SK_ID_CURR", "FEATURE_A", "FEATURE_B"],
        monitoring_config=monitoring_cfg,
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    config.paths.train_set_path = train_path
    config.paths.test_set_path = test_path

    _write_parquet(
        train_path,
        pd.DataFrame(
            {
                "SK_ID_CURR": range(50),
                "TARGET": [0] * 50,
                "FEATURE_A": list(range(50)),
                "FEATURE_B": [0.1 * i for i in range(50)],
            }
        ),
    )
    _write_parquet(
        test_path,
        pd.DataFrame(
            {
                "SK_ID_CURR": range(100, 150),
                "TARGET": [0] * 50,
                "FEATURE_A": [10] * 50,
                "FEATURE_B": [5 + 0.1 * i for i in range(50)],
            }
        ),
    )

    summary = generate_drift_report(config)

    assert summary.share_drifted_columns > 0
    payload = json.loads(config.paths.drift_report.read_text(encoding="utf-8"))
    assert payload["backend"] == "ks"


def test_constant_columns_are_dropped(tmp_path):
    monitoring_cfg = MonitoringConfig(
        enabled=True,
        drift_enabled=True,
        backend="ks",
    )
    config = build_test_config(
        tmp_path,
        selected_columns=["TARGET", "SK_ID_CURR", "CONST", "FEATURE_X"],
        monitoring_config=monitoring_cfg,
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    config.paths.train_set_path = train_path
    config.paths.test_set_path = test_path

    _write_parquet(
        train_path,
        pd.DataFrame(
            {
                "SK_ID_CURR": range(20),
                "TARGET": [0] * 20,
                "CONST": [1] * 20,
                "FEATURE_X": list(range(20)),
            }
        ),
    )
    _write_parquet(
        test_path,
        pd.DataFrame(
            {
                "SK_ID_CURR": range(100, 120),
                "TARGET": [0] * 20,
                "CONST": [1] * 20,
                "FEATURE_X": [10 + i for i in range(20)],
            }
        ),
    )

    summary = generate_drift_report(config)

    payload = json.loads(config.paths.drift_report.read_text(encoding="utf-8"))
    columns = [row["column"] for row in payload["columns"]]
    assert "CONST" not in columns
    meta = payload.get("meta", {})
    assert "CONST" in meta.get("dropped_constant_columns", [])
    assert summary.share_drifted_columns > 0
