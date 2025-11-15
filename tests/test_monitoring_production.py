from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditrisk.monitoring.production import run_production_monitor
from tests.utils import build_test_config


def _prep_config(tmp_path: Path) -> tuple:
    cfg = build_test_config(tmp_path, ["feature_1", "feature_2"])
    cfg.monitoring.enabled = True
    cfg.monitoring.drift_enabled = True
    cfg.monitoring.production_reference_path = tmp_path / "reference.parquet"
    cfg.monitoring.production_current_path = tmp_path / "current.parquet"
    cfg.monitoring.retrain_drift_threshold = 0.5
    cfg.monitoring.retrain_command = "echo retrain"
    cfg.paths.production_drift_report = tmp_path / "prod_drift.json"
    cfg.paths.production_drift_dashboard = tmp_path / "prod_drift.html"
    cfg.paths.drift_metrics_file = tmp_path / "metrics.json"
    cfg.paths.retrain_report = tmp_path / "retrain.json"
    return cfg, cfg.monitoring.production_reference_path, cfg.monitoring.production_current_path


def test_production_monitor_writes_metrics(tmp_path):
    cfg, reference_path, current_path = _prep_config(tmp_path)
    ref_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [0.9, 0.8, 0.7, 0.6],
            "TARGET": [0, 0, 1, 0],
        }
    )
    cur_df = pd.DataFrame(
        {
            "SK_ID_CURR": [5, 6, 7, 8],
            "feature_1": [0.9, 0.95, 0.97, 0.92],
            "feature_2": [0.1, 0.2, 0.3, 0.1],
            "TARGET": [0, 0, 0, 0],
        }
    )
    ref_df.to_parquet(reference_path)
    cur_df.to_parquet(current_path)

    result = run_production_monitor(cfg, retrain_handler=lambda cmd: None)
    assert result.metrics_path.exists()
    payload = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "summary" in payload
    assert not result.retrain_triggered


def test_production_monitor_can_force_retrain(tmp_path):
    cfg, reference_path, current_path = _prep_config(tmp_path)
    ref_df = pd.DataFrame(
        {"SK_ID_CURR": [1, 2], "feature_1": [0.1, 0.2], "feature_2": [0.2, 0.3], "TARGET": [0, 1]}
    )
    cur_df = pd.DataFrame(
        {"SK_ID_CURR": [3, 4], "feature_1": [0.9, 0.95], "feature_2": [0.8, 0.85], "TARGET": [0, 0]}
    )
    ref_df.to_parquet(reference_path)
    cur_df.to_parquet(current_path)

    triggered_commands = []

    def _test_handler(command: str) -> None:
        triggered_commands.append(command)

    result = run_production_monitor(
        cfg,
        force_retrain=True,
        retrain_handler=_test_handler,
    )
    assert result.retrain_triggered
    assert triggered_commands == [cfg.monitoring.retrain_command]
