from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from creditrisk.config import TestingConfig
from creditrisk.testing.post_training import PostTrainingValidator
from .utils import build_test_config


def _log_mlflow_run(config, metrics):
    tracking_uri = str(config.tracking.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.create_experiment(config.tracking.experiment_name)
    except mlflow.exceptions.MlflowException:
        pass
    mlflow.set_experiment(config.tracking.experiment_name)
    with mlflow.start_run(run_name=config.tracking.run_name):
        mlflow.log_metrics(metrics)
        if config.tracking.tags:
            mlflow.set_tags(config.tracking.tags)


def _write_lineage_file(config, payload):
    lineage_path = Path(config.paths.lineage_file)
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return lineage_path


def test_post_training_validator_passes_with_valid_artifacts(tmp_path):
    testing_cfg = TestingConfig(
        enabled=True,
        min_metrics={"roc_auc": 0.7, "normalized_gini": 0.4, "precision": 0.2},
        evaluation_artifacts=["roc_curve.png", "confusion_matrix.json"],
        require_mlflow=True,
        mlflow_metric_tolerance=1e-6,
    )
    config = build_test_config(
        tmp_path,
        ["feature_1"],
        enable_tracking=True,
        testing_config=testing_cfg,
    )
    config.tracking.tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    config.tracking.experiment_name = "post_training_test"
    config.paths.metrics_file = tmp_path / "reports" / "metrics.json"
    metrics = {"roc_auc": 0.8, "normalized_gini": 0.6, "precision": 0.3}
    config.paths.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.paths.metrics_file, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp)

    evaluation_dir = config.paths.evaluation_dir
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for artifact in config.testing.evaluation_artifacts:
        (evaluation_dir / artifact).write_text("ok", encoding="utf-8")

    _write_lineage_file(
        config,
        {
            "generated_at": "2025-01-01T00:00:00Z",
            "sources": {"application": {"rows": 4, "columns": 3}},
            "feature_store": {"rows": 4, "columns": 3},
        },
    )

    _log_mlflow_run(config, metrics)

    validator = PostTrainingValidator(config, evaluation_dir=evaluation_dir)
    summary = validator.run()

    assert summary["status"] == "passed"
    assert summary["mlflow_run_id"] is not None


def test_post_training_validator_raises_when_metrics_below_threshold(tmp_path):
    testing_cfg = TestingConfig(enabled=True, min_metrics={"roc_auc": 0.95}, require_mlflow=False)
    config = build_test_config(
        tmp_path,
        ["feature_1"],
        enable_tracking=False,
        testing_config=testing_cfg,
    )
    config.paths.metrics_file = tmp_path / "reports" / "metrics.json"
    config.paths.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.paths.metrics_file, "w", encoding="utf-8") as fp:
        json.dump({"roc_auc": 0.7}, fp)

    evaluation_dir = config.paths.evaluation_dir
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for artifact in config.testing.evaluation_artifacts:
        (evaluation_dir / artifact).write_text("ok", encoding="utf-8")

    _write_lineage_file(
        config,
        {
            "generated_at": "2025-01-01T00:00:00Z",
            "sources": {"application": {"rows": 4, "columns": 3}},
            "feature_store": {"rows": 4, "columns": 3},
        },
    )

    validator = PostTrainingValidator(config, evaluation_dir=config.paths.evaluation_dir)
    with pytest.raises(RuntimeError):
        validator.run()


def test_post_training_validator_fails_without_lineage(tmp_path):
    testing_cfg = TestingConfig(
        enabled=True,
        min_metrics={"roc_auc": 0.7},
        evaluation_artifacts=["roc_curve.png"],
        require_mlflow=False,
    )
    config = build_test_config(
        tmp_path,
        ["feature_1"],
        enable_tracking=False,
        testing_config=testing_cfg,
    )
    config.paths.metrics_file = tmp_path / "reports" / "metrics.json"
    config.paths.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.paths.metrics_file, "w", encoding="utf-8") as fp:
        json.dump({"roc_auc": 0.8}, fp)

    evaluation_dir = config.paths.evaluation_dir
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for artifact in config.testing.evaluation_artifacts:
        (evaluation_dir / artifact).write_text("ok", encoding="utf-8")

    validator = PostTrainingValidator(config, evaluation_dir=evaluation_dir)
    with pytest.raises(RuntimeError):
        validator.run()
