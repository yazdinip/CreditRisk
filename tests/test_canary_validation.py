from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from creditrisk.models.creditrisk_pd import build_training_pipeline
from creditrisk.pipelines.canary_validation import run_canary_validation
from tests.utils import build_test_config


def _train_pipeline(config, df: pd.DataFrame):
    pipeline = build_training_pipeline(
        selected_columns=config.features.selected_columns,
        model_cfg=config.model,
        training_cfg=config.training,
    )
    pipeline.fit(df[config.features.selected_columns], df["TARGET"])
    return pipeline


def test_canary_validation_passes_with_identical_models(tmp_path):
    cfg = build_test_config(tmp_path, ["feature_1", "feature_2"])
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100, 200, 300, 400],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [0.4, 0.3, 0.2, 0.1],
            "TARGET": [0, 1, 0, 1],
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    df.to_parquet(dataset_path, index=False)

    pipeline = _train_pipeline(cfg, df)
    production_path = tmp_path / "production.joblib"
    joblib.dump(pipeline, production_path)
    candidate_path = tmp_path / "candidate.joblib"
    joblib.dump(pipeline, candidate_path)

    report_path = tmp_path / "report.json"
    summary = run_canary_validation(
        cfg,
        production_model_path=production_path,
        candidate_model_path=candidate_path,
        dataset_path=dataset_path,
        max_metric_delta=0.01,
        output_path=report_path,
    )
    assert summary["status"] == "passed"
    assert report_path.exists()


def test_canary_validation_fails_when_metrics_diverge(tmp_path):
    cfg = build_test_config(tmp_path, ["feature_1", "feature_2"])
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100, 200, 300, 400],
            "feature_1": [0.1, 0.2, 0.9, 0.8],
            "feature_2": [0.5, 0.6, 0.1, 0.2],
            "TARGET": [0, 0, 1, 1],
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    df.to_parquet(dataset_path, index=False)

    production_model = _train_pipeline(cfg, df)
    production_path = tmp_path / "production.joblib"
    joblib.dump(production_model, production_path)

    candidate = _train_pipeline(cfg, df)
    # Force the candidate to predict 1s with high probability.
    candidate.named_steps["model"].coef_ = np.array([[5.0, 5.0]])
    candidate.named_steps["model"].intercept_ = np.array([5.0])
    candidate_path = tmp_path / "candidate.joblib"
    joblib.dump(candidate, candidate_path)

    with pytest.raises(RuntimeError):
        run_canary_validation(
            cfg,
            production_model_path=production_path,
            candidate_model_path=candidate_path,
            dataset_path=dataset_path,
            max_metric_delta=0.01,
            output_path=tmp_path / "report.json",
        )
