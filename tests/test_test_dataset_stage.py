from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from creditrisk.models.creditrisk_pd import build_training_pipeline
from creditrisk.testing.test_dataset import run_test_suite
from .utils import build_test_config


def test_run_test_suite_creates_metrics_predictions_and_plots(tmp_path):
    selected_columns = ["feature_1", "feature_2"]
    config = build_test_config(tmp_path, selected_columns)

    train_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
            "feature_1": np.linspace(0.1, 0.6, 6),
            "feature_2": np.linspace(0.6, 0.1, 6),
            "TARGET": [0, 0, 0, 1, 1, 1],
        }
    )
    pipeline = build_training_pipeline(
        selected_columns=selected_columns,
        model_cfg=config.model,
        training_cfg=config.training,
    )
    pipeline.fit(train_df[selected_columns], train_df["TARGET"])
    config.paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.paths.model_path)

    test_df = pd.DataFrame(
        {
            "SK_ID_CURR": [101, 102, 103],
            "feature_1": [0.15, 0.45, 0.55],
            "feature_2": [0.55, 0.25, 0.35],
            "TARGET": [0, 0, 1],
        }
    )
    test_path = tmp_path / "test.parquet"
    test_df.to_parquet(test_path, index=False)
    config.paths.test_set_path = test_path

    metrics_path = tmp_path / "reports" / "test_metrics.json"
    predictions_path = tmp_path / "reports" / "test_predictions.parquet"
    evaluation_dir = tmp_path / "reports" / "test_eval"

    metrics = run_test_suite(
        config,
        model_path=config.paths.model_path,
        split_path=test_path,
        split_name="test",
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        evaluation_dir=evaluation_dir,
    )

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert (evaluation_dir / "confusion_matrix.png").exists()
    assert "accuracy" in metrics
