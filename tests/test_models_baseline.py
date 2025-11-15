import pandas as pd
import pytest

from creditrisk.models.creditrisk_pd import build_training_pipeline, evaluate_classifier
from .utils import build_test_config


def test_evaluate_classifier_reports_normalized_gini(tmp_path):
    selected_columns = ["feature_1", "feature_2"]
    config = build_test_config(tmp_path, selected_columns)

    train_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "feature_1": [0.1, 0.8, 0.2, 0.7],
            "feature_2": [1.0, 0.4, 0.3, 0.9],
            "TARGET": [0, 1, 0, 1],
        }
    )

    pipeline = build_training_pipeline(
        selected_columns=selected_columns,
        model_cfg=config.model,
        training_cfg=config.training,
    )
    pipeline.fit(train_df[selected_columns], train_df["TARGET"])

    metrics = evaluate_classifier(pipeline, train_df[selected_columns], train_df["TARGET"])

    assert "roc_auc" in metrics
    assert "normalized_gini" in metrics
    assert metrics["normalized_gini"] == pytest.approx(2 * metrics["roc_auc"] - 1, rel=1e-6)
