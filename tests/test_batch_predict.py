import joblib
import numpy as np
import pandas as pd

from creditrisk.models.baseline import build_training_pipeline
from creditrisk.pipelines.batch_predict import run_batch_inference
from .utils import build_test_config


def test_run_batch_inference_writes_predictions(tmp_path):
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

    config.paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.paths.model_path)

    inference_df = train_df.drop(columns=["TARGET"])
    input_csv = tmp_path / "inference.csv"
    inference_df.to_csv(input_csv, index=False)
    output_csv = tmp_path / "predictions.csv"

    results = run_batch_inference(
        config=config,
        input_csv=input_csv,
        output_csv=output_csv,
    )

    assert output_csv.exists()
    assert len(results) == len(inference_df)
    assert "prediction" in results.columns
    assert "probability" in results.columns
    np.testing.assert_array_equal(
        results["prediction"].values,
        (results["probability"].values >= config.inference.decision_threshold).astype(int),
    )
