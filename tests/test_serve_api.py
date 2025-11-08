import joblib
import pandas as pd
from fastapi.testclient import TestClient

from creditrisk.models.baseline import build_training_pipeline
from creditrisk.serve.api import create_app
from .utils import build_test_config


def _train_and_save_pipeline(tmp_path, config):
    selected_columns = config.features.selected_columns or ["feature_1", "feature_2"]
    train_df = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 11, 12, 13],
            "feature_1": [0.2, 0.7, 0.3, 0.8],
            "feature_2": [0.9, 0.1, 0.4, 0.6],
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


def test_fastapi_predict_endpoint_returns_scores(tmp_path):
    selected_columns = ["feature_1", "feature_2"]
    config = build_test_config(tmp_path, selected_columns)
    _train_and_save_pipeline(tmp_path, config)

    app = create_app(config_override=config, model_path=str(config.paths.model_path))

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        payload = {
            "records": [
                {"SK_ID_CURR": 9991, "feature_1": 0.3, "feature_2": 0.8},
                {"SK_ID_CURR": 9992, "feature_1": 0.6, "feature_2": 0.2},
            ]
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 2
        for record in body["predictions"]:
            assert "prediction" in record
            assert "probability" in record
            assert record["prediction"] in (0, 1)
            assert 0 <= record["probability"] <= 1
