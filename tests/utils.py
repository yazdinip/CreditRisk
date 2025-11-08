from pathlib import Path

from creditrisk.config import (
    Config,
    DataConfig,
    FeaturesConfig,
    InferenceConfig,
    ModelConfig,
    PathsConfig,
    TrackingConfig,
    TrainingConfig,
)


def build_test_config(tmp_path: Path, selected_columns):
    paths = PathsConfig(
        model_dir=tmp_path / "models",
        model_filename="model.joblib",
        metrics_file=tmp_path / "metrics.json",
        reports_dir=tmp_path / "reports",
    )
    data = DataConfig(
        raw_path=tmp_path / "train.csv",
        target_column="TARGET",
        drop_columns=["SK_ID_CURR"],
        stratify=False,
        entity_id_column="SK_ID_CURR",
    )
    features = FeaturesConfig(selected_columns=list(selected_columns))
    training = TrainingConfig(class_weight=None)
    model = ModelConfig(type="logistic_regression", params={"max_iter": 200})
    tracking = TrackingConfig(enabled=False)
    inference = InferenceConfig(decision_threshold=0.4)
    return Config(
        paths=paths,
        data=data,
        features=features,
        training=training,
        model=model,
        tracking=tracking,
        inference=inference,
    )
