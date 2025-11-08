"""Command-line entry point for training the baseline model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

try:
    import mlflow
    import mlflow.sklearn  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

from sklearn.pipeline import Pipeline

from creditrisk.config import Config
from creditrisk.data.datasets import (
    load_dataset,
    split_features_target,
    train_test_split_df,
)
from creditrisk.features.preprocess import build_preprocessor
from creditrisk.models.baseline import (
    build_classifier,
    evaluate_classifier,
    save_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline Home Credit model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the training configuration YAML.",
    )
    return parser.parse_args()


def log_with_mlflow(config: Config, metrics: Dict[str, float], pipeline: Pipeline) -> None:
    """Log params/metrics/artifacts to MLflow if enabled."""
    if not config.tracking.enabled:
        LOGGER.info("MLflow tracking disabled in config.")
        return

    if mlflow is None:
        LOGGER.warning("MLflow is not installed. Skipping experiment logging.")
        return

    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)
    with mlflow.start_run(run_name=config.tracking.run_name):
        mlflow.log_params(
            {
                "model_type": config.model.type,
                "test_size": config.training.test_size,
                "class_weight": config.training.class_weight,
                "random_state": config.training.random_state,
            }
        )
        mlflow.log_metrics(metrics)
        if config.tracking.tags:
            mlflow.set_tags(config.tracking.tags)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")


def dump_metrics(metrics: Dict[str, float], metrics_file: Path) -> None:
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    LOGGER.info("Metrics written to %s", metrics_file)


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    LOGGER.info("Loaded config from %s", args.config)

    df = load_dataset(config.data.raw_path, sample_rows=config.data.sample_rows)
    LOGGER.info("Loaded %s with shape %s", config.data.raw_path, df.shape)

    drop_columns = list(set(config.data.drop_columns + config.features.drop_columns))

    train_df, test_df = train_test_split_df(
        df,
        target_column=config.data.target_column,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=config.data.stratify,
    )

    preprocessor, _, _ = build_preprocessor(
        train_df,
        target_column=config.data.target_column,
        feature_config=config.features,
        global_drop=drop_columns,
    )
    classifier = build_classifier(config.model, config.training)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )

    X_train, y_train = split_features_target(
        train_df,
        target_column=config.data.target_column,
        drop_columns=drop_columns,
    )
    pipeline.fit(X_train, y_train)
    LOGGER.info("Finished training model %s", config.model.type)

    X_test, y_test = split_features_target(
        test_df,
        target_column=config.data.target_column,
        drop_columns=drop_columns,
    )
    metrics = evaluate_classifier(pipeline, X_test, y_test)

    save_model(pipeline, config.paths.model_path)
    dump_metrics(metrics, Path(config.paths.metrics_file))
    log_with_mlflow(config, metrics, pipeline)

    LOGGER.info("Training pipeline completed with metrics: %s", metrics)


if __name__ == "__main__":
    main()
