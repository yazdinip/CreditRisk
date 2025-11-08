"""Command-line entry point for training the baseline model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from sklearn.pipeline import Pipeline

try:
    import mlflow
    import mlflow.sklearn  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

from creditrisk.config import Config
from creditrisk.data.datasets import split_features_target
from creditrisk.features.preprocess import resolve_selected_columns
from creditrisk.models.baseline import (
    build_training_pipeline,
    evaluate_classifier,
    rebalance_training_data,
    save_model,
)
from creditrisk.pipelines.data_workflow import (
    build_feature_store_frame,
    load_dataframe,
    save_dataframe,
    split_feature_store,
)
from creditrisk.utils.evaluation import save_evaluation_artifacts
from creditrisk.validation import ValidationRunner

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
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Optional override for the training split parquet path.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Optional override for the test split parquet path.",
    )
    parser.add_argument(
        "--feature-store",
        type=str,
        default=None,
        help="Optional override for the feature store parquet path.",
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
    validator = ValidationRunner(config.validation)

    entity_column = config.data.entity_id_column
    train_path = Path(args.train_data) if args.train_data else Path(config.paths.train_set_path)
    test_path = Path(args.test_data) if args.test_data else Path(config.paths.test_set_path)

    if train_path.exists() and test_path.exists():
        LOGGER.info("Loading cached train/test splits from %s and %s", train_path, test_path)
        train_df = load_dataframe(train_path)
        test_df = load_dataframe(test_path)
    else:
        LOGGER.info("Cached splits not found; rebuilding from raw data.")
        feature_store_path = (
            Path(args.feature_store)
            if args.feature_store
            else Path(config.paths.feature_store_path)
        )
        if feature_store_path.exists():
            LOGGER.info("Using cached feature store at %s", feature_store_path)
            feature_store_df = load_dataframe(feature_store_path)
            validator.validate_feature_store(
                feature_store_df,
                target_column=config.data.target_column,
                required_feature_columns=config.features.selected_columns,
            )
        else:
            feature_store_df = build_feature_store_frame(config, validator=validator)
            save_dataframe(feature_store_df, feature_store_path)
        train_df, test_df = split_feature_store(feature_store_df, config)
        save_dataframe(train_df, train_path)
        save_dataframe(test_df, test_path)

    validator.validate_split(
        train_df,
        target_column=config.data.target_column,
        entity_column=entity_column,
        split_name="train",
    )
    validator.validate_split(
        test_df,
        target_column=config.data.target_column,
        entity_column=entity_column,
        split_name="test",
    )
    validator.validate_split_pair(train_df, test_df, entity_column)

    selected_columns = resolve_selected_columns(
        train_df,
        config.data.target_column,
        config.features,
    )
    LOGGER.info("Using %d engineered features.", len(selected_columns))

    X_train, y_train = split_features_target(
        train_df,
        target_column=config.data.target_column,
        drop_columns=[entity_column],
    )
    X_test, y_test = split_features_target(
        test_df,
        target_column=config.data.target_column,
        drop_columns=[entity_column],
    )

    X_train = X_train[selected_columns].astype(float)
    X_test = X_test[selected_columns].astype(float)
    validator.validate_feature_matrix(X_train, selected_columns, stage_name="train_features")
    validator.validate_feature_matrix(X_test, selected_columns, stage_name="test_features")

    X_balanced, y_balanced = rebalance_training_data(
        X_train,
        y_train,
        config.training,
    )
    LOGGER.info(
        "Training rows before balancing: %d | after balancing: %d",
        len(X_train),
        len(X_balanced),
    )

    pipeline = build_training_pipeline(
        selected_columns=selected_columns,
        model_cfg=config.model,
        training_cfg=config.training,
    )
    pipeline.fit(X_balanced, y_balanced)
    LOGGER.info("Finished training model %s", config.model.type)

    metrics = evaluate_classifier(pipeline, X_test, y_test)

    y_pred = pipeline.predict(X_test)
    y_score = None
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_score = y_prob
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X_test)

    save_evaluation_artifacts(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_score=y_score,
        y_prob=y_prob,
        output_dir=config.paths.evaluation_dir,
    )

    save_model(pipeline, config.paths.model_path)
    dump_metrics(metrics, Path(config.paths.metrics_file))
    log_with_mlflow(config, metrics, pipeline)

    LOGGER.info("Training pipeline completed with metrics: %s", metrics)


if __name__ == "__main__":
    main()
