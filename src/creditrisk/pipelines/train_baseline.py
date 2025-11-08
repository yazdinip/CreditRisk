"""Command-line entry point for training the baseline model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

from creditrisk.config import Config
from creditrisk.data.datasets import (
    load_dataset,
    load_optional_dataset,
    split_features_target,
    train_test_split_df,
)
from creditrisk.features.feature_store import (
    SqlFeatureStoreInputs,
    build_feature_store_via_sql,
)
from creditrisk.features.preprocess import (
    EPS_DEFAULT,
    resolve_selected_columns,
)
from creditrisk.models.baseline import (
    build_training_pipeline,
    evaluate_classifier,
    rebalance_training_data,
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
    df["TOT_MISSING_COUNT"] = df.isna().sum(axis=1)

    entity_column = config.data.entity_id_column

    bureau_df = load_optional_dataset(config.data.bureau_path)
    if bureau_df is not None:
        LOGGER.info("Loaded bureau data from %s with shape %s", config.data.bureau_path, bureau_df.shape)
    else:
        raise ValueError("bureau_path is required to run the SQL feature store builder.")

    bureau_balance_df = load_optional_dataset(config.data.bureau_balance_path)
    if bureau_balance_df is not None:
        LOGGER.info(
            "Loaded bureau balance data from %s with shape %s",
            config.data.bureau_balance_path,
            bureau_balance_df.shape,
        )
    else:
        raise ValueError("bureau_balance_path is required to run the SQL feature store builder.")

    prev_app_df = load_optional_dataset(config.data.previous_application_path)
    if prev_app_df is not None:
        LOGGER.info(
            "Loaded previous applications data from %s with shape %s",
            config.data.previous_application_path,
            prev_app_df.shape,
        )
    else:
        raise ValueError("previous_application_path is required to run the SQL feature store builder.")

    sql_inputs = SqlFeatureStoreInputs(
        application_df=df,
        bureau_df=bureau_df,
        bureau_balance_df=bureau_balance_df,
        prev_application_df=prev_app_df,
    )
    feature_store_df = build_feature_store_via_sql(
        sql_inputs,
        eps=config.features.ratio_feature_eps or EPS_DEFAULT,
    )
    if "DAYS_EMPLOYED_ANOMALY" in feature_store_df.columns:
        feature_store_df = feature_store_df.rename(
            columns={"DAYS_EMPLOYED_ANOMALY": "DAYS_EMPLOYED_ANOM"}
        )

    anomaly_value = config.features.days_employed_anomaly_value
    if "DAYS_EMPLOYED" in feature_store_df.columns:
        feature_store_df["DAYS_EMPLOYED_REPLACED"] = feature_store_df[
            "DAYS_EMPLOYED"
        ].replace({anomaly_value: np.nan})

    if "TOT_MISSING_COUNT" in feature_store_df.columns:
        feature_store_df["missing_count"] = feature_store_df["TOT_MISSING_COUNT"]
    else:
        feature_store_df["missing_count"] = feature_store_df.isna().sum(axis=1)
    feature_store_df = feature_store_df.drop(
        columns=[
            col
            for col in config.data.drop_columns
            if col != config.data.target_column
        ],
        errors="ignore",
    )
    feature_store_df = feature_store_df.fillna(0)
    LOGGER.info("Feature store final shape: %s", feature_store_df.shape)

    train_df, test_df = train_test_split_df(
        feature_store_df,
        target_column=config.data.target_column,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=config.data.stratify,
    )

    selected_columns = resolve_selected_columns(
        feature_store_df,
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

    save_model(pipeline, config.paths.model_path)
    dump_metrics(metrics, Path(config.paths.metrics_file))
    log_with_mlflow(config, metrics, pipeline)

    LOGGER.info("Training pipeline completed with metrics: %s", metrics)


if __name__ == "__main__":
    main()
