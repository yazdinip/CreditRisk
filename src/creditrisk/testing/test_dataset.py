"""Run the trained pipeline against the held-out test split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd

from creditrisk.config import Config
from creditrisk.data.datasets import split_features_target
from creditrisk.models.baseline import evaluate_classifier
from creditrisk.pipelines.data_workflow import load_dataframe
from creditrisk.utils.evaluation import save_evaluation_artifacts

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved model on the persisted test split.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the experiment configuration YAML.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override for the trained model path.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Optional override for the test parquet path.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Where to persist the computed test metrics JSON.",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=None,
        help="Where to persist the per-record predictions parquet.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=None,
        help="Directory to store evaluation plots (roc/pr/confusion/etc).",
    )
    return parser.parse_args()


def _dump_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    LOGGER.info("Test metrics written to %s", path)


def _save_predictions(
    entity_ids: pd.Series,
    entity_column: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: Optional[pd.Series],
    output_path: Path,
) -> None:
    payload = pd.DataFrame(
        {
            entity_column: entity_ids,
            "y_true": y_true,
            "prediction": y_pred,
        }
    )
    if y_prob is not None:
        payload["probability"] = y_prob

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload.to_parquet(output_path, index=False)
    LOGGER.info("Test predictions saved to %s", output_path)


def _load_pipeline(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Could not locate trained model at {model_path}")
    return joblib.load(model_path)


def _compute_scores(pipeline, X_test: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    y_pred = pd.Series(pipeline.predict(X_test), index=X_test.index)
    y_prob = None
    y_score = None

    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X_test)
        if probas.ndim == 2 and probas.shape[1] >= 2:
            y_prob = pd.Series(probas[:, 1], index=X_test.index)
            y_score = y_prob
    if y_score is None and hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        y_score = pd.Series(scores, index=X_test.index)

    return y_pred, y_prob, y_score


def run_test_suite(
    config: Config,
    *,
    model_path: Path,
    test_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    evaluation_dir: Path,
) -> Dict[str, float]:
    LOGGER.info("Loading model from %s", model_path)
    pipeline = _load_pipeline(model_path)

    LOGGER.info("Loading test split from %s", test_path)
    test_df = load_dataframe(test_path)

    entity_column = config.data.entity_id_column
    target_column = config.data.target_column

    entity_ids = test_df[entity_column].reset_index(drop=True)
    X_test, y_test = split_features_target(
        test_df,
        target_column=target_column,
        drop_columns=[entity_column],
    )
    X_test = X_test.astype(float)

    y_pred, y_prob, y_score = _compute_scores(pipeline, X_test)
    metrics = evaluate_classifier(pipeline, X_test, y_test)

    _dump_metrics(metrics, metrics_path)
    _save_predictions(
        entity_ids,
        entity_column,
        y_test.reset_index(drop=True),
        y_pred.reset_index(drop=True),
        y_prob,
        predictions_path,
    )

    save_evaluation_artifacts(
        y_true=y_test.to_numpy(),
        y_pred=y_pred.to_numpy(),
        y_score=None if y_score is None else y_score.to_numpy(),
        y_prob=None if y_prob is None else y_prob.to_numpy(),
        output_dir=evaluation_dir,
    )
    LOGGER.info("Evaluation artifacts written to %s", evaluation_dir)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = Config.from_yaml(args.config)

    model_path = Path(args.model_path) if args.model_path else config.paths.model_path
    test_path = Path(args.test_data) if args.test_data else Path(config.paths.test_set_path)
    metrics_path = (
        Path(args.metrics_file) if args.metrics_file else Path(config.paths.metrics_file)
    )
    predictions_path = (
        Path(args.predictions_file)
        if args.predictions_file
        else Path(config.paths.reports_dir) / "test_predictions.parquet"
    )
    evaluation_dir = (
        Path(args.evaluation_dir)
        if args.evaluation_dir
        else Path(config.paths.evaluation_dir)
    )

    run_test_suite(
        config,
        model_path=model_path,
        test_path=test_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        evaluation_dir=evaluation_dir,
    )


if __name__ == "__main__":
    main()
