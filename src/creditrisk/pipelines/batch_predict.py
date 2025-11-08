"""Batch inference CLI that scores a CSV with a saved pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from creditrisk.config import Config
from creditrisk.data.datasets import load_dataset
from creditrisk.prediction.helpers import (
    build_output_frame,
    predict_with_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference with a saved model.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--input-csv", required=True, help="Path to the feature CSV to score.")
    parser.add_argument(
        "--output-csv",
        default="reports/batch_predictions.csv",
        help="Where to write the scored output CSV.",
    )
    parser.add_argument("--model-path", help="Optional explicit model path. Defaults to config.paths.model_path.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional decision threshold override. Defaults to inference.decision_threshold.",
    )
    return parser.parse_args()


def run_batch_inference(
    config: Config,
    input_csv: Path | str,
    output_csv: Path | str,
    model_path: Optional[Path | str] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Score an input CSV and persist the predictions."""
    csv_path = Path(input_csv)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_file = Path(model_path) if model_path else config.paths.model_path
    LOGGER.info("Loading pipeline from %s", model_file)
    pipeline = joblib.load(model_file)

    LOGGER.info("Loading inference data from %s", csv_path)
    raw_df = load_dataset(csv_path)
    feature_df = raw_df.drop(columns=[config.data.target_column], errors="ignore")

    decision_threshold = threshold if threshold is not None else config.inference.decision_threshold

    decisions, probabilities = predict_with_threshold(
        pipeline=pipeline,
        features=feature_df,
        threshold=decision_threshold,
    )
    result_df = build_output_frame(
        raw_df=raw_df,
        decisions=decisions,
        probabilities=probabilities,
        entity_column=config.data.entity_id_column,
    )
    result_df.to_csv(out_path, index=False)
    LOGGER.info("Wrote %d scored rows to %s", len(result_df), out_path)
    return result_df


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    run_batch_inference(
        config=config,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_path=args.model_path,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
