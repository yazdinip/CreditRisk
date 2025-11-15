"""Compare a candidate deployment against the current baseline before rollout."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from creditrisk.config import Config
from creditrisk.pipelines.data_workflow import load_dataframe
from creditrisk.prediction.helpers import predict_with_threshold

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a candidate model against the production baseline."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--baseline-model",
        required=True,
        help="Path to the currently deployed/approved model (joblib).",
    )
    parser.add_argument(
        "--candidate-model",
        required=True,
        help="Path to the model slated for promotion (joblib).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Reference dataset (parquet/csv) used to compare scoring behavior.",
    )
    parser.add_argument(
        "--max-metric-delta",
        type=float,
        default=0.02,
        help="Maximum allowed absolute delta for approval rate & avg probability.",
    )
    parser.add_argument(
        "--output",
        default="reports/canary_report.json",
        help="Where to write the validation summary.",
    )
    return parser.parse_args()


def _prepare_features(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[config.data.target_column], errors="ignore")
    selected = list(config.features.selected_columns or features.columns)
    missing = sorted(set(selected) - set(features.columns))
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {', '.join(missing)}")
    return features[selected]


def _score_model(model, features: pd.DataFrame, threshold: float) -> Dict[str, float]:
    decisions, probabilities = predict_with_threshold(model, features, threshold)
    probs = probabilities if probabilities is not None else decisions.astype(float)
    stats = {
        "approval_rate": float(np.mean(decisions)),
        "avg_probability": float(np.mean(probs)),
        "std_probability": float(np.std(probs)),
    }
    return stats


def run_canary_validation(
    config: Config,
    *,
    baseline_model_path: Path,
    candidate_model_path: Path,
    dataset_path: Path,
    max_metric_delta: float,
    output_path: Path,
) -> Dict[str, object]:
    dataset = load_dataframe(dataset_path)
    features = _prepare_features(config, dataset)

    import joblib

    baseline_model = joblib.load(baseline_model_path)
    candidate_model = joblib.load(candidate_model_path)

    threshold = config.inference.decision_threshold
    baseline_stats = _score_model(baseline_model, features, threshold)
    candidate_stats = _score_model(candidate_model, features, threshold)

    deltas = {
        "approval_rate_delta": abs(candidate_stats["approval_rate"] - baseline_stats["approval_rate"]),
        "avg_probability_delta": abs(
            candidate_stats["avg_probability"] - baseline_stats["avg_probability"]
        ),
    }

    status = (
        "passed"
        if all(delta <= max_metric_delta for delta in deltas.values())
        else "failed"
    )
    summary: Dict[str, object] = {
        "status": status,
        "threshold": threshold,
        "max_metric_delta": max_metric_delta,
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "deltas": deltas,
        "records_compared": len(features),
        "dataset_path": str(dataset_path),
        "baseline_model_path": str(baseline_model_path),
        "candidate_model_path": str(candidate_model_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if status != "passed":
        raise RuntimeError(
            f"Canary validation failed; deltas exceeded {max_metric_delta:.4f}. See {output_path}"
        )
    LOGGER.info("Canary validation passed; report written to %s", output_path)
    return summary


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.from_yaml(args.config)
    run_canary_validation(
        cfg,
        baseline_model_path=Path(args.baseline_model),
        candidate_model_path=Path(args.candidate_model),
        dataset_path=Path(args.dataset),
        max_metric_delta=args.max_metric_delta,
        output_path=Path(args.output),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
