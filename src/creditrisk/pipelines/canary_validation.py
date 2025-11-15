"""Compare a candidate deployment against the current production model before rollout."""

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
        description="Validate a candidate model against the active production deployment."
    )
    parser.add_argument("--config", default="configs/creditrisk_pd.yaml")
    parser.add_argument(
        "--production-model",
        default=None,
        help="Path to the currently deployed/approved model (joblib). Defaults to paths.production_model_path.",
    )
    parser.add_argument(
        "--candidate-model",
        required=True,
        help="Path to the model slated for promotion (joblib).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Reference dataset (parquet/csv) used to compare scoring behaviour. Defaults to the cached test split.",
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
    production_model_path: Path,
    candidate_model_path: Path,
    dataset_path: Path,
    max_metric_delta: float,
    output_path: Path,
) -> Dict[str, object]:
    if not candidate_model_path.exists():
        raise FileNotFoundError(f"Candidate model {candidate_model_path} not found.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Reference dataset {dataset_path} not found.")
    if not production_model_path.exists():
        summary = {
            "status": "skipped",
            "reason": "production_missing",
            "production_model_path": str(production_model_path),
            "candidate_model_path": str(candidate_model_path),
            "dataset_path": str(dataset_path),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.warning(
            "Production model %s missing; skipping canary validation.", production_model_path
        )
        return summary

    dataset = load_dataframe(dataset_path)
    features = _prepare_features(config, dataset)

    import joblib

    production_model = joblib.load(production_model_path)
    candidate_model = joblib.load(candidate_model_path)

    threshold = config.inference.decision_threshold
    production_stats = _score_model(production_model, features, threshold)
    candidate_stats = _score_model(candidate_model, features, threshold)

    deltas = {
        "approval_rate_delta": abs(
            candidate_stats["approval_rate"] - production_stats["approval_rate"]
        ),
        "avg_probability_delta": abs(
            candidate_stats["avg_probability"] - production_stats["avg_probability"]
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
        "production": production_stats,
        "candidate": candidate_stats,
        "deltas": deltas,
        "records_compared": len(features),
        "dataset_path": str(dataset_path),
        "production_model_path": str(production_model_path),
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
    production_path = (
        Path(args.production_model)
        if args.production_model
        else Path(cfg.paths.production_model_path)
    )
    candidate_path = Path(args.candidate_model)
    dataset_path = Path(args.dataset) if args.dataset else Path(cfg.paths.test_set_path)
    run_canary_validation(
        cfg,
        production_model_path=production_path,
        candidate_model_path=candidate_path,
        dataset_path=dataset_path,
        max_metric_delta=args.max_metric_delta,
        output_path=Path(args.output),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
