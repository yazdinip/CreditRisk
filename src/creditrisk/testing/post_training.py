"""CLI utilities for validating trained models against MLOps expectations."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from creditrisk.config import Config

try:  # pragma: no cover - optional dependency
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore

LOGGER = logging.getLogger(__name__)
_SPECIAL_TRACKING_URIS = {"databricks", "databricks-uc", "uc"}


def _normalize_tracking_uri(tracking_uri: str) -> str:
    if not tracking_uri:
        return tracking_uri
    if tracking_uri in _SPECIAL_TRACKING_URIS or "://" in tracking_uri:
        return tracking_uri
    return Path(tracking_uri).resolve().as_uri()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the trained model artifacts against configured thresholds.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the experiment configuration YAML file.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Optional override for the metrics JSON path.",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Optional override for the validation report output.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=None,
        help="Override the directory used to validate evaluation artifacts.",
    )
    return parser.parse_args()


class PostTrainingValidator:
    """Runs the configured set of post-training checks."""

    def __init__(
        self,
        config: Config,
        *,
        metrics_path: Path | None = None,
        report_path: Path | None = None,
        evaluation_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.metrics_path = Path(metrics_path) if metrics_path else Path(config.paths.metrics_file)
        report_filename = config.testing.report_filename or "validation_summary.json"
        default_report_path = Path(config.paths.reports_dir) / report_filename
        self.report_path = Path(report_path) if report_path else default_report_path
        self.evaluation_dir = (
            Path(evaluation_dir) if evaluation_dir else Path(config.paths.evaluation_dir)
        )
        self.lineage_path = Path(config.paths.lineage_file)

    def run(self) -> Dict[str, Any]:
        if not self.config.testing.enabled:
            summary = self._build_summary(status="skipped", checks=[
                {"name": "post_training_tests", "status": "skipped", "details": "Testing disabled via config."}
            ])
            self._write_report(summary)
            LOGGER.info("Post-training testing disabled; skipping.")
            return summary

        metrics = self._load_metrics()
        checks: List[Dict[str, Any]] = []

        checks.append(self._check_metric_thresholds(metrics))
        checks.append(self._check_evaluation_artifacts())
        checks.append(self._check_data_lineage())

        mlflow_run_id = None
        if self.config.testing.require_mlflow and self.config.tracking.enabled:
            mlflow_check, mlflow_run_id = self._check_mlflow_alignment(metrics)
            checks.append(mlflow_check)
        else:
            checks.append(
                {
                    "name": "mlflow_alignment",
                    "status": "skipped",
                    "details": "MLflow validation disabled via config.",
                }
            )

        status = self._aggregate_status(checks)
        summary = self._build_summary(status=status, checks=checks, metrics=metrics, mlflow_run_id=mlflow_run_id)
        self._write_report(summary)

        if status != "passed":
            raise RuntimeError("Post-training validation failed. See report for details.")

        LOGGER.info("Post-training validation succeeded.")
        return summary

    def _aggregate_status(self, checks: List[Dict[str, Any]]) -> str:
        if any(check["status"] == "failed" for check in checks):
            return "failed"
        if all(check["status"] == "skipped" for check in checks):
            return "skipped"
        return "passed"

    def _load_metrics(self) -> Dict[str, float]:
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found at {self.metrics_path}")
        with open(self.metrics_path, "r", encoding="utf-8") as fp:
            metrics = json.load(fp)
        LOGGER.info("Loaded metrics from %s", self.metrics_path)
        return metrics

    def _check_metric_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        min_metrics = self.config.testing.min_metrics or {}
        missing_metrics: List[str] = []
        underperforming: Dict[str, Dict[str, float]] = {}
        evaluated: Dict[str, float | None] = {}
        for metric_name, threshold in min_metrics.items():
            value = metrics.get(metric_name)
            evaluated[metric_name] = value
            if value is None:
                missing_metrics.append(metric_name)
                continue
            if value < threshold:
                underperforming[metric_name] = {"value": value, "threshold": threshold}

        status = "passed"
        details: Dict[str, Any] = {"evaluated_metrics": evaluated}
        if missing_metrics:
            status = "failed"
            details["missing_metrics"] = missing_metrics
        if underperforming:
            status = "failed"
            details["underperforming"] = underperforming

        return {
            "name": "metric_thresholds",
            "status": status,
            "details": details,
        }

    def _check_evaluation_artifacts(self) -> Dict[str, Any]:
        required = self.config.testing.evaluation_artifacts or []
        if not required:
            return {
                "name": "evaluation_artifacts",
                "status": "skipped",
                "details": "No evaluation artifacts configured.",
            }

        base_dir = self.evaluation_dir
        missing: List[str] = []
        for rel_path in required:
            artifact_path = base_dir / rel_path
            if not artifact_path.exists():
                missing.append(str(artifact_path))

        status = "passed" if not missing else "failed"
        details: Dict[str, Any] = {"base_dir": str(base_dir), "required": required}
        if missing:
            details["missing"] = missing

        return {
            "name": "evaluation_artifacts",
            "status": status,
            "details": details,
        }

    def _check_data_lineage(self) -> Dict[str, Any]:
        if not self.lineage_path.exists():
            return {
                "name": "data_lineage",
                "status": "failed",
                "details": f"Lineage file not found at {self.lineage_path}",
            }
        try:
            with open(self.lineage_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except json.JSONDecodeError as exc:
            return {
                "name": "data_lineage",
                "status": "failed",
                "details": f"Lineage file is not valid JSON: {exc}",
            }

        sources = payload.get("sources", {})
        feature_store = payload.get("feature_store")
        if not sources or feature_store is None:
            return {
                "name": "data_lineage",
                "status": "failed",
                "details": "Lineage file is missing required sections (sources/feature_store).",
            }

        return {
            "name": "data_lineage",
            "status": "passed",
            "details": {
                "recorded_sources": list(sources.keys()),
                "feature_store_rows": feature_store.get("rows"),
            },
        }

    def _check_mlflow_alignment(self, metrics: Dict[str, float]) -> Tuple[Dict[str, Any], str | None]:
        if mlflow is None or MlflowClient is None:
            raise ImportError("MLflow is required for MLflow validation checks.")

        tracking_uri = _normalize_tracking_uri(str(self.config.tracking.tracking_uri))
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(self.config.tracking.experiment_name)
        if experiment is None:
            raise RuntimeError(
                f"Could not find MLflow experiment '{self.config.tracking.experiment_name}'.",
            )

        runs = client.search_runs(
            [experiment.experiment_id],
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        if not runs:
            raise RuntimeError(
                f"No MLflow runs found for experiment '{self.config.tracking.experiment_name}'.",
            )

        run = runs[0]
        tolerance = self.config.testing.mlflow_metric_tolerance
        metric_diffs: Dict[str, Dict[str, float]] = {}
        missing_metrics: List[str] = []
        min_metrics = self.config.testing.min_metrics or {}

        for metric_name in min_metrics:
            run_value = run.data.metrics.get(metric_name)
            file_value = metrics.get(metric_name)
            if run_value is None or file_value is None:
                missing_metrics.append(metric_name)
                continue
            if abs(run_value - file_value) > tolerance:
                metric_diffs[metric_name] = {"metrics_file": file_value, "mlflow": run_value}

        tag_mismatches: Dict[str, Dict[str, str]] = {}
        for key, expected_value in (self.config.tracking.tags or {}).items():
            actual_value = run.data.tags.get(key)
            if actual_value != str(expected_value):
                tag_mismatches[key] = {"expected": str(expected_value), "actual": actual_value}

        status = "passed"
        details: Dict[str, Any] = {"run_id": run.info.run_id, "tracking_uri": tracking_uri}
        if metric_diffs:
            status = "failed"
            details["metric_mismatches"] = metric_diffs
        if missing_metrics:
            status = "failed"
            details["missing_metrics"] = missing_metrics
        if tag_mismatches:
            status = "failed"
            details["tag_mismatches"] = tag_mismatches

        return (
            {
                "name": "mlflow_alignment",
                "status": status,
                "details": details,
            },
            run.info.run_id,
        )

    def _build_summary(
        self,
        *,
        status: str,
        checks: List[Dict[str, Any]],
        metrics: Dict[str, float] | None = None,
        mlflow_run_id: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
            "checks": checks,
            "report_path": str(self.report_path),
        }

    def _write_report(self, summary: Dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        LOGGER.info("Validation summary written to %s", self.report_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = Config.from_yaml(args.config)
    validator = PostTrainingValidator(
        config,
        metrics_path=Path(args.metrics_file) if args.metrics_file else None,
        report_path=Path(args.report_file) if args.report_file else None,
        evaluation_dir=Path(args.evaluation_dir) if args.evaluation_dir else None,
    )
    summary = validator.run()
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
