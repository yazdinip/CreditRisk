"""Extended monitoring for production datasets with retrain triggers."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from creditrisk.config import Config
from creditrisk.monitoring.drift import DriftReportSummary, generate_drift_report

LOGGER = logging.getLogger(__name__)


@dataclass
class ProductionMonitorResult:
    summary: DriftReportSummary
    metrics_path: Path
    retrain_triggered: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run production drift monitoring and optional retraining triggers."
    )
    parser.add_argument("--config", type=str, default="configs/creditrisk_pd.yaml")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Override path for the reference dataset (defaults to monitoring.production_reference_path or train set).",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Override path for the production dataset (defaults to monitoring.production_current_path or test set).",
    )
    parser.add_argument(
        "--publish-metrics",
        action="store_true",
        help="Send drift metrics to CloudWatch when AWS credentials are available.",
    )
    parser.add_argument(
        "--auto-retrain",
        action="store_true",
        help="Force a retrain trigger when drift exceeds the configured threshold.",
    )
    return parser.parse_args()


def _publish_cloudwatch_metrics(namespace: str, metric_name: str, summary: DriftReportSummary) -> None:
    try:
        import boto3  # pragma: no cover - optional dependency
    except ImportError:  # pragma: no cover
        LOGGER.warning("Skipping CloudWatch publishing because boto3 is not installed.")
        return

    try:
        client = boto3.client("cloudwatch")
        client.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Value": summary.share_drifted_columns * 100,
                    "Unit": "Percent",
                },
                {
                    "MetricName": f"{metric_name}_flag",
                    "Value": 1 if summary.dataset_drift else 0,
                    "Unit": "Count",
                },
            ],
        )
    except Exception:  # pragma: no cover
        LOGGER.exception("Failed to publish drift metrics to CloudWatch.")


def _write_metrics_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _execute_retrain_command(command: str) -> None:
    if not command:
        LOGGER.warning("Retrain command is empty; skipping execution.")
        return
    cmd_parts = shlex.split(command)
    LOGGER.info("Triggering retrain via command: %s", command)
    subprocess.run(cmd_parts, check=True)


def run_production_monitor(
    config: Config,
    *,
    reference_override: Optional[Path] = None,
    current_override: Optional[Path] = None,
    publish_metrics: bool = False,
    force_retrain: bool = False,
    retrain_handler=_execute_retrain_command,
) -> ProductionMonitorResult:
    reference_path = (
        reference_override
        or config.monitoring.production_reference_path
        or config.paths.train_set_path
    )
    current_path = (
        current_override
        or config.monitoring.production_current_path
        or config.paths.test_set_path
    )
    if not current_path:
        raise ValueError("Production monitoring requires a current dataset path.")

    summary = generate_drift_report(
        config,
        reference_path=reference_path,
        current_path=current_path,
        json_output=config.paths.production_drift_report,
        html_output=config.paths.production_drift_dashboard,
    )

    metrics_payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "summary": summary.to_dict(),
    }
    _write_metrics_file(config.paths.drift_metrics_file, metrics_payload)

    should_publish = publish_metrics or bool(config.monitoring.cloudwatch_namespace)
    if should_publish and config.monitoring.cloudwatch_namespace:
        _publish_cloudwatch_metrics(
            config.monitoring.cloudwatch_namespace,
            config.monitoring.cloudwatch_metric_name,
            summary,
        )

    should_trigger = summary.dataset_drift and (
        summary.share_drifted_columns >= config.monitoring.retrain_drift_threshold
    )
    triggered = False
    trigger_reason = "threshold_not_met"
    if config.monitoring.auto_retrain and should_trigger:
        retrain_handler(config.monitoring.retrain_command)
        triggered = True
        trigger_reason = "dataset_drift"
    elif force_retrain:
        retrain_handler(config.monitoring.retrain_command)
        triggered = True
        trigger_reason = "manual_override"

    retrain_payload = {
        "generated_at": metrics_payload["generated_at"],
        "triggered": triggered,
        "reason": trigger_reason,
        "share_drifted_columns": summary.share_drifted_columns,
        "command": config.monitoring.retrain_command if triggered else None,
    }
    _write_metrics_file(config.paths.retrain_report, retrain_payload)

    return ProductionMonitorResult(
        summary=summary,
        metrics_path=config.paths.drift_metrics_file,
        retrain_triggered=triggered,
    )


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    run_production_monitor(
        config,
        reference_override=Path(args.reference).expanduser() if args.reference else None,
        current_override=Path(args.current).expanduser() if args.current else None,
        publish_metrics=args.publish_metrics,
        force_retrain=args.auto_retrain,
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
