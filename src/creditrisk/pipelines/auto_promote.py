"""CLI helper to promote MLflow model versions after validations pass."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from creditrisk.config import Config
from creditrisk.mlops.registry import ModelRegistryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote MLflow models when validations pass.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/creditrisk_pd.yaml",
        help="Path to the configuration YAML.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        help="Target MLflow stage to promote to.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional override for the registry promotion report.",
    )
    parser.add_argument(
        "--validation-report",
        type=str,
        default=None,
        help="Optional override for the post-training validation summary.",
    )
    parser.add_argument(
        "--transition-log",
        type=str,
        default=None,
        help="Path to write the promotion outcome summary.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)

    if not config.registry.enabled:
        LOGGER.info("Registry disabled in config; skipping promotion.")
        return

    report_path = Path(args.report) if args.report else config.paths.registry_report
    if not report_path.exists():
        LOGGER.info("Registry promotion report not found at %s; skipping.", report_path)
        return

    validation_report = (
        Path(args.validation_report)
        if args.validation_report
        else Path(config.paths.reports_dir) / config.testing.report_filename
    )
    if validation_report.exists():
        summary = _load_json(validation_report)
        if summary.get("status") != "passed":
            LOGGER.warning(
                "Validation report at %s has status %s; refusing to promote.",
                validation_report,
                summary.get("status"),
            )
            return
    else:
        LOGGER.warning("Validation report %s missing; continuing without check.", validation_report)

    payload = _load_json(report_path)
    run_id = payload.get("run_id")
    version = payload.get("model_version")
    model_name = payload.get("model_name") or config.registry.model_name

    if run_id is None or version is None:
        LOGGER.warning("Registry report missing run_id or model_version; skipping.")
        return

    registry_manager = ModelRegistryManager(config)
    registry_manager.transition_stage(
        version=int(version),
        stage=args.stage,
        archive_existing=config.registry.archive_existing_on_promote,
    )
    LOGGER.info(
        "Promoted model %s version %s from run %s to %s.",
        model_name,
        version,
        run_id,
        args.stage,
    )
    if args.transition_log:
        payload = {
            "model_name": model_name,
            "version": version,
            "run_id": run_id,
            "stage": args.stage,
        }
        log_path = Path(args.transition_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Promotion summary written to %s", log_path)


if __name__ == "__main__":
    main()
