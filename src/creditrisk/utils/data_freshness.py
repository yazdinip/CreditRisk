"""Lightweight utilities for summarizing data freshness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from creditrisk.config import Config


def _load_ingestion_report(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _ingestion_mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _summarize_artifacts(artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = Counter(artifact.get("status", "unknown") for artifact in artifacts)
    failing = [
        artifact.get("name", "<unknown>")
        for artifact in artifacts
        if artifact.get("status") in {"missing", "failed"}
    ]
    return {"status_counts": dict(status_counts), "failing_artifacts": failing}


def build_freshness_report(
    config_path: str = "configs/baseline.yaml",
    max_age_hours: float = 24.0,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """Generate a simple freshness report derived from the ingestion metadata."""
    config = Config.from_yaml(config_path)
    ingestion_report_path = config.paths.ingestion_report
    report_destination = output_path or (config.paths.reports_dir / "data_freshness.json")

    generated_at = datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "generated_at": generated_at.isoformat(),
        "threshold_hours": max_age_hours,
        "ingestion_report": str(ingestion_report_path),
        "ingestion_report_exists": ingestion_report_path.exists(),
    }

    ingestion_payload = _load_ingestion_report(ingestion_report_path)
    if ingestion_payload:
        artifacts = ingestion_payload.get("artifacts", [])
        artifact_summary = _summarize_artifacts(artifacts)
        report.update(artifact_summary)

    ingestion_time = _ingestion_mtime(ingestion_report_path)
    if ingestion_time:
        report["ingestion_report_mtime"] = ingestion_time.isoformat()
        age_hours = (generated_at - ingestion_time).total_seconds() / 3600
        report["hours_since_ingestion"] = round(age_hours, 2)

    failing_artifacts = report.get("failing_artifacts", [])
    data_age = report.get("hours_since_ingestion")
    stale = (
        data_age is None
        or data_age > max_age_hours
        or (isinstance(failing_artifacts, list) and len(failing_artifacts) > 0)
    )
    report["stale"] = stale
    report["status"] = "stale" if stale else "fresh"

    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ingestion freshness metadata.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the experiment configuration YAML.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=24.0,
        help="Maximum allowed hours since the last ingestion run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the destination for the freshness report.",
    )
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="Exit with a non-zero code when the data is stale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else None
    report = build_freshness_report(args.config, args.max_age_hours, output_path)
    print(json.dumps(report, indent=2))  # noqa: T201 - helpful context in CI logs
    if args.fail_on_stale and report.get("stale"):
        hours = report.get("hours_since_ingestion", "unknown")
        raise SystemExit(f"Data freshness threshold exceeded ({hours=} hours).")


if __name__ == "__main__":
    main()
