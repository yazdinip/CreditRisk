"""Generate data drift reports for train vs. current datasets."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from scipy.stats import ks_2samp

from creditrisk.config import Config
from creditrisk.pipelines.data_workflow import load_dataframe

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    _EVIDENTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    ColumnMapping = None  # type: ignore[assignment]
    DataDriftPreset = None  # type: ignore[assignment]
    Report = None  # type: ignore[assignment]
    _EVIDENTLY_AVAILABLE = False


@dataclass
class DriftReportSummary:
    dataset_drift: bool
    share_drifted_columns: float
    drifted_columns: List[str]

    def to_dict(self) -> dict:
        return {
            "dataset_drift": self.dataset_drift,
            "share_drifted_columns": round(float(self.share_drifted_columns), 4),
            "drifted_columns": self.drifted_columns,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Evidently-based drift reports.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the experiment configuration file.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Override path for the reference (typically training) dataset parquet.",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Override path for the current (typically test/inference) dataset parquet.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Override output path for the JSON drift report.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default=None,
        help="Override output path for the HTML drift dashboard.",
    )
    return parser.parse_args()


def _select_columns(
    df: pd.DataFrame,
    *,
    monitored_features: Optional[Iterable[str]],
    target_column: str,
    entity_column: str,
) -> pd.DataFrame:
    cols = list(monitored_features) if monitored_features else list(df.columns)
    filtered = [
        col
        for col in cols
        if col in df.columns and col not in {target_column, entity_column}
    ]
    if not filtered:
        filtered = [
            col
            for col in df.columns
            if col not in {target_column, entity_column}
        ]
    return df[filtered]


def _maybe_sample(df: pd.DataFrame, sample_size: Optional[int]) -> pd.DataFrame:
    if sample_size and sample_size < len(df):
        return df.sample(sample_size, random_state=42)
    return df


def _trim_features(df: pd.DataFrame, max_features: Optional[int]) -> pd.DataFrame:
    if max_features and max_features > 0 and len(df.columns) > max_features:
        selected = list(df.columns)[:max_features]
        return df[selected]
    return df


def generate_drift_report(
    config: Config,
    *,
    reference_path: Optional[Path] = None,
    current_path: Optional[Path] = None,
    json_output: Optional[Path] = None,
    html_output: Optional[Path] = None,
) -> DriftReportSummary:
    """Build Evidently drift artifacts and persist them to disk."""
    if not config.monitoring.enabled or not config.monitoring.drift_enabled:
        LOGGER.info("Monitoring disabled; writing empty drift summary.")
        summary = DriftReportSummary(
            dataset_drift=False,
            share_drifted_columns=0.0,
            drifted_columns=[],
        )
        json_path = json_output or config.paths.drift_report
        html_path = html_output or config.paths.drift_dashboard
        json_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"monitoring_enabled": False, **summary.to_dict()}, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(
            "<html><body><p>Monitoring disabled via configuration.</p></body></html>",
            encoding="utf-8",
        )
        return summary

    reference = load_dataframe(reference_path or config.paths.train_set_path)
    current = load_dataframe(current_path or config.paths.test_set_path)

    reference = _maybe_sample(reference, config.monitoring.reference_sample_size)
    current = _maybe_sample(current, config.monitoring.current_sample_size)

    monitored_features = config.monitoring.feature_list
    reference = _select_columns(
        reference,
        monitored_features=monitored_features,
        target_column=config.data.target_column,
        entity_column=config.data.entity_id_column,
    )
    current = _select_columns(
        current,
        monitored_features=monitored_features,
        target_column=config.data.target_column,
        entity_column=config.data.entity_id_column,
    )

    reference = _trim_features(reference, config.monitoring.max_features)
    current = current[reference.columns]

    backend = (config.monitoring.backend or "evidently").lower()
    json_path = json_output or config.paths.drift_report
    html_path = html_output or config.paths.drift_dashboard
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    if backend != "evidently" or not _EVIDENTLY_AVAILABLE:
        summary, payload, html_content = _generate_fallback_report(
            reference,
            current,
            threshold=config.monitoring.stat_test_threshold,
            backend="ks",
            error=None if backend != "evidently" else "evidently backend disabled",
        )
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        return summary

    try:
        summary, payload, html_content = _generate_evidently_report(
            reference,
            current,
            stat_test=config.monitoring.stat_test,
            stat_threshold=config.monitoring.stat_test_threshold,
        )
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        return summary
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Evidently drift report failed; falling back to KS summary.")
        summary, payload, html_content = _generate_fallback_report(
            reference,
            current,
            threshold=config.monitoring.stat_test_threshold,
            backend="ks_fallback",
            error=str(exc),
        )
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        return summary


def _generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    stat_test: str,
    stat_threshold: float,
) -> Tuple[DriftReportSummary, dict, str]:
    if not (_EVIDENTLY_AVAILABLE and Report and ColumnMapping and DataDriftPreset):
        raise RuntimeError("Evidently is not available.")

    report = Report(
        metrics=[
            DataDriftPreset(
                stattest=stat_test,
                stattest_threshold=stat_threshold,
            )
        ]
    )
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=ColumnMapping(target=None, prediction=None),
    )

    json_payload = json.loads(report.json())
    summary = _summarize_report(json_payload)
    LOGGER.info(
        "Evidently drift summary -> dataset_drift=%s, share_drifted_columns=%.2f%%, drifted_columns=%s",
        summary.dataset_drift,
        summary.share_drifted_columns * 100,
        summary.drifted_columns,
    )
    html_content = report.as_html()
    json_payload.update(
        {
            "backend": "evidently",
            "summary": summary.to_dict(),
        }
    )
    return summary, json_payload, html_content


def _generate_fallback_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    threshold: float,
    backend: str,
    error: Optional[str],
) -> Tuple[DriftReportSummary, dict, str]:
    per_column = []
    drifted_columns: List[str] = []
    for column in reference.columns:
        ref_series = pd.to_numeric(reference[column], errors="coerce").dropna()
        cur_series = pd.to_numeric(current[column], errors="coerce").dropna()
        if ref_series.empty or cur_series.empty:
            continue
        stat, p_value = ks_2samp(ref_series, cur_series, method="auto")
        is_drift = bool(p_value < threshold)
        if is_drift:
            drifted_columns.append(column)
        per_column.append(
            {
                "column": column,
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drift_detected": is_drift,
            }
        )
    total = len(per_column) or 1
    share = len(drifted_columns) / total
    summary = DriftReportSummary(
        dataset_drift=share >= 0.1 and len(drifted_columns) > 0,
        share_drifted_columns=share,
        drifted_columns=drifted_columns,
    )
    payload = {
        "backend": backend,
        "error": error,
        "summary": summary.to_dict(),
        "columns": per_column,
    }
    html_rows = "\n".join(
        f"<tr><td>{row['column']}</td><td>{row['ks_statistic']:.4f}</td><td>{row['p_value']:.4f}</td><td>{row['drift_detected']}</td></tr>"
        for row in per_column
    )
    html_content = f"""
    <html>
      <head><title>Drift Report ({backend})</title></head>
      <body>
        <h1>Data Drift Summary ({backend})</h1>
        <p>Dataset drift: {summary.dataset_drift}</p>
        <p>Share drifted columns: {summary.share_drifted_columns:.2%}</p>
        <table border="1" cellpadding="4" cellspacing="0">
          <thead>
            <tr><th>Column</th><th>KS Statistic</th><th>p-value</th><th>Drift</th></tr>
          </thead>
          <tbody>
            {html_rows}
          </tbody>
        </table>
        {"<p>Error:" + error + "</p>" if error else ""}
      </body>
    </html>
    """
    LOGGER.info(
        "Fallback drift summary -> dataset_drift=%s, share_drifted_columns=%.2f%%, drifted_columns=%s",
        summary.dataset_drift,
        summary.share_drifted_columns * 100,
        summary.drifted_columns,
    )
    return summary, payload, html_content


def _summarize_report(payload: dict) -> DriftReportSummary:
    metrics = payload.get("metrics", [])
    drift_payload = next(
        (metric.get("result", {}) for metric in metrics if "DataDrift" in metric.get("metric", "")),
        {},
    )
    dataset_drift = bool(drift_payload.get("dataset_drift", False))
    share_drifted = float(drift_payload.get("share_drifted_columns", 0.0) or 0.0)

    per_column = drift_payload.get("metrics", [])
    drifted_columns = [
        entry.get("column_name")
        for entry in per_column
        if entry.get("drift_detected")
    ]
    return DriftReportSummary(
        dataset_drift=dataset_drift,
        share_drifted_columns=share_drifted,
        drifted_columns=[col for col in drifted_columns if col],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = Config.from_yaml(args.config)
    generate_drift_report(
        config,
        reference_path=Path(args.reference) if args.reference else None,
        current_path=Path(args.current) if args.current else None,
        json_output=Path(args.output_json) if args.output_json else None,
        html_output=Path(args.output_html) if args.output_html else None,
    )


if __name__ == "__main__":
    main()
