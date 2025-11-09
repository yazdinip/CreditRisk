"""Helpers for capturing lightweight data lineage metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd


def _top_missing_fraction(df: pd.DataFrame, top_n: int = 10) -> Dict[str, float]:
    series = df.isna().mean().sort_values(ascending=False)
    series = series[series > 0].head(top_n)
    return {col: round(float(value), 4) for col, value in series.items()}


def _file_metadata(path: Optional[Path | str]) -> Tuple[Optional[int], Optional[str]]:
    if not path:
        return None, None
    path_obj = Path(path)
    if not path_obj.exists():
        return None, None
    stat = path_obj.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return stat.st_size, modified


def _summarize_table(name: str, df: pd.DataFrame, source_path: Optional[Path | str]) -> Dict[str, object]:
    size_bytes, modified_at = _file_metadata(source_path)
    summary: Dict[str, object] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_fraction_top": _top_missing_fraction(df),
        "source_path": str(source_path) if source_path else None,
        "file_size_bytes": size_bytes,
        "file_modified_at": modified_at,
    }
    return summary


def record_data_lineage(
    raw_tables: Mapping[str, Tuple[Optional[Path | str], pd.DataFrame]],
    feature_store_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Persist a JSON report describing the raw inputs and engineered feature store."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            name: _summarize_table(name, df, source_path)
            for name, (source_path, df) in raw_tables.items()
        },
        "feature_store": {
            "rows": int(feature_store_df.shape[0]),
            "columns": int(feature_store_df.shape[1]),
            "column_names": list(feature_store_df.columns),
            "missing_fraction_top": _top_missing_fraction(feature_store_df),
            "source_tables": list(raw_tables.keys()),
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
