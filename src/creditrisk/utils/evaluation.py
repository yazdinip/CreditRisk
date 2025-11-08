"""Visualization helpers for model evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def _ensure_path(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def _finalize_plot(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, path: Path | str) -> Path:
    """Save the ROC curve."""
    RocCurveDisplay.from_predictions(y_true, y_score)
    return _finalize_plot(_ensure_path(path))


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, path: Path | str) -> Path:
    """Save the precision-recall curve."""
    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    return _finalize_plot(_ensure_path(path))


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path | str
) -> Path:
    """Save the confusion matrix heatmap."""
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    return _finalize_plot(_ensure_path(path))


def plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, path: Path | str, n_bins: int = 10
) -> Path:
    """Save the calibration curve (probability reliability diagram)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    _ensure_path(path)
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    return _finalize_plot(Path(path))


def dump_confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path | str
) -> Path:
    """Persist raw confusion matrix counts for downstream reporting."""
    cm = confusion_matrix(y_true, y_pred).tolist()
    payload = {"confusion_matrix": cm}
    path_obj = _ensure_path(path)
    path_obj.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path_obj


def save_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    y_prob: Optional[np.ndarray],
    output_dir: Path | str,
) -> Dict[str, Path]:
    """Save evaluation plots + confusion counts and return their paths."""
    output = {}
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    output["confusion_matrix_plot"] = plot_confusion_matrix(
        y_true, y_pred, base / "confusion_matrix.png"
    )
    output["confusion_matrix_counts"] = dump_confusion_matrix_counts(
        y_true, y_pred, base / "confusion_matrix.json"
    )

    if y_score is not None:
        output["roc_curve"] = plot_roc_curve(
            y_true,
            y_score,
            base / "roc_curve.png",
        )
        output["precision_recall_curve"] = plot_pr_curve(
            y_true,
            y_score,
            base / "precision_recall_curve.png",
        )

    if y_prob is not None:
        output["calibration_curve"] = plot_calibration_curve(
            y_true,
            y_prob,
            base / "calibration_curve.png",
        )

    return output
