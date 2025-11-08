from pathlib import Path

import numpy as np

from creditrisk.utils.evaluation import save_evaluation_artifacts


def test_save_evaluation_artifacts_creates_expected_files(tmp_path):
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_score = np.linspace(0.1, 0.9, num=5)
    result = save_evaluation_artifacts(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        y_prob=y_score,
        output_dir=tmp_path / "evaluation",
    )

    expected_keys = {
        "confusion_matrix_plot",
        "confusion_matrix_counts",
        "roc_curve",
        "precision_recall_curve",
        "calibration_curve",
    }

    assert expected_keys.issubset(result.keys())
    for key in expected_keys:
        path = Path(result[key])
        assert path.exists(), f"{key} did not create the expected file"
