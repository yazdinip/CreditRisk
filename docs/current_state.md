# Current Repo State

## From Notebook to Package

- The latest Colab export (`notebooks/another_copy_of_home_credit_default_risk_eda (3).py`) is now the single source of truth for every engineered feature: application clean-up, bureau/bureau_balance aggregates, previous-application metrics, installments payments stats, credit-card utilization ratios, and POS cash delinquency signals.
- `src/creditrisk/features/feature_store.py` replays that DuckDB SQL verbatim, so the packaged pipeline loads the six Kaggle tables (application, bureau, bureau_balance, previous_application, installments_payments, credit_card_balance, POS_CASH) and emits the exact same ~160 features the notebook used.
- `features/preprocess.py`, `models/baseline.py`, and `pipelines/train_baseline.py` still mirror the Colab flow (missing-count injection, categorical pruning, SMOTE + downsampling, scaler + XGBoost pipeline), but they now run end-to-end without any notebook glue.
- Configuration in `configs/baseline.yaml` includes the expanded data paths and the curated `features.selected_columns`, keeping the notebook parity explicit and version-controlled.
- Evaluation artifacts (ROC, PR, calibration, confusion plots + counts) are automatically written to `reports/evaluation/` via `creditrisk.utils.evaluation`, giving reviewers tangible evidence for every training run even outside MLflow.

## DVC Integration

- `dvc.yaml` describes the `train_baseline` stage with explicit dependencies on every raw Kaggle CSV and the config/source files. Outputs now include the serialized pipeline, `reports/metrics.json`, and `reports/evaluation/`.
- Running `dvc repro train_baseline` rebuilds the DuckDB feature store, trains, evaluates, updates `dvc.lock`, and keeps large artifacts out of git while still reproducible via DVC remotes.

## MLflow Integration

- `log_with_mlflow` logs parameters, metrics (accuracy, precision, recall, F1, ROC-AUC), tags, and the sklearn pipeline to the `baseline` experiment (defaulting to the local `mlruns/` folder unless `tracking_uri` is overridden).
- Each DVC run produces one MLflow run, aligning git commit, DVC data hash, and artifact bundle for auditability.

## Inference + Testing

- `src/creditrisk/pipelines/batch_predict.py` provides a CLI to score CSVs with configurable thresholds, emitting `prediction` and `probability` columns for BI teams.
- `src/creditrisk/serve/api.py` exposes the same helpers through FastAPI (`/predict`, `/health`), paving the way for containerized deployment.
- `pytest tests` covers the evaluation helper, batch CLI, and FastAPI routes using synthetic data, so future feature-store changes can’t silently break inference behavior.
