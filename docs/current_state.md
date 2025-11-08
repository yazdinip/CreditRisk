# Current Repo State

## Production Snapshot

- The Kaggle overview frames the problem as predicting how capable each applicant is of repaying a loan; the latest research export (`notebooks/another_copy_of_home_credit_default_risk_eda (3).py`) is therefore treated as the canonical feature spec translating that question into lender-ready signals.
- `src/creditrisk/features/feature_store.py` replays that DuckDB SQL verbatim, so the packaged pipeline loads the six Kaggle tables (application, bureau, bureau_balance, previous_application, installments_payments, credit_card_balance, POS_CASH) and emits the same ~160 curated features our stakeholders validated in research.
- `features/preprocess.py`, `models/baseline.py`, and `pipelines/train_baseline.py` implement the governed preprocessing path (missing-count injection, categorical pruning, SMOTE + downsampling, scaler + XGBoost pipeline) so analysts can rerun the sanctioned flow without notebook glue.
- Configuration in `configs/baseline.yaml` keeps the expanded data paths and the curated `features.selected_columns` under version control, giving CI/CD a clear map of when the approved signal list changes.
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
- `pytest tests` covers the evaluation helper, batch CLI, and FastAPI routes using synthetic data, so future feature-store changes can't silently break inference behavior.

## Business & Monitoring Alignment

- `docs/ML_Ops_Project_Proposal.pdf` defines the success measures (>=95% nightly pipeline success, schema-drift guardrails, dual promoted MLflow versions); this repo wires those hooks directly into configs and CI entry points.
- Great Expectations / Pandera checks plus Evidently drift reports are on the short-term roadmap so data-quality regressions surface before partner dashboards or lending queues see them.
- Batch and FastAPI tooling share artifacts/logging so model owners get one source of truth for approvals, rollback drills, and regulatory evidence.
