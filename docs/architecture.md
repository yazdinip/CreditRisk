# Architecture Notes

## Problem Context

- The Home Credit Default Risk challenge (per Kaggle) asks us to predict how capable each applicant is of repaying a loan so underbanked customers can access responsible credit.
- `docs/ML_Ops_Project_Proposal.pdf` extends that challenge into an internal program charter: reproducible data pipelines, audit-ready experiment tracking, CI/CD hooks into SageMaker/FastAPI deployments, and drift/quality monitoring before decisions reach lending desks.

## High-Level Flow

1. **Data Layer**
   - All Kaggle extracts (application, bureau, bureau_balance, previous_application, installments_payments, credit_card_balance, POS_CASH) land in `data/raw/` and are tracked through DVC so the feature store always has deterministic inputs.
   - `src/creditrisk/features/feature_store.py` runs the original DuckDB SQL notebook to enrich the application table with every aggregate. Additional feature utilities live under `src/creditrisk/features`.
2. **Feature Engineering (`build_feature_store`)**
   - `python -m creditrisk.pipelines.build_feature_store` loads all raw tables, executes the DuckDB SQL, performs the project-specific post-processing (anomaly fixes, missing counts, column drops), and writes `data/processed/feature_store.parquet`.
   - The stage is cached by DVC, so expensive SQL only runs when configs or upstream CSVs change.
3. **Dataset Splitting (`split_data`)**
   - `python -m creditrisk.pipelines.split_data` consumes the feature store parquet, stratifies on `TARGET`, and materializes the deterministic splits under `data/processed/train.parquet` and `data/processed/test.parquet`.
   - Because the splits are artifacts, every retrain/evaluation pair sees identical rows unless the upstream feature store is intentionally refreshed.
4. **Training & Evaluation (`train_baseline`)**
   - `python -m creditrisk.pipelines.train_baseline` now reads the cached splits, rebalances (SMOTE + downsampling), fits the pipeline, emits metrics/plots, saves the model, and logs to MLflow.
   - DVC-tracked outputs remain: `models/baseline_model.joblib`, `reports/metrics.json`, and `reports/evaluation/`.
5. **Experiment Tracking**
   - MLflow logs parameters/metrics/tags to the `baseline` experiment (local `mlruns/` by default). Switch the tracking URI in `configs/baseline.yaml` or via `MLFLOW_TRACKING_URI` when pointing at a remote server.
6. **Serving / Deployment**
   - Batch scoring uses `creditrisk.pipelines.batch_predict`; online scoring uses the FastAPI app in `creditrisk.serve.api`.
   - Promotion flow: approve an MLflow run, push the serialized pipeline to the desired registry/storage, and reuse the same artifact for both the batch CLI and the FastAPI container.

```
raw Kaggle CSVs
      |
      v
build_feature_store  (engineered parquet)
      |
      v
split_data  (train/test parquet)
      |
      v
train_baseline -> MLflow metrics + reports/evaluation + models/baseline_model.joblib
```

## Data Contracts & Validation

- **Pandera contracts** guard each raw extract (`application`, `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `credit_card_balance`, `POS_CASH_balance`) with column-level expectations on identifiers, target encoding, monetary fields, and sentinel handling (`src/creditrisk/validation/contracts.py`).
- **Feature store checks** ensure the engineered parquet retains the entity key (`SK_ID_CURR`), binary targets, and the configured feature list before it can fan out to downstream stages.
- **Split guarantees** prevent duplicate customers or train/test leakage and make sure persisted splits always contain legal targets; validations run both when `split_data` creates the artifacts and inside `train_baseline` before model fitting.
- **Model IO validation** blocks training if the scaler/model receives NaNs, infs, or missing feature columns after preprocessing; toggles live in the `validation` section of `configs/baseline.yaml`.
- **Config-driven enforcement** lets environments relax checks (e.g., turn off raw-table contracts for quick-and-dirty experimentation) without touching the pipeline code.

## Environments & Automation

- **Local dev**: run notebooks or the packaged pipelines directly; artifacts drop into `reports/` and MLflow.
- **CI**: install requirements, execute `pytest tests`, and dry-run `dvc repro train_baseline` to ensure the DAG stays valid whenever configs/source change.
- **CD**: package the trained pipeline (MLflow model / Docker), wire GitHub Actions (or similar) to promote models and deploy the FastAPI service / batch job images.

## Observability & Risk

- Add data contracts (Great Expectations, Pandera) ahead of the DuckDB SQL to block malformed CSVs.
- Capture drift metrics with Evidently and store them beside the evaluation bundle so every run has QA context even without the MLflow UI.
- Keep MLflow tags (e.g., `tracking.tags.stage`) current; they will drive promotion logic, dashboards, and any future canary/shadow deployment strategy.
