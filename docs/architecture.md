# Architecture Notes

## Problem Context

- The Home Credit Default Risk challenge (per Kaggle) asks us to predict how capable each applicant is of repaying a loan so underbanked customers can access responsible credit.
- `docs/ML_Ops_Project_Proposal.pdf` extends that challenge into an internal program charter: reproducible data pipelines, audit-ready experiment tracking, CI/CD hooks into SageMaker/FastAPI deployments, and drift/quality monitoring before decisions reach lending desks.

## High-Level Flow

1. **Data Layer**
   - All Kaggle extracts (application, bureau, bureau_balance, previous_application, installments_payments, credit_card_balance, POS_CASH) land in `data/raw/` and are tracked through DVC so the feature store always has deterministic inputs.
   - `src/creditrisk/features/feature_store.py` runs the notebook’s DuckDB SQL to enrich the application table with every aggregate. Additional feature utilities live under `src/creditrisk/features`.
2. **Training & Evaluation**
   - `dvc repro train_baseline` shells into `python -m creditrisk.pipelines.train_baseline`, rebuilding the DuckDB feature store, balancing, fitting, and logging evaluation outputs.
   - Artifacts under DVC control: `models/baseline_model.joblib`, `reports/metrics.json`, and the plot bundle in `reports/evaluation/` (ROC, PR, calibration, confusion counts).
3. **Experiment Tracking**
   - MLflow logs parameters/metrics/tags to the `baseline` experiment (local `mlruns/` by default). Switch the tracking URI in `configs/baseline.yaml` or via `MLFLOW_TRACKING_URI` when pointing at a remote server.
4. **Serving / Deployment**
   - Batch scoring uses `creditrisk.pipelines.batch_predict`; online scoring uses the FastAPI app in `creditrisk.serve.api`.
   - Promotion flow: approve an MLflow run, push the serialized pipeline to the desired registry/storage, and reuse the same artifact for both the batch CLI and the FastAPI container.

```
raw Kaggle CSVs  ──►  DuckDB feature store (creditrisk.features)  ──►  Training pipeline  ──►  MLflow + reports/
                         ▲                                              │
                         └── batch CLI / FastAPI reuse the same saved pipeline ────────────────────────────────► consumers
```

## Environments & Automation

- **Local dev**: run notebooks or the packaged pipelines directly; artifacts drop into `reports/` and MLflow.
- **CI**: install requirements, execute `pytest tests`, and dry-run `dvc repro train_baseline` to ensure the DAG stays valid whenever configs/source change.
- **CD**: package the trained pipeline (MLflow model / Docker), wire GitHub Actions (or similar) to promote models and deploy the FastAPI service / batch job images.

## Observability & Risk

- Add data contracts (Great Expectations, Pandera) ahead of the DuckDB SQL to block malformed CSVs.
- Capture drift metrics with Evidently and store them beside the evaluation bundle so every run has QA context even without the MLflow UI.
- Keep MLflow tags (e.g., `tracking.tags.stage`) current; they will drive promotion logic, dashboards, and any future canary/shadow deployment strategy.
