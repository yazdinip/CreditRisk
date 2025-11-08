# Architecture Notes

## High-Level Flow

1. **Data Layer**
   - Raw Kaggle extracts land in `data/raw/` and are tracked with DVC (`data/raw/application_train.csv.dvc`).
   - Deterministic preparation steps (feature selection, sampling, imputations) stay inside
     `src/creditrisk/data` and `src/creditrisk/features`.
2. **Training & Evaluation**
   - `dvc repro train_baseline` calls `python -m creditrisk.pipelines.train_baseline`.
   - The `creditrisk` package builds a preprocessing `ColumnTransformer`, instantiates the
     requested estimator, and outputs both serialized models and metrics JSON files (both tracked via DVC).
3. **Experiment Tracking**
   - MLflow logs parameters/metrics to `mlruns/` by default; switch `tracking_uri` in
     `configs/baseline.yaml` to talk to a remote server (e.g., MLflow Tracking, Databricks).
4. **Registry / Deployment (future work)**
   - Once a run is promoted, push the artifact to a registry (MLflow model registry or S3) and
     reuse the same package for batch scoring or real-time endpoints.

```
┌──────────────┐      ┌────────────────────┐      ┌─────────────────┐      ┌──────────────────────┐
│ data/raw/*   │ ───▶ │ creditrisk.data     │ ───▶ │ creditrisk.model │ ───▶ │ reports/ + mlflow     │
└──────────────┘      └────────────────────┘      └─────────────────┘      └──────────────────────┘
        ▲                      │                              │                        │
        │                      ▼                              ▼                        ▼
        │             creditrisk.features             models/baseline*.joblib   dashboards / alerts
```

## Environments & Automation

- **Local dev**: run notebooks or pipeline scripts directly; artifacts saved to the repo.
- **CI**: install requirements, run unit tests, and dry-run `dvc repro` to make sure DAGs stay
  valid. Add static checks (ruff, mypy) as the package matures.
- **CD**: package the trained model (MLflow or custom Docker) and deploy via GitHub Actions
  into SageMaker. Promotion gates reference MLflow metrics and validation reports under `reports/`.

## Observability & Risk

- Add data contracts via Great Expectations inside `src/creditrisk/data` to block malformed inputs.
- Capture drift metrics with Evidently and log them alongside training metrics so the MLflow run
  already contains thresholds for monitoring.
- The project proposal calls out canary/shadow deployments—keep the `tracking.tags.stage` up to
  date so production runs can be filtered quickly.
