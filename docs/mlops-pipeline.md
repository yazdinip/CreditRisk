# MLOps Pipeline Blueprint

## Stages

| Stage | Status | Owner | Tooling | Description |
|-------|--------|-------|---------|-------------|
| `data_ingest` | TODO | Data & Monitoring | DVC + Great Expectations | Validate schema/dtypes for each Kaggle CSV, push clean parquet/duckdb tables to `data/interim`. |
| `feature_build` | ✅ | Modeling | DuckDB via `creditrisk.features.feature_store` | Recreate the notebook feature store (applications + bureau/bureau_balance/previous + installments + credit_card + POS cash) and emit the curated column list. |
| `train_baseline` | ✅ | Modeling | `dvc repro train_baseline` / `python -m creditrisk.pipelines.train_baseline` | Train the XGBoost baseline with notebook-style preprocessing, log metrics + plots, and persist the pipeline to `models/`. |
| `evaluate` | ✅ (plots) / TODO (comparisons) | QA | `creditrisk.utils.evaluation`, Evidently (planned) | Current stage saves ROC/PR/calibration/confusion artifacts to `reports/evaluation/`; next step is adding historical comparisons/drift checks. |
| `batch_infer` | ✅ | Modeling / Ops | `python -m creditrisk.pipelines.batch_predict` | Load a saved pipeline, score CSVs, and emit `prediction` + `probability` with configurable thresholds. |
| `serve_online` | ✅ (local) / TODO (prod) | DevOps | FastAPI + Uvicorn | FastAPI service (`creditrisk.serve.api`) exposes `/predict` + `/health`; needs Dockerization + infra wiring for prod. |
| `deploy` | TODO | DevOps | Docker, MLflow Registry, SageMaker | Containerize the approved model, push to registry, and deploy via GitHub Actions. |
| `monitor` | TODO | Monitoring | Evidently, CloudWatch/Grafana | Scheduled jobs scoring fresh production data, drift/latency checks, retrain triggers. |

## Orchestration

- **Local**: use DVC (`dvc repro`) to execute DAGs with reproducible params and inputs.
- **Remote**: plug the same stages into GitHub Actions or another orchestrator (Step Functions, Airflow, Dagster) by shelling out to DVC or reusing the Python entry points directly.

## Configuration Strategy

- All knobs live in YAML under `configs/`; DVC `params` keeps track of the values that should trigger re-runs.
- Secrets (AWS credentials, MLflow tokens, etc.) stay out of YAML—inject them via environment variables or CI/CD secret stores.

## Experiment Tracking

- Each training run logs metrics + artifacts to MLflow with helpful tags (`project`, `stage`). Evaluation plots are also stored in git-ignored `reports/evaluation/` for offline QA.
- Use MLflow's Model Registry (or a simple object store) to promote versions once validation succeeds. Store model cards/notes in `reports/` so reviewers get qualitative context.

## Data & Model Lineage

1. Raw CSV tracked with DVC remote (e.g., Azure Blob, S3, GCS).
2. DuckDB feature store regenerated deterministically during `train_baseline`.
3. MLflow run contains git commit SHA; add a `dvc_rev` tag when wiring CI so auditors can trace predictions back to source data.

## Next Iterations

1. Add schema/quality checks prior to running the DuckDB SQL (Great Expectations or Pandera).
2. Introduce Evidently/GX reports for drift + calibration comparisons between runs, wired into the `evaluate` stage.
3. Containerize the FastAPI app + batch CLI, bake deployment workflows, and hook the monitoring plan from the proposal into production endpoints.
