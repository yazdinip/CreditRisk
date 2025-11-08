# MLOps Pipeline Blueprint

## Stages

| Stage | Owner | Tooling | Description |
|-------|-------|---------|-------------|
| `data_ingest` (future) | Data & Monitoring | DVC + Great Expectations | Validate schema/dtypes, push clean parquet/duckdb tables to `data/interim`. |
| `feature_build` (future) | Modeling | Pandas/Feature Store | Aggregate bureau/application tables, derive risk signals, persist to `data/processed`. |
| `train_baseline` (current) | Modeling | `python -m creditrisk.pipelines.train_baseline` | Train XGBoost baseline with auto ColumnTransformer preprocessing and log metrics to MLflow. |
| `evaluate` (future) | QA | Evidently, custom scripts | Compare current model vs registry champion; emit calibration and fairness diagnostics to `reports/`. |
| `deploy` (future) | DevOps | Docker, FastAPI, SageMaker | Package the selected MLflow artifact, deploy behind an endpoint, configure alarms. |
| `monitor` (future) | Monitoring | Evidently, CloudWatch, Grafana | Scheduled jobs scoring fresh production data, drift/latency checks, retrain triggers. |

## Orchestration

- **Local**: use DVC (`dvc repro`) to execute DAGs with reproducible params and inputs.
- **Remote**: plug the same stages into GitHub Actions or an orchestrator (Step Functions,
  Airflow, Dagster) by shelling out to DVC or reusing the Python entry points directly.

## Configuration Strategy

- All stage knobs live in YAML under `configs/`.
- DVC `params` keeps track of the values that should trigger re-runs.
- Secrets (e.g., AWS credentials, MLflow tokens) should *not* live in YAML; supply them via
  environment variables or the CI/CD secret store.

## Experiment Tracking

- Each training run logs metrics + artifacts to MLflow. The run is tagged with `project` and
  `stage` so dashboards can filter baseline vs. production activity.
- Use MLflow's Model Registry (or a simple S3 bucket) to promote versions once validation
  succeeds. Store model cards/notes in `reports/` so reviewers see qualitative context.

## Data & Model Lineage

1. Raw CSV tracked with DVC remote (e.g., Azure Blob, S3, GCS).
2. Derived datasets tracked as additional DVC outputs.
3. MLflow run contains git commit SHA, DVC data version (via `mlflow.set_tag("dvc_rev", ...)`)
   — add this when wiring CI/CD so auditors can trace predictions back to source data.

## Next Iterations

1. Add `creditrisk/data/contracts.py` with Great Expectations suites and hook them into the DVC DAG.
2. Introduce a lightweight feature store (DuckDB or Feast) for shared aggregates.
3. Containerize serving code (FastAPI) and connect to the monitoring plan from the proposal
   (drift triggers, policy-threshold simulations, SMOTE variants).
