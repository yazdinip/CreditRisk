# Architecture Notes

## Problem Context

- The Home Credit Default Risk challenge (per Kaggle) asks us to predict how capable each applicant is of repaying a loan so underbanked customers can access responsible credit.
- `docs/ML_Ops_Project_Proposal.pdf` extends that challenge into an internal program charter: reproducible data pipelines, audit-ready experiment tracking, CI/CD hooks into SageMaker/FastAPI deployments, and drift/quality monitoring before decisions reach lending desks.

## High-Level Flow

1. **Data Layer & Ingestion (`ingest_data`)**
   - `python -m creditrisk.pipelines.ingest_data --config configs/baseline.yaml` validates every Kaggle extract listed under `ingestion.sources`, copies/downloads them when needed, and writes a checksum summary to `reports/ingestion_summary.json`.
   - All raw CSVs live in `data/raw/` and are tracked through DVC so downstream stages always have deterministic inputs.
   - `src/creditrisk/features/feature_store.py` runs the original DuckDB SQL notebook to enrich the application table with every aggregate. Additional feature utilities live under `src/creditrisk/features`.
2. **Feature Engineering (`build_feature_store`)**
   - CLI: `python -m creditrisk.pipelines.build_feature_store --config configs/baseline.yaml`.
   - Responsibilities: load the raw tables, enforce Pandera contracts, execute the DuckDB SQL, apply anomaly/missingness fixes, and save `data/processed/feature_store.parquet`.
   - The stage is cached by DVC, so future repros reuse the parquet unless upstream deps change.
3. **Dataset Splitting (`split_data`)**
   - CLI: `python -m creditrisk.pipelines.split_data --config configs/baseline.yaml`.
   - Responsibilities: validate the feature store, stratify on `TARGET`, block duplicate entity IDs or leakage, and persist `data/processed/train.parquet` and `data/processed/test.parquet`.
4. **Training & Evaluation (`train_baseline`)**
   - CLI: `python -m creditrisk.pipelines.train_baseline --config configs/baseline.yaml`.
   - Responsibilities: load cached splits, rebalance (SMOTE + downsampling), fit the sklearn pipeline, save the serialized model + metrics + plots, log to MLflow, and (optionally) register the run in the MLflow Model Registry.
5. **Experiment Tracking**
   - MLflow logs parameters, metrics, artifacts, and registered versions. Local runs default to the `mlruns/` folder but any tracking URI can be configured in `configs/baseline.yaml`.
6. **Serving / Deployment**
   - Batch scoring uses `creditrisk.pipelines.batch_predict`; online scoring uses the FastAPI app in `creditrisk.serve.api`.
   - Both paths now emit structured JSON logs with correlation IDs, entity counts, and latency, giving observability hooks for Splunk/CloudWatch/etc.
   - Promotion flow: the training stage emits `reports/registry_promotion.json`, and CD invokes `python -m creditrisk.pipelines.auto_promote` to advance registered versions automatically when validations pass.

```
raw Kaggle CSVs
      |
      v
ingest_data  (reports/ingestion_summary.json)
      |
      v
build_feature_store  (engineered parquet)
      |
      v
split_data  (train/test parquet)
      |
      +------> monitor_drift (reports/drift_report.{json,html})
      |
      v
train_baseline -> test_model -> validate_model -> post-training reports + MLflow + models
```

## Data Contracts & Validation

- **Pandera contracts** guard every raw table plus engineered outputs (`src/creditrisk/validation/contracts.py`). They validate key uniqueness, allowable sentinel values (e.g., `DAYS_EMPLOYED` = 365243), numeric ranges, and binary targets.
- **ValidationRunner** wires those contracts into each stage and adds business checks (duplicate IDs, train/test overlap, missing engineered columns, NaN/inf detection). See `src/creditrisk/validation/runner.py`.
- **Config toggles** (`validation.*` in `configs/baseline.yaml`) allow you to disable specific checks in exploratory environments without editing the pipeline code.
- **Fast failure semantics** mean schema drift or missingness surges fail before a model trains, keeping nightly SLOs intact.

## Model Registry & Promotion

- **MLflow-backed registry**: when `registry.enabled` is true, every `train_baseline` run registers its serialized pipeline under `registry.model_name`. Versions are automatically staged when they meet the configured metric threshold.
- **ModelRegistryManager** (`src/creditrisk/mlops/registry.py`) encapsulates registration, metric-based staging, and manual promotions. It also provisions the registered model if it does not exist.
- **Promotion CLI**: `python -m creditrisk.pipelines.promote_model --version <n> --stage Production --archive-existing` triggers stage transitions after governance review or automated tests complete.
- **Lineage**: each registered version references the MLflow run id used to train it, so audits can trace metrics, artifacts, and configs that produced the deployed model.

## Environments & Automation

- **Local dev**: run the modular CLIs (`ingest_data`, `build_feature_store`, `split_data`, `train_baseline`) directly or via `dvc repro`. Artifacts are tracked in DVC and MLflow, so you can iterate safely.
- **CI**: install dependencies, run `pytest`, and execute `dvc repro --run-all` (or at least `dvc repro train_baseline`) to catch contract or schema violations early. Failing validation aborts the build with actionable logs.
- **CD**: containerize the batch CLI and FastAPI app, promote MLflow versions via the provided CLI/helper, and deploy to your platform of choice (SageMaker, Vertex, AKS). Promotions should be tied to automated tests + approval workflows.

## Observability & Telemetry

- **Structured logging**: `creditrisk.observability.logging` enforces JSON-formatted logs with request IDs, entity counts, durations, and status codes across both batch and API inference. Evidently drift reports (`reports/drift_report.json` + `.html`) now ride alongside lineage/ingestion artifacts for nightly health checks.
- **Correlation IDs**: the FastAPI middleware issues/propagates `X-Request-ID` for every call, enabling log aggregation and alerting around latency/error spikes.
- **Artifacts**: ingestion summaries, lineage reports, evaluation bundles, validation outcomes, and registry promotion metadata all land under `reports/` for downstream monitoring and audits.

## Containerization & Deployment Artifacts

- `Dockerfile.api` packages the FastAPI service (uvicorn) for AKS/ECS/SageMaker-style hosting.
- `Dockerfile.batch` packages the batch CLI so scheduled scorers run in the same environment as training.
- The CD workflow logs in to GHCR, builds both images via Buildx, and pushes tags for `latest` and the commit SHA so downstream environments can pull immutable versions.
- `.dockerignore` keeps datasets, reports, caches, and git metadata out of the images to keep builds reproducible.
- GitHub Actions CD pulls the artifacts, runs the full pipeline, uploads the reports, and triggers MLflow promotions so environments stay in sync.

## Registry & Governance

- **MLflow-backed registry**: when `registry.enabled` is true, every `train_baseline` run registers its serialized pipeline under `CreditRiskPD`. Versions auto-stage when they meet the configured metric threshold.
- **Promotion automation**: `reports/registry_promotion.json` captures the run id + model version + metrics, and `creditrisk.pipelines.auto_promote` transitions that version to Production once validations succeed, keeping CD push-button.
- **Lineage**: registered versions reference the MLflow run id and inherit the ingestion + feature-store lineage artifacts, so auditors can trace predictions back through data snapshots and configs.
