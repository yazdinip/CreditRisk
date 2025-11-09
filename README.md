# CreditRisk Default Risk Platform

This repository was built as the production-grade implementation of Home Credit's default-risk workflow.  
The goal mirrors the Kaggle brief: *"Predict how capable each applicant is of repaying a loan so Home Credit can extend responsible financing to borrowers with limited credit histories."*  
Everything here treats the Kaggle exports as stand-ins for the lender's feeds and delivers an end-to-end MLOps system: deterministic data lineage, governed training, registry-backed promotion, and deployment-ready artifacts.

---

## Why This Project Exists

- Give credit, servicing, and collections teams an auditable probability-of-default score at application time.
- Move the research-grade notebook (`notebooks/another_copy_of_home_credit_default_risk_eda (3).py`) into governed, repeatable pipelines.
- Meet the proposal's success criteria: reproducible end-to-end runs, MLflow-governed promotions, >95 % nightly pipeline success, and deployment-ready deliverables (batch + FastAPI).
- Embed observability and data contracts so schema drift, missingness spikes, or modelling regressions are caught before they affect lending decisions.

---

## Quickstart

1. **Create an environment & install deps**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate        # Windows (use `source .venv/bin/activate` on macOS/Linux)
   pip install -e .
   ```
   Dependencies live in `requirements.txt` (notably DVC, Pandera, PyArrow, MLflow).

2. **Fetch raw datasets with DVC**
   ```bash
   dvc pull data/raw/application_train.csv.dvc
   # add the remaining CSVs (bureau, bureau_balance, etc.) via `dvc add` if you own the Kaggle exports
   ```
   Keep large artifacts out of git; DVC stores the pointers required for reproducibility.

3. **Run the full pipeline**
   ```bash
   dvc repro train_baseline
   ```
   This executes the modular DAG:
   ```
   data/raw/*.dvc -> build_feature_store -> split_data -> train_baseline
   ```
   Outputs land in:
   - `data/processed/*.parquet` (feature store + deterministic splits)
   - `models/baseline_model.joblib`
   - `reports/metrics.json` and `reports/evaluation/`
   - `mlruns/` (MLflow experiment + registry metadata)

4. **Inspect / iterate**
   - Launch `mlflow ui` to review runs, metrics, and registered model versions.
   - Update configs in `configs/baseline.yaml` (thresholds, registry behavior, validation toggles) and re-run `dvc repro`.
   - Use `python -m creditrisk.pipelines.promote_model --version <n> --stage Production --archive-existing` after governance sign-off.

---

## Pipeline Overview

| Stage | CLI Entry Point | Description | Artifacts |
|-------|-----------------|-------------|-----------|
| `ingest_data` | `python -m creditrisk.pipelines.ingest_data` | Verifies or fetches the raw Kaggle extracts defined in `configs/baseline.yaml`, captures checksums, and writes `reports/ingestion_summary.json`. | `reports/ingestion_summary.json`, validated raw CSVs |
| `build_feature_store` | `python -m creditrisk.pipelines.build_feature_store` | Loads all seven Kaggle extracts, enforces Pandera contracts, replays the DuckDB SQL feature engineering, and persists `data/processed/feature_store.parquet`. | Feature store parquet (165 cols) |
| `split_data` | `python -m creditrisk.pipelines.split_data` | Validates the feature store, stratifies on `TARGET`, enforces no-leakage guarantees, and writes deterministic train/test parquet files. | `data/processed/train.parquet`, `data/processed/test.parquet` |
| `train_baseline` | `python -m creditrisk.pipelines.train_baseline` | Loads cached splits, rebalances (SMOTE + downsampling), trains the XGBoost pipeline, writes metrics/plots, logs to MLflow, and auto-registers the model. | `models/baseline_model.joblib`, `reports/metrics.json`, `reports/evaluation/`, MLflow run & registry version |

You can invoke any stage independently (e.g., `dvc repro split_data`) for debugging or lightweight experimentation.

---

## Data Contracts & Validation

- **Pandera schemas** (`src/creditrisk/validation/contracts.py`) cover every raw table plus the engineered feature store and persisted splits. They enforce ID integrity, allowable sentinel values (`DAYS_EMPLOYED`), numeric constraints, and binary targets.
- **ValidationRunner** (`src/creditrisk/validation/runner.py`) wires those contracts into each pipeline stage. It also checks for duplicate entity IDs, train/test leakage, NaNs/inf in feature matrices, and missing engineered columns.
- **Configurable enforcement** lives under the `validation` section of `configs/baseline.yaml`. Toggle `enforce_raw_contracts`, `enforce_feature_store_contract`, `enforce_split_contracts`, or `enforce_model_io_contracts` to relax checks in ad-hoc environments without touching the code.
- **Outputs fail fast**: any schema drift, missing field, or leakage raises a descriptive exception with sample failure rows so you can troubleshoot before a model trains on bad data.

---

## Model Registry & Promotion

- The training stage logs each run to MLflow and, when `registry.enabled: true`, automatically registers the serialized pipeline as `registry.model_name` (default `CreditRiskBaseline`).
- `registry.promote_on_metric` and `registry.promote_min_value` gate automatic staging. Example: with `roc_auc` and `0.78`, only runs at or above 0.78 move into the `Staging` stage; others remain unpromoted but still versioned.
- The registry helper lives in `src/creditrisk/mlops/registry.py` and handles:
  - Creating the registered model if missing.
  - Registering new versions from `runs:/<run_id>/model`.
  - Transitioning stages and optionally archiving the previous occupant.
- Manual promotions use `python -m creditrisk.pipelines.promote_model --version <n> --stage Production --archive-existing`. Because stage transitions are config-driven, CI/CD workflows can call the same CLI or import the helper.

---

## Repo Layout

```
configs/                # YAML configs (paths, training, validation, registry)
data/                   # Raw/interim/processed storage tracked with DVC
docs/                   # Architecture notes, proposal, operating guides
notebooks/              # Research + EDA history
reports/                # Metrics JSON + evaluation plots (DVC-tracked)
src/creditrisk/         # Python package
  ├── data/             # Dataset loaders & splits
  ├── features/         # DuckDB feature store + preprocessing utilities
  ├── models/           # Model factories, balancing, evaluation helpers
  ├── pipelines/        # CLI entry points (build feature store, split, train, promote, inference, FastAPI)
  ├── utils/            # Evaluation artifact writers, misc helpers
  └── validation/       # Pandera contracts + runner
dvc.yaml                # Multi-stage pipeline definition
requirements.txt        # Environment spec (DVC, Pandera, PyArrow, MLflow, etc.)
```

---

## Tooling Highlights

- **DVC**: deterministic DAGs (`dvc repro`), cached feature-store/split artifacts, and storage for large CSVs.
- **Pandera**: enforce raw-feed and feature-store contracts so schema or missingness drift fails fast.
- **PyArrow Parquet**: lightweight, columnar storage for feature store and split artifacts.
- **MLflow**: experiment tracking, artifact logging, and registry-backed promotion; optionally point `tracking_uri` to a managed server.
- **Evaluation bundle**: `creditrisk.utils.evaluation` writes ROC, PR, calibration, and confusion-matrix artifacts under `reports/evaluation/` for downstream QA or reporting.
- **Deployment hooks**: batch CLI + FastAPI (`creditrisk.serve.api`) reuse the saved sklearn pipeline and decision threshold; a promotion CLI bridges registry stages into deployment workflows.
- **Tests**: run `pytest tests` before merge to ensure helper modules and APIs stay healthy.

## CI/CD

- `.github/workflows/ci.yaml` runs on every PR/push. It installs dependencies, runs `ruff` + `bandit`, compiles the code, executes `pytest`, and validates the DVC graph via `dvc repro --dry-run train_baseline`.
- `.github/workflows/cd.yaml` runs on `main`. It pulls datasets via DVC, executes `dvc repro train_baseline`, uploads the model, evaluation bundle, lineage + ingestion reports, and automatically promotes the latest MLflow version to Production when the validation summary passes. Configure GitHub secrets (`MLFLOW_TRACKING_URI`, `DVC_REMOTE_URL`, etc.) for remote services.
- `docs/ci_cd.md` details the automation strategy, required secrets, and future enhancements (e.g., registry promotions).

## Containers

- `Dockerfile.api` builds a uvicorn-powered FastAPI image: `docker build -f Dockerfile.api -t creditrisk-api .` then `docker run -p 8080:8080 creditrisk-api`.
- `Dockerfile.batch` packages the batch scorer (`creditrisk.pipelines.batch_predict`): `docker build -f Dockerfile.batch -t creditrisk-batch .` and pass CLI args at runtime.
- `.dockerignore` keeps datasets, reports, and caches out of the image layers for faster reproducible builds.

---

## Next Steps

1. **CI** – wire pre-commit, linting, `pytest`, and `dvc repro --dry` into GitHub Actions (or similar) so every PR validates pipeline + contracts.
2. **CD** – script container builds (batch + FastAPI) and automate registry promotions into whatever serving platform you use (SageMaker, Vertex, AKS, etc.).
3. **Monitoring** – extend the validation layer into serving (batch + API) and add drift reports (Evidently/WhyLabs) so production data is continuously checked against training stats.
4. **Data freshness** – schedule the DVC stages (cron, Airflow, Dagster) and surface metadata like “data currency” to stakeholders.
5. **Governance** – expand MLflow tags (`tracking.tags`) with risk-specific metadata (PD bands, underwriting notes) to make promotions audit-ready.

---

Questions? See `docs/architecture.md` for deeper architectural decisions or reach out via the project Slack channel. Any run issues can usually be diagnosed by checking the validation failures, DVC logs, or the MLflow UI.
