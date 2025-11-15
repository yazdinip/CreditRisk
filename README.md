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

2. **Fetch / validate the raw datasets**
   ```bash
   # ensure your Kaggle API token lives under %USERPROFILE%\.kaggle\kaggle.json (or ~/.kaggle/kaggle.json)
   python -m creditrisk.pipelines.ingest_data --config configs/baseline.yaml
   ```
   The ingestion CLI uses the connector declared in `configs/baseline.yaml` (Kaggle competition downloads by default, but S3/Azure/DVC remotes are also supported) and enforces MD5 checksums before writing `reports/ingestion_summary.json`. If you already track the CSVs with DVC, flip the `type` to `dvc` and the same command will `dvc pull` the pinned bronze snapshot.

3. **Run the full pipeline**
   ```bash
   dvc repro validate_model
   ```
   This executes the modular DAG:
   ```
   ingest_data -> build_feature_store -> split_data -> train_baseline -> test_model -> validate_model
                                                        \
                                                         -> monitor_drift
   ```
   `ingest_data` validates or fetches the raw Kaggle extracts and writes `reports/ingestion_summary.json` so downstream lineage captures the exact snapshot used for the run.
   Outputs land in:
   - `data/processed/*.parquet` (feature store + deterministic splits)
   - `models/baseline_model.joblib`
   - `reports/metrics.json`, `reports/test_metrics.json`, and `reports/evaluation/`
   - `reports/drift_report.json` + `reports/drift_report.html`
   - `reports/data_freshness.json` (generated via `python -m creditrisk.utils.data_freshness`)
   - `mlruns/` (MLflow experiment + registry metadata)

4. **Inspect / iterate**
   - Launch `mlflow ui` to review runs, metrics, and registered model versions.
   - Update configs in `configs/baseline.yaml` (thresholds, registry behavior, validation toggles) and re-run `dvc repro`.
   - Use `python -m creditrisk.pipelines.promote_model --version <n> --stage Production --archive-existing` after governance sign-off.

---

## Pipeline Overview

| Stage | CLI Entry Point | Description | Artifacts |
|-------|-----------------|-------------|-----------|
| `ingest_data` | `python -m creditrisk.pipelines.ingest_data` | Uses the configured connector (Kaggle API, S3, Azure Blob, or DVC) to pull each raw table, enforces MD5 checksums, optionally decompresses archives, and writes `reports/ingestion_summary.json`. | `reports/ingestion_summary.json`, validated raw CSVs |
| `build_feature_store` | `python -m creditrisk.pipelines.build_feature_store` | Loads all seven Kaggle extracts, enforces Pandera contracts, replays the DuckDB SQL feature engineering, and persists `data/processed/feature_store.parquet`. | Feature store parquet (165 cols) |
| `split_data` | `python -m creditrisk.pipelines.split_data` | Validates the feature store, stratifies on `TARGET`, enforces no-leakage guarantees, and writes deterministic train/test parquet files. | `data/processed/train.parquet`, `data/processed/test.parquet` |
| `train_baseline` | `python -m creditrisk.pipelines.train_baseline` | Loads cached splits, rebalances (SMOTE + downsampling), trains the XGBoost pipeline, writes metrics/plots, logs to MLflow, and auto-registers the model. | `models/baseline_model.joblib`, `reports/metrics.json`, `reports/evaluation/`, MLflow run & registry version |
| `test_model` | `python -m creditrisk.testing.test_dataset` | Scores the held-out test set, persists per-entity predictions, and generates evaluation plots. | `reports/test_metrics.json`, `reports/test_evaluation/`, `reports/test_predictions.parquet` |
| `validate_model` | `python -m creditrisk.testing.post_training` | Applies governance checks (metric thresholds, artifact integrity, lineage presence, MLflow alignment) before promotion. | `reports/post_training_validation.json` |
| `monitor_drift` | `python -m creditrisk.monitoring.drift` | Runs Evidently’s drift preset on the persisted train vs. test splits to quantify distribution shifts and emit HTML/JSON dashboards. | `reports/drift_report.json`, `reports/drift_report.html` |

You can invoke any stage independently (e.g., `dvc repro split_data`) for debugging or lightweight experimentation.

### Raw Data Connectors & Checksums

`configs/baseline.yaml` enumerates bronze datasets under `ingestion.sources`. Each entry specifies a `type`, optional `uri`, and connector-specific `options` so `python -m creditrisk.pipelines.ingest_data` knows where to fetch the table:

1. **`kaggle` / `kaggle_competition` / `kaggle_dataset`** – authenticate with the Kaggle API token in `%USERPROFILE%\.kaggle\kaggle.json` (or `~/.kaggle/kaggle.json`). Provide `options.competition` and `options.file` for competition downloads, or `options.dataset` for dataset sources.
2. **`s3` / `aws_s3`** – pull from `s3://bucket/key` using the default AWS credential chain (env vars, shared credentials, IAM). `options` accepts overrides like `bucket`, `key`, `profile`, `region`, `endpoint_url`, and `version_id`.
3. **`azure_blob` / `azure`** – download from Azure Blob Storage via `options.connection_string` or (`options.account_url` + `options.credential`/`sas_token`). Always specify `options.container` and `options.blob`.
4. **`dvc` / `dvc_remote`** – shell out to `dvc pull <target>` so nightly builds can hydrate the tracked snapshot without running ad-hoc scripts.

All connectors funnel through the same materialization layer: archives are optionally decompressed (`decompress: true`), MD5 checksums are enforced (`checksum: ...`), and the results are logged to `reports/ingestion_summary.json` along with resolved URIs and file sizes. Example:

```yaml
ingestion:
  sources:
    - name: application_train
      type: kaggle_competition
      output_path: data/raw/application_train.csv
      decompress: true
      skip_if_exists: true  # drop the CSV manually to bypass the Kaggle call
      checksum: 793a017f41fbac1dc28176b26dbab30e
      options:
        competition: home-credit-default-risk
        file: application_train.csv
```

Downstream lineage (`reports/data_lineage.json`) links back to the same snapshot so every run remains auditable.

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
- **Serving governance**: the FastAPI service runs the same Pandera/ValidationRunner contracts at inference time, rejects malformed payloads, and logs underwriting metadata + request-level metrics (risk bands, approval rate, entity counts) to MLflow for auditability.
- **Tests**: run `pytest tests` before merge to ensure helper modules and APIs stay healthy.

## CI/CD

- `.github/workflows/ci.yaml` runs on every PR/push. It installs dependencies, runs `ruff` + `bandit`, compiles the code, executes `pytest`, and validates the DVC graph via `dvc repro --dry-run validate_model`.
- `.github/workflows/cd.yaml` runs on `main`. It pulls datasets via DVC, executes `dvc repro validate_model` followed by `dvc repro monitor_drift`, uploads the expanded observability bundle (metrics, lineage, ingestion, drift), and automatically promotes the latest MLflow version to Production when the validation summary passes. The job also builds/pushes the batch + API Docker images to GHCR, updates the ECS service (using the task definition already running in your cluster), waits for stability, and smoke-tests `/health` + `/predict`. Failures trigger an automatic rollback to the previous task definition so a bad deploy never lingers.

  Required deployment secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECS_CLUSTER_NAME`, `ECS_SERVICE_NAME`, and `PREDICT_ENDPOINT_URL` (the load balancer URL that fronts the service). When these are omitted the workflow skips the ECS steps, so forks can still run the rest of the pipeline without cloud credentials.
- `.github/workflows/nightly.yaml` is a lightweight scheduler replacement. It runs every morning (UTC) and is manually dispatchable, forces `dvc repro --force validate_model` + `monitor_drift`, runs the production-focused monitor (`python -m creditrisk.monitoring.production`) against the latest scoring extracts, pushes drift metrics into CloudWatch (when AWS creds exist), and writes `reports/retrain_trigger.json`. If the drift share crosses the configured threshold the job invokes the configured retrain command so the DAG restarts automatically.

### Production Data Monitoring & Retrain Automation

Use the new CLI to evaluate fresh production pulls (e.g., daily scored applications), publish metrics for dashboards, and trigger retraining when drift persists:

```bash
python -m creditrisk.monitoring.production \
  --config configs/baseline.yaml \
  --current data/production/<yyyy-mm-dd>.parquet \
  --publish-metrics \
  --auto-retrain
```

- Outputs land in `reports/production_drift_report.{json,html}` plus `reports/drift_metrics.json` for Grafana.
- CloudWatch metrics (`CreditRisk/Monitoring`) receive the drift share + binary drift flag so alarms can page when SLAs are missed.
- `reports/retrain_trigger.json` records whether a retrain was launched (and why); when the flag is set, the configured command (default `dvc repro validate_model`) runs automatically.
- `.github/workflows/nightly.yaml` is a lightweight scheduler replacement. It runs every morning (UTC) and is manually dispatchable, forces `dvc repro --force validate_model` + `monitor_drift`, and generates `reports/data_freshness.json`. The job fails if the ingestion snapshot is older than the configured SLA (36h by default), which gives you paging-on-failure without a heavy orchestrator.
- `docs/ci_cd.md` details the automation strategy, required secrets, and future enhancements (e.g., registry promotions).

## Containers

- `Dockerfile.api` builds a uvicorn-powered FastAPI image: `docker build -f Dockerfile.api -t creditrisk-api .` then `docker run -p 8080:8080 creditrisk-api`.
- `Dockerfile.batch` packages the batch scorer (`creditrisk.pipelines.batch_predict`): `docker build -f Dockerfile.batch -t creditrisk-batch .` and pass CLI args at runtime.
- `.dockerignore` keeps datasets, reports, and caches out of the image layers for faster reproducible builds.

## Scheduling & Data Freshness

- The nightly GitHub Action covers cron-style orchestration: it forces the full DVC DAG daily, publishes artifacts, and updates `reports/data_freshness.json` so stakeholders know when the last ingestion happened.
- Need a local check? Run `python -m creditrisk.utils.data_freshness --config configs/baseline.yaml --max-age-hours 24 --fail-on-stale` (after `pip install -e .`). The CLI inspects `reports/ingestion_summary.json`, records the run timestamp, and flips `status` to `stale` when the snapshot is too old or missing.
- Because the workflow only relies on Actions + the existing DVC graph, you don’t need Airflow/Dagster unless you decide to scale beyond this repo.

---

## Next Steps

1. **CI** – wire pre-commit, linting, `pytest`, and `dvc repro --dry` into GitHub Actions (or similar) so every PR validates pipeline + contracts.
2. **CD** – script container builds (batch + FastAPI) and automate registry promotions into whatever serving platform you use (SageMaker, Vertex, AKS, etc.).
3. **Monitoring** – extend the validation layer into serving (batch + API) and add drift reports (Evidently/WhyLabs) so production data is continuously checked against training stats.
4. **Data freshness** – push the nightly Action’s `reports/data_freshness.json` into Slack/email or Opsgenie so stale snapshots alert humans automatically, and consider exposing the same JSON via FastAPI for dashboards.
5. **Governance** – expand MLflow tags (`tracking.tags`) with risk-specific metadata (PD bands, underwriting notes) to make promotions audit-ready.

---

Questions? See `docs/architecture.md` for deeper architectural decisions or reach out via the project Slack channel. Any run issues can usually be diagnosed by checking the validation failures, DVC logs, or the MLflow UI.
