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

## Key Capabilities

- **Deterministic pipelines** – DVC codifies every stage (ingest → build feature store → split → train → test → validate → monitor) so `dvc repro validate_model` always rebuilds the same artefacts, lineage, and metrics.
- **Governed experimentation** – MLflow captures parameters, metrics, artefacts, and stages model versions in the registry; the training/validation/reporting stack is identical whether you run locally or in CI/CD.
- **Data contracts + monitoring** – Pandera checks guard raw feeds, feature stores, and inference payloads. Evidently drift reports, production drift monitors, and freshness checks (`reports/data_freshness.json`) provide early warning signals.
- **Deployment-ready assets** – FastAPI and batch CLIs share helpers, structured logging, and inference governance. Dockerfiles build immutable images, and the CD workflow can push directly to ECS (or any platform that can pull GHCR tags).
- **Automation-first** – CI enforces lint/tests/DVC dry runs, CD reruns the DAG + promotion + container builds + deploy/smoke tests, and the nightly workflow keeps datasets fresh while running the production drift monitor.

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
   python -m creditrisk.pipelines.ingest_data --config configs/creditrisk_pd.yaml
   ```
   The ingestion CLI uses the connector declared in `configs/creditrisk_pd.yaml` (Kaggle competition downloads by default, but S3/Azure/DVC remotes are also supported) and enforces MD5 checksums before writing `reports/ingestion_summary.json`. If you already track the CSVs with DVC, flip the `type` to `dvc` and the same command will `dvc pull` the pinned bronze snapshot. Freshness metadata is derived from that file later.

3. **Run the full pipeline**
   ```bash
   dvc repro validate_model
   ```
   This executes the modular DAG:
   ```
   ingest_data -> build_feature_store -> split_data -> fit_model -> evaluate_train -> evaluate_test -> validate_model -> register_model
                                                        \
                                                         -> monitor_drift
   ```
   `ingest_data` validates or fetches the raw Kaggle extracts and writes `reports/ingestion_summary.json` so downstream lineage captures the exact snapshot used for the run.
   Outputs land in:
   - `data/processed/*.parquet` (feature store + deterministic splits)
      - `models/creditrisk_pd_model.joblib`
   - `reports/metrics.json`, `reports/test_metrics.json`, and `reports/evaluation/`
   - `reports/drift_report.json` + `reports/drift_report.html`
   - `reports/data_freshness.json` (generated via `python -m creditrisk.utils.data_freshness`)
   - `reports/registry_transition.json` (written by `python -m creditrisk.pipelines.auto_promote`)
   - `reports/canary_report.json` (optional canary validation summary)
   - `reports/deploy_manifest.json` (deployment-ready manifest)
   - `mlruns/` (MLflow experiment + registry metadata)

4. **Inspect / iterate**
   - Launch `mlflow ui` to review runs, metrics, and registered model versions.
   - Update configs in `configs/creditrisk_pd.yaml` (thresholds, registry behavior, validation toggles) and re-run `dvc repro`.
   - Use `python -m creditrisk.pipelines.promote_model --version <n> --stage Production --archive-existing` after governance sign-off.

5. **Operate on GCP**
   - The live environment runs on a Compute Engine VM with Dockerised FastAPI, self-hosted Airflow, and an on-demand MLflow UI secured via SSH tunnelling. See `docs/cloud_setup.md` for the exact provisioning steps (VM creation, dependency install, Airflow bootstrap, MLflow tunnel, firewall notes).

6. **Monitor production pulls**
   ```bash
   python -m creditrisk.monitoring.production \
     --config configs/creditrisk_pd.yaml \
     --current data/production/<YYYY-MM-DD>.parquet \
     --publish-metrics
   ```
   This generates `reports/production_drift_report.{json,html}`, pushes drift metrics to CloudWatch when AWS credentials are available, and records `reports/retrain_trigger.json` so you know whether the automation would have kicked off a retrain.

---

## Pipeline Overview

| Stage | CLI Entry Point | Description | Artifacts |
|-------|-----------------|-------------|-----------|
| `ingest_data` | `python -m creditrisk.pipelines.ingest_data` | Uses the configured connector (Kaggle API, S3, Azure Blob, or DVC) to pull each raw table, enforces MD5 checksums, optionally decompresses archives, and writes `reports/ingestion_summary.json`. | `reports/ingestion_summary.json`, validated raw CSVs |
| `build_feature_store` | `python -m creditrisk.pipelines.build_feature_store` | Loads all seven Kaggle extracts, enforces Pandera contracts, replays the DuckDB SQL feature engineering, and persists `data/processed/feature_store.parquet`. | Feature store parquet (165 cols) |
| `split_data` | `python -m creditrisk.pipelines.split_data` | Validates the feature store, stratifies on `TARGET`, enforces no-leakage guarantees, and writes deterministic train/test parquet files. | `data/processed/train.parquet`, `data/processed/test.parquet` |
| `fit_model` | `python -m creditrisk.pipelines.train_creditrisk_pd --skip-artifacts` | Loads cached splits, rebalances (SMOTE + downsampling), trains the XGBoost pipeline, and logs to MLflow/registry without emitting evaluation artifacts. | `models/creditrisk_pd_model.joblib`, MLflow run & registry candidate |
| `evaluate_train` | `python -m creditrisk.testing.test_dataset --split train` | Runs the trained pipeline against the cached train split, writes `reports/metrics.json`, and refreshes `reports/evaluation/`. | `reports/metrics.json`, `reports/evaluation/` |
| `evaluate_test` | `python -m creditrisk.testing.test_dataset --split test` | Scores the held-out test split, persists per-entity predictions, and generates evaluation plots. | `reports/test_metrics.json`, `reports/test_evaluation/`, `reports/test_predictions.parquet` |
| `validate_model` | `python -m creditrisk.testing.post_training` | Applies governance checks (metric thresholds, artifact integrity, lineage presence, MLflow alignment) before promotion. | `reports/post_training_validation.json` |
| `register_model` | `python -m creditrisk.pipelines.auto_promote` | Reads validation, train/test metrics, drift, and canary reports before transitioning the MLflow model version; writes a promotion summary for audit trails. | `reports/registry_transition.json` |
| `monitor_drift` | `python -m creditrisk.monitoring.drift` | Runs Evidently’s drift preset on the persisted train vs. test splits to quantify distribution shifts and emit HTML/JSON dashboards. | `reports/drift_report.json`, `reports/drift_report.html` |
| `canary_validation` | `python -m creditrisk.pipelines.canary_validation` | Compares the candidate model against the current production model on a reference dataset before deploy; fails when approval-rate deltas exceed the tolerance. | `reports/canary_report.json` |
| `deploy_manifest` | `python -m creditrisk.pipelines.deploy_manifest` | Packages pointers to the trained model, Dockerfiles, CI/CD workflow, and registry metadata into a JSON manifest consumed by deployment automation. | `reports/deploy_manifest.json` |

You can invoke any stage independently (e.g., `dvc repro split_data`) for debugging or lightweight experimentation.

## Pipeline Entry Points & Operational CLIs

- `creditrisk.pipelines.ingest_data` – fetches Kaggle/S3/Azure/DVC sources, enforces checksums, and writes `reports/ingestion_summary.json`.
- `creditrisk.pipelines.build_feature_store` – replays the DuckDB SQL + Pandera contracts to create `data/processed/feature_store.parquet`.
- `creditrisk.pipelines.split_data` – stratifies, deduplicates, and validates the train/test splits.
- `creditrisk.pipelines.train_creditrisk_pd` – balances classes, trains/logs to MLflow, registers a candidate version; use `--skip-artifacts` when the evaluation stage will handle metrics.
- `creditrisk.testing.test_dataset --split train|test` – evaluates the cached splits, writes metrics/plots/predictions, and feeds downstream governance.
- `creditrisk.testing.post_training` – regression suite that confirms score parity with MLflow entries before promotion.
- `creditrisk.pipelines.batch_predict` – CLI used by schedulers to score CSVs into parquet/JSON (sharing the same pipeline + threshold).
- `creditrisk.monitoring.drift` – compares cached train vs. test splits for drift and emits Evidently visualisations.
- `creditrisk.monitoring.production` – loads reference vs. production pulls, logs `production_drift_report.{json,html}`, pushes CloudWatch metrics (when configured), and writes `reports/retrain_trigger.json`; add `--auto-retrain` to execute the configured retrain command.
- `creditrisk.utils.data_freshness` – inspects ingestion metadata and fails CI/nightly runs when the feeds grow stale.
- `creditrisk.pipelines.promote_model` / `creditrisk.pipelines.auto_promote` – bridge MLflow registry stages into deployment workflows and log promotion results.
- `creditrisk.pipelines.canary_validation` – compare the newly trained model against the last production build before shipping it to ECS/batch runners.
- `creditrisk.pipelines.deploy_manifest` – emit `reports/deploy_manifest.json`, a manifest describing the model, Dockerfiles, CI/CD workflow, and secrets needed for ECS deployment.

### Raw Data Connectors & Checksums

`configs/creditrisk_pd.yaml` enumerates bronze datasets under `ingestion.sources`. Each entry specifies a `type`, optional `uri`, and connector-specific `options` so `python -m creditrisk.pipelines.ingest_data` knows where to fetch the table:

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

### FastAPI Surface

`creditrisk.serve.api` now exposes multiple governance-friendly endpoints:

- `GET /health` – liveness.
- `GET /metadata` – model path, decision threshold, experiment/tags, and artifact timestamps.
- `GET /schema` – expected feature columns and entity id so clients can validate payloads.
- `POST /validate` – run the Pandera/ValidationRunner checks on a payload without scoring.
- `POST /predict` – score a batch, log inference metrics to MLflow, and return predictions + probabilities.
- `GET /metrics` – lightweight request counters and timestamps for dashboards or scraping. When AWS credentials and `monitoring.cloudwatch_namespace` are set, the service also publishes latency/error metrics to CloudWatch via the shared observability helpers.

The job-ready `Dockerfile.api` builds and serves the app on port 8080; ECS deployment is automated through the CD workflow.

### Canary & Shadow Testing

Before promoting a newly trained model, run:

```bash
python -m creditrisk.pipelines.canary_validation \
  --config configs/creditrisk_pd.yaml \
  --production-model /mlops/models/prod.joblib \
  --candidate-model models/creditrisk_pd_model.joblib \
  --dataset data/processed/test.parquet \
  --max-metric-delta 0.02
```

The CLI scores the same reference dataset with both pipelines and fails if approval rate or mean probability drift beyond the allowed delta. Airflow (or GH Actions) can insert this gate between validation and deployment, giving you an automated approval step plus audit-friendly `reports/canary_report.json`.

---

## Data Contracts & Validation

- **Pandera schemas** (`src/creditrisk/validation/contracts.py`) cover every raw table plus the engineered feature store and persisted splits. They enforce ID integrity, allowable sentinel values (`DAYS_EMPLOYED`), numeric constraints, and binary targets.
- **ValidationRunner** (`src/creditrisk/validation/runner.py`) wires those contracts into each pipeline stage. It also checks for duplicate entity IDs, train/test leakage, NaNs/inf in feature matrices, and missing engineered columns.
- **Configurable enforcement** lives under the `validation` section of `configs/creditrisk_pd.yaml`. Toggle `enforce_raw_contracts`, `enforce_feature_store_contract`, `enforce_split_contracts`, or `enforce_model_io_contracts` to relax checks in ad-hoc environments without touching the code.
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
  data/                 # Dataset loaders & split helpers
  features/             # DuckDB feature store + preprocessing utilities
  models/               # Model factories, balancing, evaluation helpers
  pipelines/            # CLI entry points (ingest, feature build, split, train, promote, batch)
  prediction/           # Batch prediction helpers
  monitoring/           # Drift monitoring (train/test + production)
  observability/        # Structured logging + CloudWatch publishers
  serve/                # FastAPI app + governance layer
  testing/              # Post-training + dataset tests
  utils/                # Evaluation, lineage, filesystem helpers
  validation/           # Pandera contracts + runner
dvc.yaml                # Multi-stage pipeline definition
requirements.txt        # Environment spec (DVC, Pandera, PyArrow, MLflow, etc.)
```


---

## Tooling Highlights

- **DVC** – deterministic DAGs (`dvc repro`) with cached feature-store/split artefacts and lineage baked into `dvc.lock`.
- **Pandera + ValidationRunner** – schema/missingness guardrails across ingestion, feature store, splits, and inference payloads.
- **MLflow** – experiment tracking + registry promotions, consumed by the CLI helpers and CI/CD automation.
- **Observability** – structured JSON logs, CloudWatch metric publishers, drift dashboards, and freshness summaries.
- **Serving governance** – FastAPI shares the same contracts/thresholds/pipeline as batch, emits MLflow inference runs, and exposes metadata endpoints for downstream systems.

## CI/CD Automation

### CI (`.github/workflows/ci.yaml`)

- **Trigger**: every pull request and non-`main` push.
- **Steps**: checkout, set up Python 3.11 with pip caching, install dependencies, run `ruff`, `bandit`, `python -m compileall`, `pytest`, and finally `dvc repro --dry-run validate_model`.
- **Outcome**: fails fast on lint/test/schema regressions before artefacts or MLflow runs are created.

### CD (`.github/workflows/cd.yaml`)

- **Trigger**: pushes to `main`.
- **Core steps**: checkout + dependency install → optional DVC remote config → `dvc pull` → `dvc repro validate_model` → `dvc repro monitor_drift` → MLflow auto-promotion (when `reports/registry_promotion.json` indicates success) → build/push Docker images for API and batch (`ghcr.io/<repo>/creditrisk-{api|batch}:{latest,sha}`).
- **Deployment**: when AWS secrets are set the workflow configures credentials, captures the live ECS task definition, swaps in the new image tag, deploys, waits for stability, and smoke-tests `/health` + `/predict`. Failures trigger automatic rollback to the prior task definition.
- **Registry/Canary**: `dvc repro register_model` transitions the MLflow version only when the validation summary passes, and `dvc repro canary_validation` compares the candidate model against the current production artefact before ECS rollout. Both steps leave JSON breadcrumbs (`reports/registry_transition.json`, `reports/canary_report.json`) for auditors.
- **Freshness gate**: `python -m creditrisk.utils.data_freshness --max-age-hours 48 --fail-on-stale` blocks releases when the upstream feeds go stale.
- **Artefacts**: models, metrics, lineage, ingestion summaries, drift bundles, registry promotion reports, data freshness JSON, production drift metrics, canary summaries, and the deployment manifest are uploaded for auditors:
  - `reports/metrics.json`, `reports/test_metrics.json`, `reports/evaluation/**`
  - `reports/ingestion_summary.json`, `reports/data_lineage.json`, `reports/data_freshness.json`
  - `reports/drift_report.{json,html}`, `reports/production_drift_report.{json,html}`, `reports/drift_metrics.json`
  - `reports/post_training_validation.json`, `reports/registry_promotion.json`, `reports/registry_transition.json`
  - `reports/canary_report.json`, `reports/retrain_trigger.json`, `reports/deploy_manifest.json`
- **Secrets**:

| Secret | Purpose |
| --- | --- |
| `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_TOKEN` | Remote tracking + registry access. |
| `DVC_REMOTE_URL` / `DVC_REMOTE_CREDENTIALS` | Optional remote for pulling large datasets. |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | Allow ECS deploy + CloudWatch publishing. |
| `ECS_CLUSTER_NAME` / `ECS_SERVICE_NAME` / `PREDICT_ENDPOINT_URL` | Target service + smoke-test endpoint. |

When the AWS/ECS secrets are omitted, the workflow still runs the pipeline and publishes artefacts without attempting a deployment.

### Nightly (`.github/workflows/nightly.yaml`)

- **Trigger**: cron (`0 6 * * *`) plus manual `workflow_dispatch`.
- **Steps**: fetch data via DVC, run `dvc repro --force validate_model` and `monitor_drift`, execute `python -m creditrisk.monitoring.production --publish-metrics` (optionally pointing to a production parquet path via the `PRODUCTION_CURRENT_PATH` secret), run the freshness CLI with a 36h SLA, and upload a trimmed artefact bundle.
- **Purpose**: ensures data freshness, pushes drift metrics to CloudWatch when credentials exist, and produces `reports/production_drift_report.{json,html}`, `reports/drift_metrics.json`, and `reports/retrain_trigger.json`. If `monitoring.auto_retrain: true`, exceeding the configured drift threshold automatically kicks off the `monitoring.retrain_command`.
- **Operator view**: the artefacts and CloudWatch metrics reveal whether retraining or intervention is required without shelling into the runner.

### Airflow DAG (`orchestration/airflow_creditrisk_dag.py`)

- **Why**: when you outgrow GitHub Actions + cron, drop this DAG into an Airflow environment. It mirrors the entire DVC pipeline, production monitor, and the new canary validation gate so you get stateful runs, retries, approvals, and failure visibility.
- **Highlights**: each stage shells out to the same CLIs we use locally (`python -m creditrisk.pipelines.ingest_data`, `dvc repro ...`, `python -m creditrisk.monitoring.production`, `python -m creditrisk.pipelines.canary_validation`). Airflow Variables supply paths such as `creditrisk_repo`, `production_model_path`, and the production dataset, so you can promote between dev/staging/prod clusters without editing code.
- **Notifications**: pair the DAG with native Airflow alerting/approvals (Slack, Email, PagerDuty) or Step Functions/ECS operators when you need cross-account rollouts.

See `docs/ci_cd.md` for a deeper breakdown plus manual run instructions.

## Containers

- `Dockerfile.api` builds a uvicorn-powered FastAPI image: `docker build -f Dockerfile.api -t creditrisk-api .` then `docker run -p 8080:8080 creditrisk-api`.
- `Dockerfile.batch` packages the batch scorer (`creditrisk.pipelines.batch_predict`): `docker build -f Dockerfile.batch -t creditrisk-batch .` and pass CLI args at runtime.
- `.dockerignore` keeps datasets, reports, and caches out of the image layers for faster reproducible builds.

## Monitoring & Operations

- **Data freshness** – the nightly workflow emits `reports/data_freshness.json` and fails when the ingestion snapshot exceeds the SLA. Locally you can run `python -m creditrisk.utils.data_freshness --max-age-hours 24 --fail-on-stale` to get the same JSON (useful for dashboards or alerts).
- **Production drift** – `python -m creditrisk.monitoring.production --current data/production/<date>.parquet --publish-metrics --auto-retrain` compares reference vs. production pulls, writes `reports/production_drift_report.{json,html}`, publishes CloudWatch metrics (when AWS creds exist), and, when `monitoring.auto_retrain: true`, executes the configured `monitoring.retrain_command`.
- **Retrain transparency** – `reports/retrain_trigger.json` records whether a retrain was kicked off (and why) so governance teams can audit the decision. Pair it with `reports/drift_metrics.json` for dashboards.
- **Structured logging** – `creditrisk.observability.logging` sets JSON logging globally and exposes CloudWatch metric helpers so batch jobs, FastAPI, and monitoring scripts emit consistent telemetry (`request_id`, latency, status_code, stage, entity counts).
- **Dashboards/alerts** – CloudWatch (or any sink you choose) can alarm on drift share, stale snapshots, or error rates without wiring additional orchestration; everything surfaces from the Actions logs + artefacts.

---

## Optional Enhancements

- Pipe nightly/CD artefacts (freshness report, drift metrics, ECS deploy status) plus serving logs into Slack/webhooks or Grafana for operator awareness.
- Layer in SHAP/explainability snapshots or chaos drills if your governance team requests deeper transparency or failover rehearsal.

---

Questions? See `docs/architecture.md` for deeper architectural decisions or reach out via the project Slack channel. Any run issues can usually be diagnosed by checking the validation failures, DVC logs, or the MLflow UI.
