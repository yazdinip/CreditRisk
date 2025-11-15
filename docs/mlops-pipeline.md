# MLOps Pipeline Blueprint

## Stages

| Stage | Status | Owner | Tooling | Description |
|-------|--------|-------|---------|-------------|
| `data_ingest` | done | Data & Monitoring | DVC + Pandera + ingestion CLI | `python -m creditrisk.pipelines.ingest_data` drives Kaggle/S3/Azure/DVC connectors, enforces MD5 checksums/decompression, and writes `reports/ingestion_summary.json` so every run knows the exact bronze snapshot. |
| `feature_build` | done | Modeling | DuckDB via `creditrisk.features.feature_store` | Replays the notebook SQL to merge bureau/balance/previous/installments/credit_card/POS cash tables and emits the curated feature store tracked by DVC. |
| `fit_model` | done | Modeling | `dvc repro fit_model` / `python -m creditrisk.pipelines.train_creditrisk_pd --skip-artifacts` | Trains the XGBoost pipeline, logs to MLflow, registers a candidate version, and persists the serialized model. |
| `evaluate_train` | done | QA | `python -m creditrisk.testing.test_dataset --split train` | Scores the cached train split, refreshes `reports/metrics.json`, and regenerates `reports/evaluation/`. |
| `evaluate_test` | done | QA | `python -m creditrisk.testing.test_dataset --split test` | Saves ROC/PR/calibration/confusion artifacts plus `reports/test_metrics.json`/`reports/test_predictions.parquet` for the held-out split. |
| `batch_infer` | done | Modeling / Ops | `python -m creditrisk.pipelines.batch_predict` | Scores CSVs with configurable thresholds, emitting structured JSON logs (request IDs, entity counts, duration). |
| `serve_online` | done (Dockerized) / planned (prod) | DevOps | FastAPI + Uvicorn + Dockerfile.api | FastAPI exposes `/predict` + `/health` behind structured logging middleware; CD can roll the image out to ECS automatically. |
| `validate_model` | done | QA / Risk | `python -m creditrisk.testing.post_training` | Applies governance checks (thresholds, artifact integrity, lineage/MLflow alignment) before promotion. |
| `register_model` | done | DevOps | `python -m creditrisk.pipelines.auto_promote` | Consumes the validation + candidate reports and transitions the MLflow version, logging `reports/registry_transition.json`. |
| `monitor` | done (train/test & production) | Monitoring | Evidently, CloudWatch, Grafana | `python -m creditrisk.monitoring.drift` compares cached train vs. test splits while `python -m creditrisk.monitoring.production` ingests fresh production pulls, writes `production_drift_report.{json,html}`, pushes metrics to CloudWatch, and (optionally) triggers a retrain when drift persists. |
| `canary_validation` | done (pre-deploy) | DevOps | `python -m creditrisk.pipelines.canary_validation` | Compares the candidate model against the currently deployed artifact on a reference dataset; blocks rollout when approval-rate deltas exceed tolerance. |

## Orchestration

- **Local**: use DVC (`dvc repro`) to execute DAGs with reproducible params/inputs; stages reuse cached artifacts unless upstream dependencies change.
- **Remote CI/CD**: GitHub Actions shells out to the same entry points. `ci.yaml` runs lint/tests + `dvc repro --dry-run validate_model`, and `cd.yaml` rebuilds the DAG, generates drift/freshness/canary summaries, auto-promotes MLflow versions, and publishes Docker images.
- **Lightweight scheduling**: `.github/workflows/nightly.yaml` runs on cron or manual dispatch, forces `dvc repro --force validate_model` + `monitor_drift`, calls `python -m creditrisk.utils.data_freshness --fail-on-stale`, executes `python -m creditrisk.monitoring.production`, and records retrain triggers so you get continuous freshness/drift coverage without Airflow/Dagster.

## Configuration Strategy

- All knobs live in YAML under `configs/`; DVC `params` keeps track of values that should trigger re-runs.
- Secrets (MLflow tokens, DVC remotes, etc.) stay out of VCS—inject them through GitHub Actions secrets or local environment variables.

## Experiment Tracking & Registry

- Training logs metrics/artifacts to MLflow (`creditrisk_pd` experiment) with `project` + `stage` tags; evaluation plots, ingestion summaries, data freshness, and lineage reports land under `reports/`.
- The training stage writes a candidate report containing run id, model version, metrics, and desired stage. DVC’s `register_model` stage consumes that via `python -m creditrisk.pipelines.auto_promote --stage Production` to advance versions in the `CreditRiskPD` registry once governance gates pass.

## Data, Model & Observability

1. **Raw data** – tracked with DVC remotes (S3/Azure/GCS) plus checksum metadata in `reports/ingestion_summary.json`.
2. **Feature store** – deterministically regenerated; lineage recorded in `reports/data_lineage.json`.
3. **Monitoring** – Evidently drift outputs (`reports/drift_report.json/html`), production drift reports, freshness summaries (`reports/data_freshness.json`), and structured serving logs feed the observability layer and can be streamed straight into dashboards/alerts today.
4. **Model runs** – MLflow records git SHA + parameters + artifacts; registry promotion report links run id to the deployed stage for audits.
5. **Serving telemetry** – batch CLI and FastAPI service emit JSON logs with `request_id`, `entity_count`, `duration_ms`, and status codes so Splunk/CloudWatch dashboards and alerts have consistent signals.

## Operational Enhancements

- Slack/webhook notifications and Grafana/Looker dashboards slot directly on top of the emitted JSON metrics whenever stakeholders need richer observability.
