# MLOps Pipeline Blueprint

## Stages

| Stage | Status | Owner | Tooling | Description |
|-------|--------|-------|---------|-------------|
| `data_ingest` | done | Data & Monitoring | DVC + Pandera + ingestion CLI | `python -m creditrisk.pipelines.ingest_data` drives Kaggle/S3/Azure/DVC connectors, enforces MD5 checksums/decompression, and writes `reports/ingestion_summary.json` so every run knows the exact bronze snapshot. |
| `feature_build` | done | Modeling | DuckDB via `creditrisk.features.feature_store` | Replays the notebook SQL to merge bureau/balance/previous/installments/credit_card/POS cash tables and emits the curated feature store tracked by DVC. |
| `train_baseline` | done | Modeling | `dvc repro train_baseline` / `python -m creditrisk.pipelines.train_baseline` | Trains the XGBoost pipeline, logs metrics/plots, writes lineage + registry metadata, and persists the serialized model. |
| `evaluate` | done (plots) / planned (comparisons) | QA | `creditrisk.utils.evaluation`, Evidently (planned) | Saves ROC/PR/calibration/confusion artifacts to `reports/evaluation/`; future work will add historical comparisons + drift checks. |
| `batch_infer` | done | Modeling / Ops | `python -m creditrisk.pipelines.batch_predict` | Scores CSVs with configurable thresholds, emitting structured JSON logs (request IDs, entity counts, duration). |
| `serve_online` | done (Dockerized) / planned (prod) | DevOps | FastAPI + Uvicorn + Dockerfile.api | FastAPI exposes `/predict` + `/health` behind structured logging middleware; container image is ready for AKS/ECS/SageMaker. |
| `deploy` | planned | DevOps | Docker, MLflow Registry, GitHub Actions | CD runs the full DVC pipeline, uploads artifacts, executes the auto-promotion CLI when validations pass, and builds both containers for downstream rollout. |
| `monitor` | done (drift) | Monitoring | Evidently, CloudWatch/Grafana | `python -m creditrisk.monitoring.drift` compares persisted train vs. test splits and emits HTML/JSON drift dashboards; production telemetry ingestion hooks are next. |

## Orchestration

- **Local**: use DVC (`dvc repro`) to execute DAGs with reproducible params/inputs; stages reuse cached artifacts unless upstream dependencies change.
- **Remote CI/CD**: GitHub Actions shells out to the same entry points. `ci.yaml` runs lint/tests + `dvc repro --dry-run validate_model`, and `cd.yaml` rebuilds the DAG, generates drift/freshness summaries, auto-promotes MLflow versions, and publishes Docker images.
- **Lightweight scheduling**: `.github/workflows/nightly.yaml` runs on cron or manual dispatch, forces `dvc repro --force validate_model` + `monitor_drift`, and calls `python -m creditrisk.utils.data_freshness --fail-on-stale` so stale Kaggle/DVC snapshots fail loudly without adding Airflow/Dagster.

## Configuration Strategy

- All knobs live in YAML under `configs/`; DVC `params` keeps track of values that should trigger re-runs.
- Secrets (MLflow tokens, DVC remotes, etc.) stay out of VCS—inject them through GitHub Actions secrets or local environment variables.

## Experiment Tracking & Registry

- Training logs metrics/artifacts to MLflow (`creditrisk_pd` experiment) with `project` + `stage` tags; evaluation plots, ingestion summaries, data freshness, and lineage reports land under `reports/`.
- The training stage writes `reports/registry_promotion.json` containing run id, model version, metrics, and desired stage. CD consumes that via `python -m creditrisk.pipelines.auto_promote --stage Production` to advance versions in the `CreditRiskPD` registry once governance gates pass.

## Data, Model & Observability

1. **Raw data** – tracked with DVC remotes (S3/Azure/GCS) plus checksum metadata in `reports/ingestion_summary.json`.
2. **Feature store** – deterministically regenerated; lineage recorded in `reports/data_lineage.json`.
3. **Monitoring** – Evidently drift outputs (`reports/drift_report.json/html`), freshness summaries (`reports/data_freshness.json`), and structured serving logs feed the observability layer; next iteration pipes those into dashboards/alerts.
4. **Model runs** – MLflow records git SHA + parameters + artifacts; registry promotion report links run id to the deployed stage for audits.
5. **Serving telemetry** – batch CLI and FastAPI service emit JSON logs with `request_id`, `entity_count`, `duration_ms`, and status codes so Splunk/CloudWatch dashboards and alerts have consistent signals.

## Next Iterations

1. Pipe Evidently drift metrics + serving logs into dashboards/alerting (CloudWatch/Grafana) and define retraining triggers.
2. Extend the CD job to roll the freshly published container images into staging/prod clusters with smoke tests.
3. Layer monitoring jobs that score fresh production data, compare distributions, and trigger retraining when drift or performance decay crosses thresholds.
