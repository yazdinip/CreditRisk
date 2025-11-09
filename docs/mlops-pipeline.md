# MLOps Pipeline Blueprint

## Stages

| Stage | Status | Owner | Tooling | Description |
|-------|--------|-------|---------|-------------|
| `data_ingest` | ✅ | Data & Monitoring | DVC + Pandera + ingestion CLI | `python -m creditrisk.pipelines.ingest_data` validates/copies each Kaggle CSV and writes checksum metadata to `reports/ingestion_summary.json` so every run knows exactly which raw snapshot it used. |
| `feature_build` | ✅ | Modeling | DuckDB via `creditrisk.features.feature_store` | Recreates the notebook feature store (applications + bureau/bureau_balance/previous + installments + credit_card + POS cash) and emits the curated column list tracked by DVC. |
| `train_baseline` | ✅ | Modeling | `dvc repro train_baseline` / `python -m creditrisk.pipelines.train_baseline` | Trains the XGBoost pipeline, logs metrics + plots, writes lineage + registry metadata, and persists the serialized model for downstream jobs. |
| `evaluate` | ✅ (plots) / 🔜 (comparisons) | QA | `creditrisk.utils.evaluation`, Evidently (planned) | Saves ROC/PR/calibration/confusion artifacts to `reports/evaluation/`; next iteration will add historical comparisons/drift checks. |
| `batch_infer` | ✅ | Modeling / Ops | `python -m creditrisk.pipelines.batch_predict` | Scores CSVs with configurable thresholds, emitting structured JSON logs (request IDs, entity counts, duration) for observability. |
| `serve_online` | ✅ (Dockerized) / 🔜 (prod) | DevOps | FastAPI + Uvicorn + Dockerfile.api | FastAPI service exposes `/predict` + `/health`, wrapped in structured logging middleware; container image is ready for AKS/ECS/SageMaker deployment. |
| `deploy` | 🔜 | DevOps | Docker, MLflow Registry, GitHub Actions | CD runs the full DVC pipeline, uploads artifacts, and executes the auto-promotion CLI when validations pass; next step is wiring the container push + environment rollout. |
| `monitor` | TODO | Monitoring | Evidently, CloudWatch/Grafana | Scheduled jobs scoring fresh production data, drift/latency checks, retrain triggers. |

## Orchestration

- **Local**: use DVC (`dvc repro`) to execute DAGs with reproducible params/inputs; stages reuse cached artifacts unless upstream dependencies change.
- **Remote**: GitHub Actions CI/CD shells out to the same entry points. CD now auto-promotes MLflow versions when `reports/post_training_validation.json` reports `status: passed`.

## Configuration Strategy

- All knobs live in YAML under `configs/`; DVC `params` keeps track of values that should trigger re-runs.
- Secrets (MLflow tokens, DVC remotes, etc.) remain outside of version control—inject them through CI/CD secret stores or environment variables.

## Experiment Tracking & Registry

- Training logs metrics/artifacts to MLflow (`creditrisk_pd` experiment) with `project` + `stage` tags; evaluation plots, ingestion summaries, and lineage reports land under `reports/`.
- The training stage writes `reports/registry_promotion.json` containing run id, model version, metrics, and desired stage. CD consumes that report via `python -m creditrisk.pipelines.auto_promote --stage Production` to advance versions in the `CreditRiskPD` registry as soon as governance gates pass.

## Data, Model & Observability

1. **Raw data** – tracked with DVC remote (S3/Azure/GCS) plus checksum metadata in `reports/ingestion_summary.json`.
2. **Feature store** – deterministically regenerated; lineage recorded in `reports/data_lineage.json`.
3. **Model runs** – MLflow records git SHA + parameters + artifacts; registry promotion report links run id to the deployed stage for audits.
4. **Serving telemetry** – batch CLI and FastAPI service emit JSON logs with `request_id`, `entity_count`, `duration_ms`, and status codes so Splunk/CloudWatch dashboards and alerts have consistent signals.

## Next Iterations

1. Integrate Evidently/GX drift reports into the `evaluate` stage and surface them in CI/CD.
2. Wire the Docker images into the CD pipeline to push to a registry and roll out to staging/production clusters.
3. Layer monitoring jobs that score fresh production data, compare distributions, and trigger retraining when drift or performance decay crosses thresholds.
