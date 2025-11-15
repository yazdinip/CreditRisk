# Current Repo State

## Production Snapshot

- **End-to-end pipeline** – `dvc.yaml` captures every stage (ingest → feature store → split → train → test → validate → monitor). `dvc repro validate_model` now reproduces the exact artefacts that ship to production, including drift dashboards and lineage JSON.
- **Feature parity with research** – `src/creditrisk/features/feature_store.py` replays the vetted DuckDB SQL so the engineered feature set matches the Kaggle notebook the credit team signed off on. Pandera contracts guard each table so schema drift is caught immediately.
- **Governed experimentation** – MLflow logs parameters/metrics/tags for every training run, persists the sklearn pipeline, and registers the version automatically when metrics clear the configured threshold. `reports/registry_promotion.json` drives GitHub Actions promotions so releases are auditable.
- **Deployment-ready services** – `creditrisk.serve.api` exposes `/health`, `/metadata`, `/schema`, `/validate`, `/predict`, and `/metrics` endpoints. Requests pass through the same Pandera/ValidationRunner checks as training, structured logs include correlation IDs, and inference metrics are written back to MLflow.
- **Monitoring hooks** – Evidently drift reports (`reports/drift_report.json/html`), production drift reports (`reports/production_drift_report.json/html`), CloudWatch metrics, and `reports/data_freshness.json` provide health signals. A retrain trigger file (`reports/retrain_trigger.json`) records when automation kicked off `dvc repro validate_model` due to persistent drift.

## DVC Integration

- `dvc.yaml` defines each stage with explicit `deps` and `outs`, so cache invalidation is deterministic. For example, `build_feature_store` depends on every raw CSV plus the feature scripts, while `fit_model` consumes the processed splits and saved feature metadata and `evaluate_{train,test}` reuse the cached splits + pipeline without re-training.
- `dvc.lock` captures the exact git hashes + data checksums used for every successful run. Nightly and CD workflows call `dvc pull` to hydrate caches and `dvc repro` to regenerate artefacts before promotion or drift evaluation.
- Remote storage can be injected via `DVC_REMOTE_URL`/`DVC_REMOTE_CREDENTIALS`, allowing runners to download datasets without embedding secrets in the repo.

## MLflow Integration

- `creditrisk.pipelines.train_creditrisk_pd` logs hyperparameters, metrics (ROC-AUC, KS, precision/recall, etc.), tags, and the serialized pipeline under `runs:/<run_id>/model`, while the evaluation stages persist the governance artifacts (`reports/metrics.json`, `reports/test_metrics.json`, `reports/evaluation/**`).
- Registry automation (`src/creditrisk/mlops/registry.py`) stages new versions and archives old Production builds when `registry.promote_on_metric` thresholds are met. CD calls `python -m creditrisk.pipelines.auto_promote` so humans don’t have to run manual CLI commands.
- Post-training validation (`src/creditrisk/testing/post_training.py`) compares `reports/test_metrics.json` with the linked MLflow run to ensure the artefacts match what the registry recorded.
- FastAPI inference sessions open nested MLflow runs (with request_id/entity_count tags) so serving behaviour is observable alongside training history.

## Inference, Testing & Deployment

- `src/creditrisk/pipelines/batch_predict.py` offers a CLI to score CSVs or Parquet files, emitting `prediction` + `probability` columns and structured logs. It is packaged inside `Dockerfile.batch` for cron/Glue/Airflow style workloads.
- `src/creditrisk/serve/api.py` is containerised via `Dockerfile.api`, deployed by GitHub Actions to ECS, and smoke-tested automatically (`/health` + `/predict`) before traffic is shifted. Rollbacks happen automatically if the smoke test fails.
- `tests/` covers evaluation utilities, the FastAPI layer, and the production drift monitor so contracts don’t regress between releases.

## Automation & Monitoring

- **CI (`ci.yaml`)** – lint/test/compile pipeline plus `dvc repro --dry-run validate_model` on every PR/push.
- **CD (`cd.yaml`)** – full pipeline run, drift generation, MLflow promotion, Docker builds, optional ECS deploy + rollback, data freshness gating, and artefact uploading on `main` pushes.
- **Nightly (`nightly.yaml`)** – scheduled/dispatch workflow that forces the DAG, runs train/test + production drift monitors, publishes CloudWatch metrics, updates `reports/data_freshness.json`, and records `reports/retrain_trigger.json`.
- **Deployment manifest** – final DVC stage (`deploy_manifest`) writes `reports/deploy_manifest.json`, consolidating the trained model path, Dockerfiles, CI/CD workflow, and registry metadata so release engineers have a single JSON manifest to feed ECS, batch, or on-prem schedulers.

## Business & Monitoring Alignment

- The automation meets the proposal’s goals: reproducible nightly runs (>95 % success), MLflow-governed promotions, and deployment-ready artefacts for both batch and API surfaces.
- Drift and freshness reports already provide the “data currency” visibility stakeholders asked for and can be piped into Slack/dashboards with the existing JSON payloads.
- Governance artefacts (ingestion summary, lineage, validation summary, registry promotion report, retrain trigger report) live under `reports/` so auditors can trace every scoring decision back through data snapshots, configs, and code.
