# Current Repo State

## Production Snapshot

- **Compute Engine footprint** – The entire stack now lives on a hardened e2-series VM in `us-central1`. The VM runs FastAPI (port 80), the self-hosted Airflow scheduler/webserver (port 8080 exposed via firewall), and an on-demand MLflow UI that stays private behind SSH tunnelling. Raw Kaggle exports are mirrored to `gs://creditriskbucket` so the VM and CI can `dvc pull` the canonical snapshot before every run.
- **End-to-end pipeline** – `dvc.yaml` continues to capture the ingest → feature store → split → train → evaluate → validate → monitor DAG. `dvc repro validate_model` rebuilds the exact artefacts we ship (models, drift dashboards, lineage JSON, promotion manifests).
- **Feature parity with research** – `src/creditrisk/features/feature_store.py` replays the curated DuckDB SQL notebook and applies Pandera contracts so schema drift is detected before features reach modelling stages.
- **Governed experimentation** – MLflow tracks parameters, metrics, tags, and registry state locally in `mlruns/`. The CD workflow promotes runs only when validation clears thresholds and writes breadcrumbs to `reports/registry_promotion.json` / `reports/registry_transition.json` for auditors.
- **Self-hosted Airflow** – `~/airflow-venv` runs Airflow 2.9. The DAG at `~/airflow/dags/airflow_creditrisk_dag.py` mirrors the DVC stages (ingest → build → split → train → evaluate_train → evaluate_test → validate → monitor → production_monitor/canary). Airflow Variables (`creditrisk_repo`, `production_dataset_path`, `production_model_path`, `canary_max_delta`) point to the repo and deployed artefacts.
- **Monitoring hooks** – Evidently drift outputs (`reports/drift_report.*`, `reports/production_drift_report.*`), CloudWatch metric pushes, `reports/data_freshness.json`, and `reports/retrain_trigger.json` give operators deterministic health signals across nightly, Airflow, and CD runs.
- **On-demand MLflow UI** – When model inspection is needed, `mlflow ui --host 127.0.0.1 --port 5000` runs on the VM and Cloud Shell (or a local laptop) creates an SSH tunnel (`gcloud compute ssh creditrisk-api --zone us-central1-f -- -4 -L 9000:localhost:5000`). Cloud Shell’s Web Preview targets port 9000 so the UI is reachable without opening the port publicly.

## DVC Integration

- `dvc.yaml` defines each stage with explicit `deps` and `outs`, so cache invalidation is deterministic. For example, `build_feature_store` depends on every raw CSV plus the feature scripts, while `fit_model` consumes the processed splits and saved metadata, and `evaluate_{train,test}` reuse the cached splits + pipeline.
- `dvc.lock` captures the exact git hashes + data checksums used for every successful run. Nightly/Airflow/CD workflows call `dvc pull` to hydrate caches and `dvc repro` to regenerate artefacts before promotion or drift evaluation.
- Remote storage (e.g., `gs://creditriskbucket/dvc-cache`) can be injected via `DVC_REMOTE_URL`/`DVC_REMOTE_CREDENTIALS`, letting runners download datasets without embedding long-lived secrets.

## MLflow Integration

- `creditrisk.pipelines.train_creditrisk_pd` logs hyperparameters, metrics (ROC-AUC, KS, precision/recall, etc.), tags, and the serialized pipeline under `runs:/<run_id>/model`, while the evaluation stages persist governance artefacts (`reports/metrics.json`, `reports/test_metrics.json`, `reports/evaluation/**`).
- Registry automation (`src/creditrisk/mlops/registry.py`) stages/archives versions when `registry.promote_on_metric` thresholds are met. CD drives promotion via `python -m creditrisk.pipelines.auto_promote` so humans don’t have to run manual CLI commands.
- Post-training validation (`src/creditrisk/testing/post_training.py`) compares `reports/test_metrics.json` with the linked MLflow run to ensure local artefacts match the registry payload.
- FastAPI inference sessions open nested MLflow runs (with `request_id`, `entity_count`, etc.) so online behaviour is observable alongside training history.

## Inference, Testing & Deployment

- `src/creditrisk/pipelines/batch_predict.py` scores CSV/Parquet inputs and is packaged in `Dockerfile.batch` for cron/Glue/Airflow/Cloud Run style workloads.
- `src/creditrisk/serve/api.py` is containerised via `Dockerfile.api`, deployed by GitHub Actions (ECS-ready) or run directly on Compute Engine. `/health`, `/metadata`, `/schema`, `/validate`, `/predict`, and `/metrics` all reuse the training contracts.
- `tests/` covers evaluation utilities, FastAPI routes, and the production drift monitor so contracts don’t regress between releases.

## Automation & Monitoring

- **CI (`ci.yaml`)** – lint/tests/compileall plus `dvc repro --dry-run validate_model` on every PR/push.
- **CD (`cd.yaml`)** – full DAG rebuild, drift generation, MLflow promotion, Docker builds/pushes, optional ECS deploy + rollback, data-freshness gating, and artefact uploads on `main` pushes.
- **Nightly (`nightly.yaml`)** – scheduled invocation that forces the DAG, runs train/test + production drift monitors, publishes CloudWatch metrics, updates `reports/data_freshness.json`, and records `reports/retrain_trigger.json`.
- **Airflow DAG** – mirrors the same stages via BashOperators so operators can re-run ingest/train/validate/monitor directly from a UI on the Compute Engine host.
- **Deployment manifest** – the final DVC stage (`deploy_manifest`) writes `reports/deploy_manifest.json`, consolidating the trained model path, Dockerfiles, CI/CD workflow, and registry metadata so release engineers have a single JSON manifest to feed ECS, batch, or on-prem schedulers.

## Business & Monitoring Alignment

- Automation still meets the proposal’s goals: reproducible nightly runs (>95% success), MLflow-governed promotions, and deployment-ready artefacts for batch + API surfaces.
- Drift and freshness reports provide the “data currency” visibility stakeholders requested and can be piped into Slack/dashboards using the JSON payloads in `reports/`.
- Governance artefacts (ingestion summary, lineage, validation summary, registry promotion report, retrain trigger report, Airflow logs) now live side-by-side on the Compute Engine instance, enabling auditors to trace every scoring decision back through data snapshots, configs, and code.
