# CI/CD Strategy

This document describes how we automate quality and deployment gates for the CreditRisk pipeline.  It focuses on reproducibility, governance, and safe promotion of the Kaggle-style probability-of-default model.

## Objectives

1. **Prove every change is safe**: run linting, tests, and a DVC dry-run of the pipeline on every PR / push.
2. **Capture traceable artifacts**: persist metrics, lineage (`reports/data_lineage.json`), and packaged models as GitHub Action artifacts for auditability.
3. **Controlled promotions**: only merge-to-main triggers the CD workflow, which rebuilds the train pipeline, logs to MLflow, and uploads deployment bundles for batch + API serving.

## Environments

| Stage | Runner | Purpose |
| --- | --- | --- |
| pull_request / push | `ubuntu-latest` | Fast checks (linters, pytest, DVC `repro --dry`). Uses light-weight cache for `.venv`, `.dvc/cache`. |
| main (CD) | `ubuntu-latest` (self-host optional) | Full pipeline execution (`dvc repro validate_model`), artifact upload, optional registry promotion. |
| nightly schedule / dispatch | `ubuntu-latest` | Cron-like orchestration that forces the full DAG, produces drift + freshness reports, and fails if ingestion falls outside the SLA. |

## Credentials / Secrets

The workflows expect these GitHub secrets:

| Secret | Description |
| --- | --- |
| `MLFLOW_TRACKING_URI` | Remote tracking server (or local path). |
| `MLFLOW_TRACKING_TOKEN` | Optional PAT/API token for MLflow. |
| `DVC_REMOTE_URL` | Remote storage (e.g., Azure blob, S3). |
| `DVC_REMOTE_CREDENTIALS` | Encoded credentials/env vars for DVC remote. |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | IAM user/role creds that can update ECS services + read task definitions. |
| `AWS_REGION` | Region hosting the ECS cluster + load balancer. |
| `ECS_CLUSTER_NAME` / `ECS_SERVICE_NAME` | Target ECS service that should run the FastAPI container. |
| `PREDICT_ENDPOINT_URL` | HTTPS endpoint (ALB / API Gateway) for smoke tests hitting `/health` + `/predict`. |

Secrets are injected only in the CD job.

## Workflow Summary

### 1. `ci.yaml`

- Triggers: `pull_request`, `push` (non-main).
- Steps:
  - Checkout with `fetch-depth: 0`.
  - Set up Python 3.11.
  - Cache `.venv` and `.dvc/cache`.
  - Install dependencies via `pip install -e .`.
  - `ruff check .`, `bandit -q -r src`, and `python -m compileall`.
  - `pytest`.
  - `dvc repro --dry-run validate_model` (which exercises every stage, ensuring ingestion/feature/store/split/train/test/validation remain reproducible).

### 2. `cd.yaml`

- Trigger: push to `main`.
- Steps:
  - Checkout with fetch.
  - Configure Python/DVC.
  - Authenticate DVC remote using `DVC_REMOTE_*`.
  - `dvc pull` to get raw datasets.
  - `dvc repro validate_model`.
  - `dvc repro monitor_drift`.
  - `python -m creditrisk.utils.data_freshness --max-age-hours 48 --fail-on-stale` (captures `reports/data_freshness.json` and prevents stale merges).
  - Build/push containers: leverage `docker/build-push-action` to publish `Dockerfile.api` and `Dockerfile.batch` images to GHCR (`ghcr.io/<org>/creditrisk-{api|batch}`) tagged with both `latest` and the Git SHA so environments downstream can pull immutable artifacts.
  - Configure AWS credentials, download the currently active ECS task definition, render it with the freshly pushed image tag, deploy to the configured cluster/service, and wait for stability. `/health` and `/predict` are smoke-tested via the load balancer; failures trigger an automatic rollback to the previously running task definition so production traffic never sees a broken build.
  - Upload artifacts:
    - `reports/metrics.json`
    - `reports/data_lineage.json`
    - `reports/ingestion_summary.json`
    - `reports/data_freshness.json`
    - `models/baseline_model.joblib`
    - `reports/evaluation/**`
    - `reports/post_training_validation.json`
  - Auto-promote: run `python -m creditrisk.pipelines.auto_promote --stage Production` when `reports/registry_promotion.json` exists and the validation summary reports `status: passed`.

Artifacts are retained for 30 days so reviewers can download the exact model bundle.

### 3. `nightly.yaml`

- Triggers: cron (`0 6 * * *`) and manual `workflow_dispatch`.
- Steps:
  - Same environment prep + DVC remote configuration as CD.
  - `dvc repro --force validate_model` and `dvc repro --force monitor_drift` to guarantee a fresh DAG run, even if no code changed.
  - `python -m creditrisk.utils.data_freshness --max-age-hours 36 --fail-on-stale` to produce `reports/data_freshness.json` and fail loudly when the ingestion snapshot exceeds the SLA.
  - `python -m creditrisk.monitoring.production --publish-metrics` to compare the latest production extracts against the training baseline, push the drift share/flag into CloudWatch (when AWS credentials are scoped), and write `reports/retrain_trigger.json`. When the drift share exceeds the configured threshold the workflow automatically runs the retrain command so a fresh model is queued without human intervention.
  - Upload a trimmed artifact set (`ingestion_summary.json`, `data_freshness.json`, drift outputs) so on-call engineers can inspect the results without reproducing them locally.

## Next Steps

1. Link GH Actions status to PR requirements.
2. Add slack/webhook notifications for CD success/failure.
3. Add manual approval gates or change-management hooks before Production promotion if required by governance.
