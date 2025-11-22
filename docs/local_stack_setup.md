# Local Stack Setup & Lessons Learned

This guide walks through running the full CreditRisk platform locally on Windows (PowerShell) exactly as exercised during the latest dry run: Python/DVC pipeline, Dockerised FastAPI, MLflow UI, and Airflow. It also records the bumps we hit so you can skip the guesswork next time.

## 1. Prerequisites

| Component | Notes |
|-----------|-------|
| Python 3.12 (or 3.11) | Install via `py` on Windows. We use 3.12 for the repo and 3.12 for Airflow�s Docker container. |
| Git + PowerShell | Repo checkout + commands. |
| Docker Desktop | Required for the FastAPI image and for running Airflow reliably on Windows. |
| Kaggle data | Place the raw CSVs under `data/raw/` (as tracked by DVC) or run the ingestion pipeline. |
| Optional: WSL2 | Not required once Docker Desktop is running, but it avoids compatibility gaps if you want to rerun everything outside Windows. |

## 2. Bootstrap the Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy==2.0.2          # avoids long source builds on Windows
pip install -e .
```

> **Lesson learned:** the editable install pulls a large dependency set (DVC, MLflow, Evidently, XGBoost). Pinning `numpy==2.0.2` before `pip install -e .` prevented repeated source builds that timed out earlier runs.

## 3. Reproduce the DVC Pipeline

1. Hydrate data (either `python -m creditrisk.pipelines.ingest_data --config configs/creditrisk_pd.yaml` or ensure `data/raw/*.csv` exist).
2. Run the full validation DAG:
   ```powershell
   dvc repro validate_model
   dvc repro monitor_drift
   ```
3. Inspect `reports/post_training_validation.json` and `reports/drift_report.*` to confirm metrics (latest `roc_auc` 0.7770, `normalized_gini` 0.5540).

> **Lesson learned:** most stages were cached; only `validate_model` reran. Keep `dvc.lock` under Git to record the fingerprint, and expect `mlruns/` to fill with the MLflow run used by the training stages.

## 4. FastAPI via Docker

Build and run the image once Docker Desktop is up (`docker info` should succeed):

```powershell
docker build -f Dockerfile.api -t creditrisk-api:local .
docker run -d --name creditrisk-api -p 8080:8080 `
  -v ${PWD}/configs:/app/configs `
  -v ${PWD}/models:/app/models `
  -e CONFIG_PATH=configs/creditrisk_pd.yaml `
  -e MLFLOW_TRACKING_URI=mlruns `
  creditrisk-api:local
```

Smoke tests:
```powershell
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/metadata
```

Stop the container when finished: `docker stop creditrisk-api && docker rm creditrisk-api`.

> **Lesson learned:** without AWS credentials the container logs  `botocore.exceptions.NoRegionError` when trying to publish CloudWatch metrics. It�s safe to ignore locally or set `MONITORING__CLOUDWATCH_ENABLED=false`.

## 5. MLflow UI On Demand

```powershell
.\.venv\Scripts\Activate.ps1
$env:MLFLOW_DISABLE_HTML_HOST_CHECK = '1'
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Verify locally (or via a PowerShell background job):
```powershell
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:5000/
```

CTRL+C (or kill the spawned `python` PIDs) to stop the UI when you�re done. Use SSH tunnelling if you need to expose it beyond localhost.

> **Lesson learned:** `mlflow ui` leaves background Python processes when started via PowerShell jobs. Use `Get-Process python | Stop-Process` as needed to clean up before rerunning.

## 6. Airflow on Windows (Docker-backed)

Direct `airflow db init` on Windows failed due to path and symlink restrictions (see **Pitfalls** below). We run the official image instead.

```powershell
docker run -d --name creditrisk-airflow -p 8081:8080 `
  -e AIRFLOW_UID=50000 -e AIRFLOW_GID=0 `
  -e AIRFLOW__CORE__LOAD_EXAMPLES=False `
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/orchestration `
  -v ${PWD}:/opt/airflow `
  apache/airflow:2.9.2 airflow standalone
```

What this does:
- Mounts the repo at `/opt/airflow`, so `orchestration/airflow_creditrisk_dag.py` is auto-loaded.
- Starts the scheduler, triggerer, and webserver on host port `8081`.
- Generates `standalone_admin_password.txt` in the repo root. Username is `admin`, password is the file contents (ignored via `.gitignore`).

Common follow-up commands:
```powershell
docker exec creditrisk-airflow airflow dags list
docker logs -f creditrisk-airflow
docker stop creditrisk-airflow && docker rm creditrisk-airflow   # when finished
```

> **Lessons learned:**
> 1. Native Windows install raised `AirflowConfigException: Cannot use relative path: sqlite:///C:/...` even after forcing absolute URIs, and `OSError while attempting to symlink the latest log directory`. Docker sidesteps both.
> 2. Mounting the repo root means Airflow writes `airflow.cfg`, `airflow.db`, `logs/`, and password files beside the code. They�re now ignored by `.gitignore`; relocate the bind mount if you want a cleaner workspace (`-v ${env:USERPROFILE}/airflow-data:/opt/airflow`).

## 7. Housekeeping & Troubleshooting

| Symptom | Fix |
|---------|-----|
| `pip install -e .` hangs compiling numpy | Pre-install `numpy==2.0.2` (or grab a wheel). |
| `docker build` fails with `docker: The term is not recognized` | Docker Desktop/daemon not running; start the service or relaunch Docker. |
| `/health` works but logs show CloudWatch errors | Ignore locally or set `MONITORING__CLOUDWATCH_ENABLED=false`. |
| `mlflow ui` reports host check failures | Set `MLFLOW_DISABLE_HTML_HOST_CHECK=1` or stick to `127.0.0.1`. |
| PowerShell still shows `webserver_config.py` / `airflow.db` as untracked | Ensure `.gitignore` is updated (see repo version) or delete the files if you stopped the container. |
| Airflow UI login fails | Read `standalone_admin_password.txt` for the generated password, or exec into the container to create a new user (`airflow users create ...`). |
| Airflow DAG paused | Open http://127.0.0.1:8081, log in as admin, toggle `creditrisk_pipeline` to *On*, and set Variables (`creditrisk_repo`, `production_dataset_path`, `production_model_path`, `canary_max_delta`). |

## 8. Quick Reference Checklist

1. `python -m venv .venv` + `pip install -e .` (pin numpy first if on Windows).
2. `dvc repro validate_model` → confirm metrics in `reports/`.
3. `docker build -f Dockerfile.api -t creditrisk-api:local .` and run the container; hit `/health` & `/metadata`.
4. `mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000` when you need the experiment UI.
5. `docker run ... apache/airflow:2.9.2 airflow standalone` for Airflow; grab the password file and enable the DAG.
6. Stop containers/cleanup processes when finished; Git stays clean because `.gitignore` excludes the generated Airflow files.

Following these steps reproduces the entire local workflow exactly as we executed it, including the lessons that keep the environment predictable on Windows hosts.
