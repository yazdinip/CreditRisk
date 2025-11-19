# Local Pipeline & UI Guide

Run the full CreditRisk stack locally (training, monitoring, FastAPI, Airflow optional, MLflow UI) with minimal setup.

## Prerequisites
- Python 3.10+
- Git
- Docker (for containerized serving; optional if you run uvicorn directly)
- DVC (installed via `pip install -e .`)
- Kaggle API token (`~/.kaggle/kaggle.json`) if pulling data from Kaggle instead of a local ZIP/DVC remote

## 1) Clone + Python Environment
```bash
git clone https://github.com/<org>/CreditRisk.git
cd CreditRisk
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## 2) Data Hydration
Pick one:
- **Local ZIP**: place `home-credit-default-risk.zip` locally, unzip to `data/raw/`:
  ```bash
  mkdir -p data/raw
  unzip ~/home-credit-default-risk.zip -d data/raw
  ```
- **Kaggle CLI**: ensure `~/.kaggle/kaggle.json` exists, then:
  ```bash
  python -m creditrisk.pipelines.ingest_data --config configs/creditrisk_pd.yaml
  ```
- **DVC remote**: configure your remote (`dvc remote add --default <name> <url>`) and run `dvc pull`.

## 3) Reproduce the Pipeline
```bash
dvc repro validate_model
```
Outputs:
- `models/creditrisk_pd_model.joblib`
- `reports/metrics.json`, `reports/test_metrics.json`, `reports/drift_report.*`, `reports/post_training_validation.json`
- `data/processed/{train,test}.parquet`, `data/processed/feature_store.parquet`

## 4) FastAPI (local)

Direct uvicorn:
```bash
source .venv/bin/activate
uvicorn creditrisk.serve.api:app --host 0.0.0.0 --port 8080
```
Test:
```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d '{"records":[]}'
```

Dockerized:
```bash
docker build -f Dockerfile.api -t creditrisk-api:local .
docker run -d --name creditrisk-api \
  -p 8080:8080 \
  -v $PWD/configs:/app/configs \
  -v $PWD/models:/app/models \
  -e CONFIG_PATH=configs/creditrisk_pd.yaml \
  -e MLFLOW_TRACKING_URI=mlruns \
  creditrisk-api:local
```

## 5) MLflow UI (local)
```bash
source .venv/bin/activate
MLFLOW_DISABLE_HTML_HOST_CHECK=1 \
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
# open http://127.0.0.1:5000
```

## 6) Airflow (optional local scheduler/UI)
```bash
python3 -m venv ~/airflow-venv
source ~/airflow-venv/bin/activate
export AIRFLOW_HOME=~/airflow
AIRFLOW_VERSION=2.9.2
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.10.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
airflow db init
airflow users create --username admin --password '<pw>' --firstname Admin --lastname Local --role Admin --email you@example.com
mkdir -p ~/airflow/dags
cp ~/CreditRisk/orchestration/airflow_creditrisk_dag.py ~/airflow/dags/
airflow scheduler    # terminal 1
airflow webserver --port 8081   # terminal 2
```
Open `http://127.0.0.1:8081`, enable `creditrisk_pipeline`, set Airflow Variables if needed (`creditrisk_repo` → `/path/to/CreditRisk`, `production_dataset_path`, `production_model_path`, `canary_max_delta`).

## 7) Batch/Monitoring CLIs
- Train/eval: `dvc repro validate_model`
- Drift (train vs test): `dvc repro monitor_drift`
- Production drift: `python -m creditrisk.monitoring.production --config configs/creditrisk_pd.yaml --current data/production/current.parquet --publish-metrics`
- Canary: `dvc repro canary_validation`

## 8) Quick Checks
- API: `curl http://127.0.0.1:8080/health` and `/predict`.
- MLflow UI: `http://127.0.0.1:5000`.
- Airflow UI (if enabled): `http://127.0.0.1:8081`.
- Artefacts: confirm `models/creditrisk_pd_model.joblib`, `reports/*.json/html` exist after `dvc repro`.

## Common Pitfalls (Local)
- **Kaggle auth**: `~/.kaggle/kaggle.json` must exist/0600 for ingestion; otherwise ingest fails. Alternatively use local ZIP or DVC remote.
- **Ports in use**: change Airflow webserver port if 8081 is occupied; stop stale MLflow/uvicorn processes before restarting.
- **MLflow host check**: use `MLFLOW_DISABLE_HTML_HOST_CHECK=1` locally if you access via non-localhost; otherwise stay on `http://127.0.0.1:5000`.
- **Data files missing**: `build_feature_store`/`split` expect all raw CSVs present; re-run ingest or `dvc pull` before `dvc repro`.
- **Airflow Variables**: if using local Airflow, set `creditrisk_repo` and production paths or keep defaults aligned with your workspace paths.
