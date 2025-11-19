# GCP Platform Setup

This guide documents the exact steps we used to stand up the CreditRisk platform on Google Cloud. Follow it when you need to recreate the Compute Engine host, self-hosted Airflow, or the on-demand MLflow UI.

## 1. Provision the Compute Engine VM

1. **Create the instance**
   - Machine series: `e2` (e.g., `e2-standard-2` or `e2-highmem-2` if you plan to train on the box).
   - Region/zone: `us-central1` / (any zone).
   - Image: Ubuntu 22.04 LTS, balanced persistent disk (≥30 GB).
   - Firewall: enable HTTP + HTTPS. Add a dedicated rule for SSH/MLflow later if required.
2. **Install dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose git python3-venv unzip
   sudo usermod -aG docker $USER
   # re-login so docker group applies
   ```
3. **Clone the repo and hydrate data**
   ```bash
   git clone https://github.com/<org>/CreditRisk.git
   cd CreditRisk
   python3 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip && pip install -e .
   unzip ~/home-credit-default-risk.zip -d data/raw  # or run python -m creditrisk.pipelines.ingest_data
   dvc repro validate_model
   ```
4. **Build/run the API container**
   ```bash
   docker build -f Dockerfile.api -t creditrisk-api:latest .
   docker run -d --name creditrisk-api \
     -p 80:8080 \
     -v $PWD/configs:/app/configs \
     -v $PWD/models:/app/models \
     -e CONFIG_PATH=configs/creditrisk_pd.yaml \
     -e MLFLOW_TRACKING_URI=mlruns \
     creditrisk-api:latest
   ```

## 2. Self-Hosted Airflow

1. **Create an isolated virtualenv**
   ```bash
   python3 -m venv ~/airflow-venv
   source ~/airflow-venv/bin/activate
   export AIRFLOW_HOME=~/airflow
   AIRFLOW_VERSION=2.9.2
   CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.10.txt"
   pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
   airflow db init
   airflow users create --username admin --password '<secret>' \
     --firstname Admin --lastname User --role Admin --email you@example.com
   ```
2. **Stage the DAG**
   ```bash
   mkdir -p ~/airflow/dags
   cp ~/CreditRisk/orchestration/airflow_creditrisk_dag.py ~/airflow/dags/
   ```
3. **Configure Airflow Variables** (UI → Admin → Variables):
   - `creditrisk_repo`: `/home/p_yazdinia/CreditRisk`
   - `production_dataset_path`: path or GCS URI for the latest production parquet (e.g., `data/production/current.parquet`).
   - `production_model_path`: deployed model (`models/creditrisk_pd_model.joblib` or prod copy).
   - `canary_max_delta`: tolerance (default `0.02`).
4. **Run scheduler and webserver**
   ```bash
   source ~/airflow-venv/bin/activate
   export AIRFLOW_HOME=~/airflow
   airflow scheduler
   # In another terminal
   airflow webserver --port 8080
   ```
5. **Expose the UI**
   - Firewall rule: allow TCP 8080 for instances tagged `http-server` (or your own tag).
   - Browse to `http://<vm-ip>:8080`, log in with the admin user, enable the `creditrisk_pipeline` DAG, and trigger runs as needed.

## 2.1 After VM Restart / Reuse an Existing Container

If the VM was stopped and restarted (or you already built the image once):

1) **Start the API container** (reuse existing):
```bash
docker start creditrisk-api
```
If you need to recreate it (e.g., after a rebuild), stop/remove first, then:
```bash
docker stop creditrisk-api && docker rm creditrisk-api
docker run -d --name creditrisk-api \
  -p 80:8080 \
  -v $PWD/configs:/app/configs \
  -v $PWD/models:/app/models \
  -e CONFIG_PATH=configs/creditrisk_pd.yaml \
  -e MLFLOW_TRACKING_URI=mlruns \
  creditrisk-api:latest
```
Verify:
```bash
curl http://<vm-ip>/health
curl -X POST http://<vm-ip>/predict -H "Content-Type: application/json" -d '{"records":[]}'
```

2) **Restart Airflow services** (if not managed by systemd):
```bash
source ~/airflow-venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow scheduler
# in another terminal
airflow webserver --port 8080
```
Open `http://<vm-ip>:8080` to confirm the DAG is visible.

3) **Run MLflow UI on demand** (tunnel as below):
```bash
cd ~/CreditRisk && source .venv/bin/activate
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
gcloud compute ssh creditrisk-api --zone us-central1-f -- -4 -L 9000:localhost:5000
```
Use Cloud Shell Web Preview → port 9000 to view the UI.

## 3. MLflow UI (on-demand)

1. **Launch the UI only when needed**
   ```bash
   cd ~/CreditRisk
   source .venv/bin/activate
   mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
   ```
2. **Forward the port securely**
   - From Cloud Shell or a local machine with gcloud installed:
     ```bash
     gcloud compute ssh creditrisk-api \
       --zone us-central1-f \
       -- -4 -L 9000:localhost:5000
     ```
   - Keep the SSH session open.
   - In Cloud Shell, use **Web Preview → Change port → 9000** (or visit `http://localhost:9000` if tunnelling from your laptop).
   - When finished, stop the tunnel (Ctrl+C) and terminate the MLflow UI process.
3. **(Optional) Expose publicly**
   - Open firewall port 5000 and start MLflow with `MLFLOW_DISABLE_HTML_HOST_CHECK=1` if you need permanent public access. Tunnelling is recommended for security.

## 5. Common Pitfalls (GCP)

- **Airflow Variables missing**: define `creditrisk_repo`, `production_dataset_path`, `production_model_path`, `canary_max_delta` or the templated tasks fail; defaults fall back to local paths but production monitors need real inputs.
- **Production dataset placeholder**: if you don’t have a production parquet yet, point `production_dataset_path` to `data/processed/test.parquet` until real pulls exist.
- **Resource limits**: `build_feature_store` / `train` can OOM on small VMs; upsize (e.g., 16 GB RAM) or add swap temporarily when reproducing the pipeline.
- **MLflow host checks**: the UI will reject external hosts unless you tunnel or disable the host check. Prefer SSH/Web Preview tunnels instead of opening port 5000.
- **Port conflicts**: ensure nothing else binds to 80/8080/5000 before starting Docker, Airflow, or MLflow.
- **Data hydration**: keep raw Kaggle CSVs in `data/raw/` or push them to your DVC remote (`gs://creditriskbucket/dvc-cache`) and run `dvc pull`; otherwise ingest will fail.
## 4. Daily Operations Checklist

- `dvc repro validate_model` or the Airflow DAG keeps the training stack reproducible.
- `docker ps` / `docker logs creditrisk-api` validate the API container after restarts.
- Airflow UI (port 8080) provides manual backfills, per-task logs, and DAG status.
- MLflow UI is tunneled only when needed; otherwise MLflow continues logging silently under `mlruns/`.
- Nightly GitHub Actions + DVC remotes keep datasets/models synced with the repo, while Airflow covers ad-hoc reruns directly on the VM.
