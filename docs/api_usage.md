# FastAPI Serving Guide

This service wraps the latest validated model for online inference. All routes run inside the `creditrisk-api` container (uvicorn) and reuse the same validation/governance logic used in training.

## Endpoints

- `GET /health` — simple liveness check (`{"status":"ok"}`).
- `GET /metadata` — model path, last modified, size, decision threshold, tracking experiment, governance tags, entity column.
- `GET /schema` — expected feature columns, entity and target columns, total feature count.
- `POST /validate` — validate a payload against the Pandera/ValidationRunner contracts without scoring (returns errors if present).
- `POST /predict` — score a batch of records; returns predictions + probabilities; logs inference metrics/tags to MLflow if configured.
- `GET /metrics` — lightweight request counters and timestamps (suitable for scraping).

## Payload and Validation

All request bodies are JSON. The primary input model:
```json
{
  "records": [ { "<feature>": value, ... } ],
  "threshold": 0.5   // optional; overrides default decision threshold
}
```
- `records` must be a list of dicts with columns matching training schema. Unknown/missing columns are rejected by Pandera/ValidationRunner.
- The service drops any `TARGET` column if present; it expects only features + entity id.
- The decision threshold defaults to `inference.decision_threshold` in `configs/creditrisk_pd.yaml`, but callers can override per-request via `threshold`.

## Example Requests

Base URL: `http://<vm-ip>` (port 80) when running the Docker image on Compute Engine.

Liveness:
```bash
curl http://34.63.206.75/health
```

Schema:
```bash
curl http://34.63.206.75/schema
```

Empty predict (connectivity/validation smoke test):
```bash
curl -X POST http://34.63.206.75/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[]}'
```

Predict with sample rows (fill in real feature values; omit TARGET):
```bash
curl -X POST http://34.63.206.75/predict \
  -H "Content-Type: application/json" \
  -d '{
        "records": [
          {
            "SK_ID_CURR": 100001,
            "CNT_CHILDREN": 0,
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 406597.5,
            "...": "..."
          }
        ]
      }'
```

## Running the API Container

Redeploy/restart the container on the VM:
```bash
cd ~/CreditRisk
docker run -d --name creditrisk-api \
  -p 80:8080 \
  -v $PWD/configs:/app/configs \
  -v $PWD/models:/app/models \
  -e CONFIG_PATH=configs/creditrisk_pd.yaml \
  -e MLFLOW_TRACKING_URI=mlruns \
  creditrisk-api:latest
```
Rebuild after code changes:
```bash
docker stop creditrisk-api && docker rm creditrisk-api
docker build -f Dockerfile.api -t creditrisk-api:latest .
```

## Observability and Governance

- Request/response validation uses the same Pandera contracts as training; malformed payloads return HTTP 422/400 with error details.
- Inference sessions are logged to MLflow (if `MLFLOW_TRACKING_URI` is set), including request_id/entity_count tags and prediction metrics.
- `/metrics` exposes lightweight counters; CloudWatch publishing is available when AWS creds are set and `monitoring.cloudwatch_namespace` is configured.
- Large payloads: for very big batches, prefer the batch CLI (`python -m creditrisk.pipelines.batch_predict`) or the `Dockerfile.batch` image to avoid HTTP timeouts.
- Schema drift: ensure `entity_id_column` and all selected features exist; unexpected/extra columns are rejected by validation.

## Authentication

The API is currently open on port 80 (exposed via the VM firewall). Add a reverse proxy (nginx, Cloud Load Balancer) with TLS and auth if you need to restrict access.***
