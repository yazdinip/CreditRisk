# FastAPI Serving Guide

This service wraps the latest validated model for online inference. It runs via uvicorn (see `Dockerfile.api`) and reuses the same validation/governance logic as training.

## Endpoints at a Glance
- `GET /health` – liveness (`{"status":"ok"}`).
- `GET /metadata` – model path, mtime, size, decision threshold, tracking experiment, governance tags, entity column.
- `GET /schema` – expected feature columns, entity and target columns, total feature count.
- `POST /validate` – schema/contract check only; no scoring.
- `POST /predict` – batch scoring; returns predictions + probabilities; logs inference metrics/tags to MLflow if configured.
- `GET /metrics` – lightweight request counters and timestamps (scrape-friendly).

## Payload Model
All request bodies are JSON:
```json
{
  "records": [ { "<feature>": value, ... } ],
  "threshold": 0.5
}
```
- `records` must match the training schema (same feature names as `configs/creditrisk_pd.yaml`). Unknown/missing columns trigger a 400/422 with details.
- Any `TARGET` column is ignored; only features + entity id are expected.
- `threshold` overrides `inference.decision_threshold` per request (optional).

## Quick Tests Against a Live Host
Example base URL: `http://136.113.244.176` (port 80). Swap to your host/IP as needed. Commands assume repo root so `sample_payload.json` exists.

### Health
```bash
curl http://136.113.244.176/health
# {"status":"ok"}
```
Latest run (UTC): HTTP 200, `X-Request-ID: 70c539ce49e64b71b5e77ee417774990`, body `{"status":"ok"}`.

### Schema
```bash
curl http://136.113.244.176/schema
```
Latest run: HTTP 200, `total_features: 162`, `entity_column: "SK_ID_CURR"` (full feature list returned).

### Predict with the provided sample payload
PowerShell (forces curl.exe):
```powershell
curl.exe -s -D - -H "Content-Type: application/json" -X POST `
  --data "@sample_payload.json" http://136.113.244.176/predict
```

Invoke-RestMethod (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://136.113.244.176/predict" -Method Post `
  -ContentType "application/json" -InFile "sample_payload.json"
```

bash:
```bash
curl -s -D - -H "Content-Type: application/json" \
  -X POST --data @sample_payload.json http://136.113.244.176/predict
```

Sample live response:
```json
{
  "predictions": [
    {
      "SK_ID_CURR": 396899.0,
      "prediction": 0,
      "probability": 0.0848717987537384
    }
  ]
}
```
Headers from last call: HTTP 200, `X-Request-ID: aa68a7e0d572455eb88eedb729bf7a44`.

### Validate-only (no scoring)
```bash
curl -s -H "Content-Type: application/json" \
  -X POST --data @sample_payload.json http://136.113.244.176/validate
```
Returns a validity summary; schema mismatches are 400/422 with details.
Latest run: HTTP 200, body `{"records":1,"valid":true,"missing_columns":[]}`, `X-Request-ID: a4b634e5c68748cbae7b62cadd45941a`.

### Metadata snapshot
```bash
curl http://136.113.244.176/metadata
```
Fields include `model_path`, `model_last_modified`, `model_size_bytes`, `decision_threshold`, `tracking_experiment`, `governance_tags`, `entity_column`.
Latest run: HTTP 200, `model_path: models/creditrisk_pd_model.joblib`, `model_last_modified: 2025-11-26T01:06:02.080445+00:00`, `model_size_bytes: 1243226`, `decision_threshold: 0.5`, `X-Request-ID: aba0783e3f9a413fa84883f174a934e0`.

### Metrics counters
```bash
curl http://136.113.244.176/metrics
```
Expect `requests_total`, `predictions_total`, `errors_total`, `last_request_at`, `last_predict_at`.
Latest run: HTTP 200, `requests_total: 107`, `predictions_total: 3`, `errors_total: 0`, `last_request_at: 2025-11-26T01:16:08Z`, `last_predict_at: 2025-11-26T01:15:57Z`, `X-Request-ID: b04079c29cf640309a695eb48d93c191`.

### Empty predict (connectivity smoke)
```bash
curl -X POST http://136.113.244.176/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[]}'
```

### Python (requests) example
```python
import json, requests

API_URL = "http://136.113.244.176/predict"
payload = json.load(open("sample_payload.json"))
resp = requests.post(API_URL, json=payload, timeout=10)
resp.raise_for_status()
print(resp.json())
```

## Running the API Container (VM)
Redeploy/restart:
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
- Validation is identical to training (Pandera + ValidationRunner); malformed payloads return 400/422 with details.
- MLflow logging occurs when `MLFLOW_TRACKING_URI` is set; request_id/entity_count tags and prediction metrics are emitted.
- `/metrics` gives lightweight counters; CloudWatch publishing is active when AWS creds exist and `monitoring.cloudwatch_namespace` is set.
- For large batches use the batch CLI (`python -m creditrisk.pipelines.batch_predict`) or `Dockerfile.batch` to avoid HTTP timeouts.
- Schema drift protection: ensure `entity_id_column` and all selected features exist; extra/missing columns are rejected.

## Authentication
Currently open on port 80 via the VM firewall. Add TLS + auth (nginx/ingress/load balancer) if exposure must be restricted.***
