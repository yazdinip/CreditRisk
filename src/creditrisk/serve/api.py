"""FastAPI service for online inference (no deployment logic included)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from creditrisk.config import Config
from creditrisk.observability.logging import (
    configure_context,
    generate_request_id,
    publish_metric,
    setup_json_logging,
)
from creditrisk.prediction.helpers import build_output_frame, predict_with_threshold
from creditrisk.serve.governance import InferenceAuditLogger
from creditrisk.validation.runner import ValidationRunner


setup_json_logging()
LOGGER = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., description="List of feature dictionaries matching the training schema."
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Optional override for the decision threshold.",
    )


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


def _load_pipeline(config: Config, model_override: Optional[Path]) -> Any:
    model_path = Path(model_override) if model_override else config.paths.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Could not locate serialized model at {model_path}")
    return joblib.load(model_path)


def create_app(
    config_path: str = "configs/creditrisk_pd.yaml",
    model_path: Optional[str] = None,
    config_override: Optional[Config] = None,
) -> FastAPI:
    app = FastAPI(title="CreditRisk Inference API", version="0.1.0")

    state: Dict[str, Any] = {}

    @app.on_event("startup")
    async def _startup() -> None:
        cfg = config_override or Config.from_yaml(config_path)
        pipeline = _load_pipeline(cfg, Path(model_path) if model_path else None)
        state["config"] = cfg
        state["pipeline"] = pipeline
        state["validator"] = ValidationRunner(cfg.validation)
        state["expected_features"] = list(cfg.features.selected_columns or [])
        state["audit_logger"] = InferenceAuditLogger(cfg.tracking, cfg.inference)
        state["metrics"] = {
            "requests_total": 0,
            "predictions_total": 0,
            "errors_total": 0,
            "last_request_at": None,
            "last_predict_at": None,
        }
        configure_context(service="creditrisk-api", model_name=cfg.registry.model_name)

    @app.middleware("http")
    async def _logging_middleware(request: Request, call_next):
        cfg: Config | None = state.get("config")
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        request.state.request_id = request_id
        start = time.perf_counter()
        metrics: Dict[str, Any] = state.get("metrics", {})
        metrics["requests_total"] = metrics.get("requests_total", 0) + 1
        metrics["last_request_at"] = datetime.now(timezone.utc).isoformat()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - middleware guard
            duration_ms = (time.perf_counter() - start) * 1000
            metrics["errors_total"] = metrics.get("errors_total", 0) + 1
            # Skip CloudWatch publishing when no namespace is configured to avoid boto3 errors on air-gapped hosts.
            if cfg and cfg.monitoring.cloudwatch_namespace:
                publish_metric(
                    namespace=cfg.monitoring.cloudwatch_namespace,
                    metric_name="RequestFailures",
                    value=1,
                    dimensions={"Path": request.url.path},
                )
            LOGGER.exception(
                "API request failed",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        LOGGER.info(
            "API request completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        if cfg and cfg.monitoring.cloudwatch_namespace:
            publish_metric(
                namespace=cfg.monitoring.cloudwatch_namespace,
                metric_name="RequestLatencyMs",
                value=round(duration_ms, 2),
                unit="Milliseconds",
                dimensions={"Path": request.url.path},
            )
        response.headers["X-Request-ID"] = request_id
        return response

    def _ensure_ready() -> None:
        if "pipeline" not in state or "config" not in state:
            raise HTTPException(status_code=503, detail="Model artifacts not ready.")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/metadata")
    async def metadata() -> JSONResponse:
        _ensure_ready()
        cfg: Config = state["config"]
        model_path = cfg.paths.model_path
        modified = None
        size = None
        if model_path.exists():
            stat = model_path.stat()
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        payload = {
            "model_path": str(model_path),
            "model_last_modified": modified,
            "model_size_bytes": size,
            "decision_threshold": cfg.inference.decision_threshold,
            "tracking_experiment": cfg.tracking.experiment_name,
            "governance_tags": cfg.inference.governance_tags,
            "entity_column": cfg.data.entity_id_column,
        }
        return JSONResponse(payload)

    @app.get("/schema")
    async def schema() -> JSONResponse:
        _ensure_ready()
        cfg: Config = state["config"]
        features = state.get("expected_features", [])
        payload = {
            "feature_columns": features,
            "entity_column": cfg.data.entity_id_column,
            "target_column": cfg.data.target_column,
            "total_features": len(features),
        }
        return JSONResponse(payload)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        _ensure_ready()
        cfg: Config = state["config"]
        pipeline = state["pipeline"]

        request_id = getattr(request.state, "request_id", generate_request_id())

        if not payload.records:
            return PredictResponse(predictions=[])

        payload_df = pd.DataFrame(payload.records)
        if payload_df.empty:
            return PredictResponse(predictions=[])

        feature_df = payload_df.drop(columns=[cfg.data.target_column], errors="ignore")

        validator: ValidationRunner | None = state.get("validator")
        expected_features: List[str] = state.get("expected_features", [])
        if validator and expected_features:
            try:
                validator.validate_inference_request(
                    feature_df,
                    expected_columns=expected_features,
                    entity_column=cfg.data.entity_id_column,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        threshold = (
            payload.threshold
            if payload.threshold is not None
            else cfg.inference.decision_threshold
        )
        LOGGER.info(
            "Predict request received",
            extra={
                "request_id": request_id,
                "entity_count": len(payload_df),
                "path": "/predict",
            },
        )
        try:
            decisions, probabilities = predict_with_threshold(
                pipeline=pipeline,
                features=feature_df,
                threshold=threshold,
            )
        except ValueError as exc:  # column mismatch, etc.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result_df = build_output_frame(
            raw_df=payload_df,
            decisions=decisions,
            probabilities=probabilities,
            entity_column=cfg.data.entity_id_column,
        )
        records = result_df.to_dict(orient="records")
        LOGGER.info(
            "Predict request served",
            extra={
                "request_id": request_id,
                "entity_count": len(records),
                "path": "/predict",
            },
        )
        audit_logger: InferenceAuditLogger | None = state.get("audit_logger")
        if audit_logger:
            try:
                audit_logger.log_request(
                    request_id=request_id,
                    threshold=threshold,
                    payload_df=payload_df,
                    result_df=result_df,
                    probabilities=probabilities,
                    entity_column=cfg.data.entity_id_column,
                    extra_tags={"model_path": str(cfg.paths.model_path)},
                )
            except Exception:  # pragma: no cover
                LOGGER.exception("Failed to log inference metadata")
        metrics = state.get("metrics")
        if metrics is not None:
            metrics["predictions_total"] = metrics.get("predictions_total", 0) + len(records)
            metrics["last_predict_at"] = datetime.now(timezone.utc).isoformat()
        return PredictResponse(predictions=records)

    @app.post("/validate")
    async def validate(payload: PredictRequest) -> JSONResponse:
        _ensure_ready()
        cfg: Config = state["config"]
        validator: ValidationRunner | None = state.get("validator")
        expected_features: List[str] = state.get("expected_features", [])
        payload_df = pd.DataFrame(payload.records)
        response: Dict[str, Any] = {"records": len(payload.records), "valid": True, "missing_columns": []}
        if validator and expected_features:
            try:
                validator.validate_inference_request(
                    payload_df.drop(columns=[cfg.data.target_column], errors="ignore"),
                    expected_columns=expected_features,
                    entity_column=cfg.data.entity_id_column,
                )
            except ValueError as exc:
                response["valid"] = False
                response["error"] = str(exc)
        return JSONResponse(response)

    @app.get("/metrics")
    async def metrics() -> JSONResponse:
        metrics_snapshot = dict(state.get("metrics", {}))
        return JSONResponse(metrics_snapshot)

    return app


app = create_app()
