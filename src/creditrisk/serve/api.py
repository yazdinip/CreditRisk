"""FastAPI service for online inference (no deployment logic included)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from creditrisk.config import Config
from creditrisk.observability.logging import generate_request_id, setup_json_logging
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
    config_path: str = "configs/baseline.yaml",
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

    @app.middleware("http")
    async def _logging_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - middleware guard
            duration_ms = (time.perf_counter() - start) * 1000
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
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        if "pipeline" not in state or "config" not in state:
            raise HTTPException(status_code=503, detail="Model artifacts not ready.")

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
        return PredictResponse(predictions=records)

    return app


app = create_app()
