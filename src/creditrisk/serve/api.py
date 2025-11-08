"""FastAPI service for online inference (no deployment logic included)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from creditrisk.config import Config
from creditrisk.prediction.helpers import build_output_frame, predict_with_threshold


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

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        if "pipeline" not in state or "config" not in state:
            raise HTTPException(status_code=503, detail="Model artifacts not ready.")

        cfg: Config = state["config"]
        pipeline = state["pipeline"]

        if not request.records:
            return PredictResponse(predictions=[])

        payload_df = pd.DataFrame(request.records)
        if payload_df.empty:
            return PredictResponse(predictions=[])

        feature_df = payload_df.drop(columns=[cfg.data.target_column], errors="ignore")

        threshold = (
            request.threshold
            if request.threshold is not None
            else cfg.inference.decision_threshold
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
        return PredictResponse(predictions=records)

    return app


app = create_app()
