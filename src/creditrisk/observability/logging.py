"""Structured logging helpers used across serving surfaces."""

from __future__ import annotations

import json
import logging
import socket
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore

class JsonFormatter(logging.Formatter):
    """Simple JSON formatter that injects standard metadata into each record."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "host": socket.gethostname(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        for attr in (
            "request_id",
            "entity_count",
            "path",
            "status_code",
            "duration_ms",
            "stage",
        ):
            if hasattr(record, attr):
                payload[attr] = getattr(record, attr)

        payload.update(GLOBAL_LOG_TAGS)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


GLOBAL_LOG_TAGS: Dict[str, Any] = {}


def configure_context(**kwargs: Any) -> None:
    """Set global tags applied to every log record."""
    GLOBAL_LOG_TAGS.update(kwargs)


def setup_json_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a JSON formatter if not already attached."""
    root = logging.getLogger()
    existing = [
        handler
        for handler in root.handlers
        if isinstance(getattr(handler, "formatter", None), JsonFormatter)
    ]
    if existing:
        root.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def generate_request_id() -> str:
    """Return a hex request identifier suitable for correlation."""
    return uuid.uuid4().hex


def publish_metric(
    *,
    namespace: str,
    metric_name: str,
    value: float,
    unit: str = "Count",
    dimensions: Optional[Dict[str, str]] = None,
) -> None:
    """Publish a single metric to CloudWatch (best-effort)."""
    if boto3 is None:
        logging.getLogger(__name__).debug("boto3 not installed; skipping metric publish.")
        return
    try:
        client = boto3.client("cloudwatch")
        metric_payload = {
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit,
        }
        if dimensions:
            metric_payload["Dimensions"] = [
                {"Name": key, "Value": value} for key, value in dimensions.items()
            ]
        client.put_metric_data(Namespace=namespace, MetricData=[metric_payload])
    except Exception:  # pragma: no cover - optional
        logging.getLogger(__name__).exception("Failed to publish CloudWatch metric.")
