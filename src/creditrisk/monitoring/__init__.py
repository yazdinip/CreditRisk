"""Monitoring utilities (drift, telemetry) for the CreditRisk project."""

from __future__ import annotations

from .drift import generate_drift_report

__all__ = ["generate_drift_report"]
