"""MLflow Model Registry helpers."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from creditrisk.config import Config, RegistryConfig

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover - runtime optional
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MlflowException = Exception  # type: ignore[misc]

LOGGER = logging.getLogger(__name__)


class ModelRegistryManager:
    """Wrapper around MLflow model registration and promotion."""

    def __init__(self, config: Config):
        if mlflow is None or MlflowClient is None:
            raise ImportError("MLflow is required for model registry operations.")
        self.config = config
        mlflow.set_tracking_uri(config.tracking.tracking_uri)
        self.client = MlflowClient()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register_run(
        self,
        run_id: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[int]:
        """Register a logged MLflow run as a model version."""
        registry_cfg = self.config.registry
        if not registry_cfg.enabled:
            LOGGER.info("Model registry disabled; skipping registration for run %s.", run_id)
            return None

        self._ensure_registered_model(registry_cfg)
        source = f"runs:/{run_id}/model"
        description = f"Auto-registered from run {run_id}"
        try:
            model_version = self.client.create_model_version(
                name=registry_cfg.model_name,
                source=source,
                run_id=run_id,
                description=description,
                tags=tags or {},
            )
        except MlflowException as exc:  # pragma: no cover - network/config specific
            LOGGER.error("Model registration failed: %s", exc)
            raise

        LOGGER.info(
            "Registered run %s as %s version %s.",
            run_id,
            registry_cfg.model_name,
            model_version.version,
        )

        target_stage = self._resolve_stage_for_metrics(registry_cfg, metrics or {})
        if target_stage:
            self.transition_stage(
                version=model_version.version,
                stage=target_stage,
                archive_existing=registry_cfg.archive_existing_on_promote,
            )
        return int(model_version.version)

    def _ensure_registered_model(self, registry_cfg: RegistryConfig) -> None:
        try:
            self.client.get_registered_model(registry_cfg.model_name)
        except MlflowException:
            LOGGER.info(
                "Registered model %s not found; creating it.",
                registry_cfg.model_name,
            )
            self.client.create_registered_model(registry_cfg.model_name)

    def _resolve_stage_for_metrics(
        self,
        registry_cfg: RegistryConfig,
        metrics: Dict[str, float],
    ) -> Optional[str]:
        if not registry_cfg.stage_on_register:
            return None
        metric_name = registry_cfg.promote_on_metric
        if not metric_name:
            return registry_cfg.stage_on_register
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            LOGGER.warning(
                "Metric %s not present; skipping automatic promotion.",
                metric_name,
            )
            return None
        threshold = registry_cfg.promote_min_value
        if threshold is None or metric_value >= threshold:
            return registry_cfg.stage_on_register
        LOGGER.info(
            "Metric %s=%.4f below threshold %.4f; not promoting.",
            metric_name,
            metric_value,
            threshold,
        )
        return None

    # ------------------------------------------------------------------ #
    # Promotion helpers
    # ------------------------------------------------------------------ #

    def transition_stage(
        self,
        version: int,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """Transition a model version to the requested stage."""
        registry_cfg = self.config.registry
        if not registry_cfg.enabled:
            raise ValueError("Model registry is disabled; cannot transition stages.")

        try:
            self.client.transition_model_version_stage(
                name=registry_cfg.model_name,
                version=version,
                stage=stage,
                archive_existing=archive_existing,
            )
        except TypeError as exc:
            LOGGER.warning(
                "transition_model_version_stage does not accept 'archive_existing' parameter on this MLflow version (%s); falling back to default behaviour.",
                exc,
            )
            self.client.transition_model_version_stage(
                name=registry_cfg.model_name,
                version=version,
                stage=stage,
            )
        LOGGER.info(
            "Promoted %s version %s to stage %s (archive_existing=%s).",
            registry_cfg.model_name,
            version,
            stage,
            archive_existing,
        )
