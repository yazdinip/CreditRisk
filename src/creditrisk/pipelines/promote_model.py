"""CLI helper to promote MLflow model versions."""

from __future__ import annotations

import argparse
import logging

from creditrisk.config import Config
from creditrisk.mlops.registry import ModelRegistryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote MLflow model versions between stages.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the configuration YAML.",
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version to promote.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Target MLflow stage (e.g., Staging, Production).",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        default=False,
        help="Archive previously deployed versions in the target stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    registry_manager = ModelRegistryManager(config)
    registry_manager.transition_stage(
        version=args.version,
        stage=args.stage,
        archive_existing=args.archive_existing,
    )
    LOGGER.info(
        "Model %s version %s promoted to %s.",
        config.registry.model_name,
        args.version,
        args.stage,
    )


if __name__ == "__main__":
    main()
