"""CLI entry-point for building the engineered feature store."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from creditrisk.config import Config
from creditrisk.pipelines.data_workflow import build_feature_store_frame, save_dataframe
from creditrisk.validation import ValidationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Home Credit feature store and persist it to parquet.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/creditrisk_pd.yaml",
        help="Path to the experiment configuration YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output location for the feature store parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    output_path = Path(args.output) if args.output else config.paths.feature_store_path
    validator = ValidationRunner(config.validation)
    feature_store_df = build_feature_store_frame(config, validator=validator)
    save_dataframe(feature_store_df, output_path)
    LOGGER.info("Feature store persisted to %s", output_path)


if __name__ == "__main__":
    main()
