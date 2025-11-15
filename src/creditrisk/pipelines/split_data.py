"""Split the feature store into train/test sets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from creditrisk.config import Config
from creditrisk.pipelines.data_workflow import load_dataframe, save_dataframe, split_feature_store
from creditrisk.validation import ValidationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the engineered feature store.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/creditrisk_pd.yaml",
        help="Path to the configuration YAML.",
    )
    parser.add_argument(
        "--feature-store",
        type=str,
        default=None,
        help="Optional override for the feature store parquet location.",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default=None,
        help="Optional override for the training split parquet path.",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default=None,
        help="Optional override for the test split parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    validator = ValidationRunner(config.validation)

    feature_store_path = (
        Path(args.feature_store) if args.feature_store else config.paths.feature_store_path
    )
    feature_store_df = load_dataframe(feature_store_path)
    validator.validate_feature_store(
        feature_store_df,
        target_column=config.data.target_column,
        required_feature_columns=config.features.selected_columns,
    )

    train_df, test_df = split_feature_store(feature_store_df, config)
    validator.validate_split(
        train_df,
        target_column=config.data.target_column,
        entity_column=config.data.entity_id_column,
        split_name="train",
    )
    validator.validate_split(
        test_df,
        target_column=config.data.target_column,
        entity_column=config.data.entity_id_column,
        split_name="test",
    )
    validator.validate_split_pair(train_df, test_df, config.data.entity_id_column)

    train_path = Path(args.train_output) if args.train_output else config.paths.train_set_path
    test_path = Path(args.test_output) if args.test_output else config.paths.test_set_path
    save_dataframe(train_df, train_path)
    save_dataframe(test_df, test_path)
    LOGGER.info("Train/test splits saved to %s and %s", train_path, test_path)


if __name__ == "__main__":
    main()
