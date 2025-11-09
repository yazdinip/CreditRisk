"""Configuration helpers for the CreditRisk project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    """Filesystem locations for generated artifacts."""

    model_dir: Path = Path("models")
    model_filename: str = "baseline_model.joblib"
    metrics_file: Path = Path("reports/metrics.json")
    reports_dir: Path = Path("reports")
    feature_store_path: Path = Path("data/processed/feature_store.parquet")
    train_set_path: Path = Path("data/processed/train.parquet")
    test_set_path: Path = Path("data/processed/test.parquet")
    lineage_file: Optional[Path] = None

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.metrics_file = Path(self.metrics_file)
        self.reports_dir = Path(self.reports_dir)
        self.feature_store_path = Path(self.feature_store_path)
        self.train_set_path = Path(self.train_set_path)
        self.test_set_path = Path(self.test_set_path)
        if self.lineage_file is None:
            self.lineage_file = self.reports_dir / "data_lineage.json"
        else:
            self.lineage_file = Path(self.lineage_file)

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir) / self.model_filename

    @property
    def evaluation_dir(self) -> Path:
        return Path(self.reports_dir) / "evaluation"


@dataclass
class DataConfig:
    """Data source information."""

    raw_path: Path
    target_column: str = "TARGET"
    drop_columns: List[str] = field(default_factory=lambda: ["SK_ID_CURR"])
    sample_rows: Optional[int] = None
    stratify: bool = True
    entity_id_column: str = "SK_ID_CURR"
    bureau_path: Optional[Path] = None
    bureau_balance_path: Optional[Path] = None
    previous_application_path: Optional[Path] = None
    installments_payments_path: Optional[Path] = None
    credit_card_balance_path: Optional[Path] = None
    pos_cash_balance_path: Optional[Path] = None


@dataclass
class FeaturesConfig:
    """Feature engineering directives mirroring the original notebook preprocessing."""

    categorical: Optional[List[str]] = None
    numerical: Optional[List[str]] = None
    drop_columns: List[str] = field(default_factory=list)
    add_missing_count: bool = True
    missing_threshold: float = 0.4  # drop columns with >40% missing values
    categorical_drop: List[str] = field(
        default_factory=lambda: ["NAME_TYPE_SUITE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    )
    add_days_employed_anomaly: bool = True
    days_employed_anomaly_value: int = 365243
    missing_indicator_columns: List[str] = field(
        default_factory=lambda: ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OWN_CAR_AGE"]
    )
    add_ratio_features: bool = True
    add_count_features: bool = True
    ratio_feature_eps: float = 1e-6
    selected_columns: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Model training hyperparameters."""

    test_size: float = 0.2
    random_state: int = 7
    class_weight: Optional[str] = "balanced"
    n_jobs: int = -1
    use_gpu: bool = False
    early_stopping_rounds: Optional[int] = None
    use_smote: bool = True
    smote_sampling_strategy: float = 0.2
    downsample_majority: bool = True


@dataclass
class ModelConfig:
    """Classifier selection."""

    type: str = "xgboost"
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        }
    )


@dataclass
class TrackingConfig:
    """Experiment tracking settings."""

    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "baseline"
    run_name: str = "baseline_xgboost"
    tags: Dict[str, str] = field(
        default_factory=lambda: {"project": "credit-risk", "stage": "baseline"}
    )


@dataclass
class InferenceConfig:
    """Inference-time options."""

    decision_threshold: float = 0.5


@dataclass
class RegistryConfig:
    """Model registry behavior."""

    enabled: bool = False
    model_name: str = "CreditRiskBaseline"
    stage_on_register: Optional[str] = "Staging"
    promote_on_metric: Optional[str] = "roc_auc"
    promote_min_value: float = 0.75
    archive_existing_on_promote: bool = True


@dataclass
class ValidationConfig:
    """Validation and data-contract toggles."""

    enabled: bool = True
    enforce_raw_contracts: bool = True
    enforce_feature_store_contract: bool = True
    enforce_split_contracts: bool = True
    enforce_model_io_contracts: bool = True


@dataclass
class TestingConfig:
    """Post-training testing thresholds and expectations."""

    enabled: bool = True
    min_metrics: Dict[str, float] = field(
        default_factory=lambda: {
            "roc_auc": 0.75,
            "normalized_gini": 0.50,
        }
    )
    evaluation_artifacts: List[str] = field(
        default_factory=lambda: [
            "confusion_matrix.png",
            "confusion_matrix.json",
            "roc_curve.png",
            "precision_recall_curve.png",
            "calibration_curve.png",
        ]
    )
    require_mlflow: bool = True
    mlflow_metric_tolerance: float = 1e-4
    report_filename: str = "validation_summary.json"


@dataclass
class Config:
    """Container for all configuration sections."""

    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        with open(path, "r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}

        def build(section_cls, key, default=None):
            section_data = payload.get(key, {})
            if default and not section_data:
                return default
            return section_cls(**section_data)

        return cls(
            paths=build(PathsConfig, "paths", PathsConfig()),
            data=build(DataConfig, "data"),
            features=build(FeaturesConfig, "features", FeaturesConfig()),
            training=build(TrainingConfig, "training", TrainingConfig()),
            model=build(ModelConfig, "model", ModelConfig()),
            tracking=build(TrackingConfig, "tracking", TrackingConfig()),
            inference=build(InferenceConfig, "inference", InferenceConfig()),
            registry=build(RegistryConfig, "registry", RegistryConfig()),
            validation=build(ValidationConfig, "validation", ValidationConfig()),
            testing=build(TestingConfig, "testing", TestingConfig()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": {
                "model_dir": str(self.paths.model_dir),
                "model_filename": self.paths.model_filename,
                "metrics_file": str(self.paths.metrics_file),
                "reports_dir": str(self.paths.reports_dir),
                "feature_store_path": str(self.paths.feature_store_path),
                "train_set_path": str(self.paths.train_set_path),
                "test_set_path": str(self.paths.test_set_path),
                "lineage_file": str(self.paths.lineage_file),
            },
            "data": {
                "raw_path": str(self.data.raw_path),
                "target_column": self.data.target_column,
                "drop_columns": self.data.drop_columns,
                "sample_rows": self.data.sample_rows,
                "stratify": self.data.stratify,
                "entity_id_column": self.data.entity_id_column,
                "bureau_path": str(self.data.bureau_path) if self.data.bureau_path else None,
                "bureau_balance_path": str(self.data.bureau_balance_path)
                if self.data.bureau_balance_path
                else None,
                "previous_application_path": str(self.data.previous_application_path)
                if self.data.previous_application_path
                else None,
                "installments_payments_path": str(self.data.installments_payments_path)
                if self.data.installments_payments_path
                else None,
                "credit_card_balance_path": str(self.data.credit_card_balance_path)
                if self.data.credit_card_balance_path
                else None,
                "pos_cash_balance_path": str(self.data.pos_cash_balance_path)
                if self.data.pos_cash_balance_path
                else None,
            },
            "features": {
                "categorical": self.features.categorical,
                "numerical": self.features.numerical,
                "drop_columns": self.features.drop_columns,
                "add_missing_count": self.features.add_missing_count,
                "missing_threshold": self.features.missing_threshold,
                "categorical_drop": self.features.categorical_drop,
                "add_days_employed_anomaly": self.features.add_days_employed_anomaly,
                "days_employed_anomaly_value": self.features.days_employed_anomaly_value,
                "missing_indicator_columns": self.features.missing_indicator_columns,
                "add_ratio_features": self.features.add_ratio_features,
                "add_count_features": self.features.add_count_features,
                "ratio_feature_eps": self.features.ratio_feature_eps,
                "selected_columns": self.features.selected_columns,
            },
            "training": {
                "test_size": self.training.test_size,
                "random_state": self.training.random_state,
                "class_weight": self.training.class_weight,
                "n_jobs": self.training.n_jobs,
                "use_gpu": self.training.use_gpu,
                "early_stopping_rounds": self.training.early_stopping_rounds,
                "use_smote": self.training.use_smote,
                "smote_sampling_strategy": self.training.smote_sampling_strategy,
                "downsample_majority": self.training.downsample_majority,
            },
            "model": {
                "type": self.model.type,
                "params": self.model.params,
            },
            "tracking": {
                "enabled": self.tracking.enabled,
                "tracking_uri": self.tracking.tracking_uri,
                "experiment_name": self.tracking.experiment_name,
                "run_name": self.tracking.run_name,
                "tags": self.tracking.tags,
            },
            "inference": {
                "decision_threshold": self.inference.decision_threshold,
            },
            "registry": {
                "enabled": self.registry.enabled,
                "model_name": self.registry.model_name,
                "stage_on_register": self.registry.stage_on_register,
                "promote_on_metric": self.registry.promote_on_metric,
                "promote_min_value": self.registry.promote_min_value,
                "archive_existing_on_promote": self.registry.archive_existing_on_promote,
            },
            "validation": {
                "enabled": self.validation.enabled,
                "enforce_raw_contracts": self.validation.enforce_raw_contracts,
                "enforce_feature_store_contract": self.validation.enforce_feature_store_contract,
                "enforce_split_contracts": self.validation.enforce_split_contracts,
                "enforce_model_io_contracts": self.validation.enforce_model_io_contracts,
            },
            "testing": {
                "enabled": self.testing.enabled,
                "min_metrics": self.testing.min_metrics,
                "evaluation_artifacts": self.testing.evaluation_artifacts,
                "require_mlflow": self.testing.require_mlflow,
                "mlflow_metric_tolerance": self.testing.mlflow_metric_tolerance,
                "report_filename": self.testing.report_filename,
            },
        }
