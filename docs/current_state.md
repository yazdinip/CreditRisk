# Current Repo State

## From Notebook to Package

- The original Colab export (`notebooks/01_home_credit_default_risk_eda.py`) performed column-pruning (drop >40 % missing, remove `NAME_TYPE_SUITE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`), added a row-level `missing_count`, created `DAYS_EMPLOYED_ANOM`/`DAYS_EMPLOYED_REPLACED` plus `EXT_SOURCE_[1-3]` & `OWN_CAR_AGE` missing indicators, one-hot encoded categoricals, selected a fixed list of ~100 columns, balanced with SMOTE + majority downsampling, scaled via `StandardScaler`, and trained an `XGBClassifier`.
- That exact flow is now codified under `src/creditrisk/`:
  - `features/preprocess.py` mirrors the cleaning and feature engineering (missing count, sparsity filter, categorical drops, get_dummies, median fill, column selection).
  - `models/baseline.py` reproduces the SMOTE + downsampling sequence and builds the `ColumnSubsetter → StandardScaler → model` pipeline.
  - `pipelines/train_baseline.py` orchestrates loading `application_train.csv`, preprocessing, balancing, training, evaluation, and artifact logging.
- Configuration lives in `configs/baseline.yaml`, so the feature list, missingness threshold, SMOTE settings, etc. stay version-controlled.

## DVC Integration

- The repo is initialized as a DVC project (`.dvc/`, `.dvcignore`). The Kaggle CSV is tracked via `data/raw/application_train.csv.dvc`, keeping the large file out of git but fully reproducible.
- `dvc.yaml` defines the `train_baseline` stage that depends on the config + source files + raw data and produces `models/baseline_model.joblib` and `reports/metrics.json`.
- Running `dvc repro train_baseline` executes the packaged trainer, regenerates artifacts when inputs change, and writes the hashes to `dvc.lock`. Generated artifacts are ignored by git and instead pulled/pushed through DVC remotes when configured.

## MLflow Integration

- `log_with_mlflow` in `src/creditrisk/pipelines/train_baseline.py` logs parameters, metrics (accuracy/precision/recall/F1/ROC‑AUC), tags, and the serialized sklearn pipeline to the `baseline` experiment.
- By default runs appear under `mlruns/` (see `mlruns/459041177692104927`). Update the `tracking_uri` in `configs/baseline.yaml` or set `MLFLOW_TRACKING_URI` to point at a remote server.
- Each DVC-triggered training run automatically produces an MLflow run, giving a consistent lineage between data hash, code version, and logged metrics/artifacts.
