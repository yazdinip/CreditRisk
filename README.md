# CreditRisk MLOps Playground

Home Credit Default Risk Kaggle competition re-imagined as an end-to-end MLOps exercise.
The goal is not to squeeze every basis point of model performance but to practice building
repeatable, observable workflows with open-source tools (DVC, MLflow, GitHub Actions, etc.).

## Repository Layout

```
.
├── configs/                # YAML configs that feed the pipelines (DVC params)
├── data/                   # Raw/interim/processed data (kept out of git, tracked via DVC)
├── docs/                   # Architecture notes, operating model, proposal summaries
├── notebooks/              # Research + EDA exports (Colab/Jupyter)
├── reports/                # Metrics, evaluation artifacts, presentation-ready plots
├── src/creditrisk/         # Re-usable Python package for pipelines
│   ├── data/               # Data loading helpers
│   ├── features/           # Feature engineering + preprocessing logic
│   ├── models/             # Model factory + evaluation utilities
│   └── pipelines/          # Orchestrated CLI entry points (train/evaluate/etc.)
├── dvc.yaml                # Stage definitions for deterministic runs
└── requirements.txt        # Reproducible environment spec
```

## Quickstart

1. **Create a virtual environment** and install dependencies.
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # or source .venv/bin/activate on macOS/Linux
   pip install -e .
   ```
2. **Fetch the Kaggle data** and place `application_train.csv` under `data/raw/`.  
   A `.dvc` pointer already exists; run `dvc pull` if a remote is configured or replace the file locally and run `dvc add data/raw/application_train.csv`.
3. **Train the baseline pipeline** (this registers a run in MLflow and writes artifacts under DVC control):
   ```bash
   dvc repro train_baseline
   # or without DVC
   python -m creditrisk.pipelines.train_baseline --config configs/baseline.yaml
   ```
4. **Inspect outputs** in `models/` (serialized pipeline) and `reports/metrics.json`.  
   Point `MLFLOW_TRACKING_URI` or edit `configs/baseline.yaml` if you prefer a remote server.

## Baseline Recipe

- Mirrors the original Colab workflow housed in `notebooks/01_home_credit_default_risk_eda.py`: drop columns with >40% missing data, remove a few high-cardinality categoricals, one-hot encode the rest, and add the per-row `missing_count` feature.
- Adds the notebook’s domain tweaks: `DAYS_EMPLOYED_ANOM`/`DAYS_EMPLOYED_REPLACED` plus missingness indicators for `EXT_SOURCE_[1-3]` and `OWN_CAR_AGE` before pruning sparse columns.
- Uses the manual feature shortlist from that notebook (`features.selected_columns` in `configs/baseline.yaml`) so training/inference always operate on the same 90+ engineered columns.
- Balances the classes exactly like the notebook: SMOTE with `sampling_strategy=0.2` followed by downsampling the majority class before fitting the XGBoost model (see `TrainingConfig` in `configs/baseline.yaml`).
- The `train_baseline` DVC stage now produces the serialized model + `reports/metrics.json`, and every run logs to the `baseline` MLflow experiment (stored locally under `mlruns/` by default).

## Tooling Highlights

- **DVC** provides deterministic DAGs (`dvc repro`) and data versioning for large CSVs.
- **MLflow** tracks parameters, metrics, and serialized models. Local runs default to the
  `mlruns/` folder but any tracking URI can be configured.
- **Modular package** (`src/creditrisk`) makes it easy to extend the pipeline with new stages
  such as feature store materialization, monitoring jobs, or deployment scripts.
- **Notebooks folder** holds the original Colab EDA / baseline model for context while keeping
  production code cleanly separated.

## Next Steps

- Wire this repo into CI (pre-commit, unit tests, `dvc repro`) and CD (GitHub Actions → SageMaker).
- Add feature store + data quality jobs (e.g., Great Expectations, Evidently).
- Stand up deployment scaffolding (FastAPI service + Dockerfile) and monitoring hooks so we can
  exercise the full MLOps lifecycle outlined in the project proposal.
