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
2. **Fetch the Kaggle data** and place the following under `data/raw/`: `application_train.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `installments_payments.csv`, `credit_card_balance.csv`, and `POS_CASH_balance.csv`.  
   A `.dvc` pointer already exists for `application_train.csv`; add/pull the remaining files with DVC so experiments stay reproducible.
3. **Train the baseline pipeline** (this registers a run in MLflow and writes metrics + plots under DVC control):
   ```bash
   dvc repro train_baseline
   # or without DVC
   python -m creditrisk.pipelines.train_baseline --config configs/baseline.yaml
   ```
4. **Inspect outputs** in `models/` (serialized pipeline), `reports/metrics.json`, and `reports/evaluation/` (ROC, PR, calibration, and confusion artifacts).  
   Point `MLFLOW_TRACKING_URI` or edit `configs/baseline.yaml` if you prefer a remote server.
5. **(Optional)** Score new data using the batch CLI or FastAPI service (see *Inference Tooling*) and run `pytest tests` to verify the helper modules.

## Baseline Recipe

- Mirrors the Colab workflow housed in `notebooks/another_copy_of_home_credit_default_risk_eda (3).py`: drop columns with >40% missing data, remove a few high-cardinality categoricals, one-hot encode the rest, and add the per-row `missing_count` feature.
- Adds the notebook’s domain tweaks: age/tenure features (`AGE_YEARS`, `EMPLOYED_YEARS`, `EMPLOYMENT_YEARS_TO_AGE`), `DAYS_EMPLOYED_ANOM`/`DAYS_EMPLOYED_REPLACED`, and missingness indicators for `EXT_SOURCE_[1-3]` + `OWN_CAR_AGE` before pruning sparse columns.
- Replays the DuckDB feature store SQL bit-for-bit (application enrichment + bureau, bureau_balance, previous_application, installments_payments, credit_card_balance, and POS_CASH aggregates) via `creditrisk.features.feature_store`, so notebooks and pipelines share identical engineered columns.
- Generates the same ratio/count features defined in SQL (`PAYMENT_RATE`, `CREDIT_TO_INCOME`, `DOC_COUNT`, `CONTACT_COUNT`, `ADDR_MISMATCH_SUM`, etc.) so downstream stages see the exact engineered signals without running DuckDB inline.
- Uses the curated feature shortlist from the notebook (`features.selected_columns` in `configs/baseline.yaml`) so training, batch inference, and FastAPI serving all operate on the same 160+ engineered columns.
- Balances the classes exactly like the notebook: SMOTE with `sampling_strategy=0.2` followed by downsampling the majority class before fitting the XGBoost model (see `TrainingConfig` in `configs/baseline.yaml`).
- Training logs metrics + plots to `reports/` and the `baseline` MLflow experiment while persisting the full sklearn pipeline for downstream reuse.

## Tooling Highlights

- **DVC** provides deterministic DAGs (`dvc repro`) and data versioning for large CSVs.
- **MLflow** tracks parameters, metrics, and serialized models. Local runs default to the
  `mlruns/` folder but any tracking URI can be configured.
- **Evaluation artifacts** (`creditrisk.utils.evaluation`) automatically emit ROC, PR, calibration, and confusion-matrix plots to `reports/evaluation/` for lightweight QA without the MLflow UI.
- **Batch inference CLI** (`python -m creditrisk.pipelines.batch_predict`) loads any saved pipeline, applies the configured decision threshold, and writes predictions + probabilities for downstream consumers.
- **FastAPI service** (`uvicorn creditrisk.serve.api:app --reload`) exposes `/health` and `/predict`, sharing the same thresholding/output helpers as the batch CLI for parity between offline and online scoring.
- **Tests** (`pytest tests`) cover the evaluation helper, batch CLI, and API contract so feature-store changes don’t silently break inference.
- **Modular package** (`src/creditrisk`) makes it easy to extend the pipeline with new stages
  such as feature store materialization, monitoring jobs, or deployment scripts.
- **Notebooks folder** holds the original Colab EDA / baseline model for context while keeping
  production code cleanly separated.

## Inference Tooling

- **Batch scoring**  
  ```bash
  python -m creditrisk.pipelines.batch_predict \
    --config configs/baseline.yaml \
    --input-csv data/raw/application_test.csv \
    --output-csv reports/batch_predictions.csv
  ```
  (`--threshold` and `--model-path` override the defaults; outputs include `prediction` + `probability`.)
- **FastAPI service**  
  ```bash
  uvicorn creditrisk.serve.api:app --port 8000 --reload
  ```
  POST `{"records": [...]}` to `/predict` for scoring; include `threshold` in the body to override the config’s decision boundary.

## Next Steps

- Wire this repo into CI (pre-commit, unit tests, `dvc repro`) and CD (GitHub Actions → SageMaker).
- Add automated data-quality checks (Great Expectations, Pandera) ahead of the DuckDB feature-store build and plug Evidently drift reports into the evaluation artifacts.
- Containerize the FastAPI service + batch CLI, promote MLflow models via CI, and hook the monitoring plan from the proposal into production deployments.

