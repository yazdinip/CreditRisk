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
   pip install -r requirements.txt
   ```
2. **Fetch the Kaggle data** and place `application_train.csv` under `data/raw/`.  
   Optionally track it with DVC so teammates can `dvc pull`.
3. **Train the baseline pipeline** (this registers a run in MLflow and writes artifacts):
   ```bash
   dvc repro train_baseline
   # or without DVC
   python -m creditrisk.pipelines.train_baseline --config configs/baseline.yaml
   ```
4. **Inspect outputs** in `models/` (serialized pipeline) and `reports/metrics.json`.  
   Point `MLFLOW_TRACKING_URI` or edit `configs/baseline.yaml` if you prefer a remote server.

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
