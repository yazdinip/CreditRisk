# Data Management

Raw Kaggle files (for example `application_train.csv`) live under `data/raw/` and are **not**
committed to git. Use DVC to version and share them instead.

```bash
dvc add data/raw/application_train.csv
git add data/raw/application_train.csv.dvc .gitignore
git commit -m "Track raw training data with DVC"
```

To pull the data that another teammate pushed to the DVC remote:

```bash
dvc pull
```

Intermediate outputs (feature stores, prepared datasets, etc.) should be written to
`data/interim` or `data/processed` and also tracked via DVC if they need to be reproduced.
