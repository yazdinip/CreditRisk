"""Airflow DAG that orchestrates the CreditRisk pipeline with canary validation."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "creditrisk",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ml-ops@creditrisk.example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="creditrisk_pipeline",
    description="End-to-end CreditRisk pipeline with canary validation.",
    default_args=default_args,
    schedule_interval="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    repo_dir = "{{ var.value.creditrisk_repo | default('/opt/creditrisk') }}"
    base_env = "source /opt/venv/bin/activate"

    ingest = BashOperator(
        task_id="ingest_data",
        bash_command=f"cd {repo_dir} && {base_env} && python -m creditrisk.pipelines.ingest_data",
    )

    build = BashOperator(
        task_id="build_feature_store",
        bash_command=f"cd {repo_dir} && {base_env} && dvc repro build_feature_store",
    )

    split = BashOperator(
        task_id="split_data",
        bash_command=f"cd {repo_dir} && {base_env} && dvc repro split_data",
    )

    train = BashOperator(
        task_id="train_creditrisk_pd",
        bash_command=(
            f"cd {repo_dir} && {base_env} && "
            "python -m creditrisk.pipelines.train_creditrisk_pd --config configs/creditrisk_pd.yaml"
        ),
    )

    test = BashOperator(
        task_id="test_model",
        bash_command=f"cd {repo_dir} && {base_env} && dvc repro test_model",
    )

    validate = BashOperator(
        task_id="post_training_validation",
        bash_command=f"cd {repo_dir} && {base_env} && dvc repro validate_model",
    )

    monitor = BashOperator(
        task_id="monitor_drift",
        bash_command=f"cd {repo_dir} && {base_env} && dvc repro monitor_drift",
    )

    production_monitor = BashOperator(
        task_id="production_monitor",
        bash_command=(
            f"cd {repo_dir} && {base_env} && "
            "python -m creditrisk.monitoring.production "
            "--config configs/creditrisk_pd.yaml "
            "--current {{ var.value.production_dataset_path | default('data/production/current.parquet') }} "
            "--publish-metrics"
        ),
    )

    canary = BashOperator(
        task_id="canary_validation",
        bash_command=(
            f"cd {repo_dir} && {base_env} && "
            "python -m creditrisk.pipelines.canary_validation "
            "--config configs/creditrisk_pd.yaml "
            "--production-model {{ var.value.production_model_path }} "
            "--candidate-model models/creditrisk_pd_model.joblib "
            "--dataset data/processed/test.parquet "
            "--max-metric-delta {{ var.value.canary_max_delta | default('0.02') }}"
        ),
    )

    ingest >> build >> split >> train >> test >> validate
    validate >> monitor
    validate >> canary
    monitor >> production_monitor
