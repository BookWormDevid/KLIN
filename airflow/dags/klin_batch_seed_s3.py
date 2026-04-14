"""Airflow DAG for seeding daily fake batch videos when a date partition is empty."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from batch_dag_config import (
    build_seed_runtime_env,
    common_docker_operator_args,
)


DAG_ID = "klin_batch_seed_s3"
SEED_SOURCE_BUCKET = "ufc-crime-klin-dataset"
SEED_ENV = build_seed_runtime_env(DAG_ID, source_bucket=SEED_SOURCE_BUCKET)
COMMON_OPERATOR_ARGS = common_docker_operator_args(DAG_ID)


with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2026, 4, 14),
    schedule="0 1 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    tags=["klin", "batch", "seed", "docker"],
) as dag:
    inspect_seed_target_partition = DockerOperator(
        task_id="inspect_seed_target_partition_for_date",
        command="python -m app.batch.inspect_batch_inputs --date {{ ds }}",
        private_environment=SEED_ENV,
        **COMMON_OPERATOR_ARGS,
    )
    seed_batch = DockerOperator(
        task_id="seed_batch_for_date",
        command="python -m app.batch.seed_fake_batch --date {{ ds }}",
        private_environment=SEED_ENV,
        **COMMON_OPERATOR_ARGS,
    )

    inspect_seed_target_partition >> seed_batch
