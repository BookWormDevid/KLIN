"""Airflow DAG for date-partitioned KLIN batch processing from UI-managed secrets."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from batch_dag_config import (
    build_batch_runtime_env,
    common_docker_operator_args,
)


DAG_ID = "klin_batch_s3"
BATCH_ENV = build_batch_runtime_env(DAG_ID)
COMMON_OPERATOR_ARGS = common_docker_operator_args(DAG_ID)
BATCH_OPERATOR_ARGS = common_docker_operator_args(DAG_ID, host_network=True)

with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2026, 4, 10),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=24),
    tags=["klin", "batch", "docker"],
) as dag:
    inspect_batch_inputs = DockerOperator(
        task_id="inspect_batch_inputs_for_date",
        command="python -m app.batch.inspect_batch_inputs --date {{ ds }}",
        private_environment=BATCH_ENV,
        **COMMON_OPERATOR_ARGS,
    )
    run_batch = DockerOperator(
        task_id="run_batch_for_date",
        command="python -m app.batch.run_batch --date {{ ds }}",
        private_environment=BATCH_ENV,
        **BATCH_OPERATOR_ARGS,
    )

    inspect_batch_inputs >> run_batch
