import os
from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator


BATCH_ENV_KEYS = (
    "DATABASE_URL",
    "RABBIT_URL",
    "S3_ENDPOINT_URL",
    "S3_BUCKET_NAME",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_REGION",
    "S3_ADDRESSING_STYLE",
    "TRITON_GRPC_URL",
    "KEEP_S3_SOURCE_OBJECTS",
    "KLIN_BATCH_S3_PREFIX",
    "KLIN_BATCH_FILE_EXTENSIONS",
    "MAX_RETRY_ATTEMPTS",
)

BATCH_ENV = {
    key: value for key in BATCH_ENV_KEYS if (value := os.environ.get(key)) is not None
}


with DAG(
    dag_id="klin_batch_s3",
    start_date=datetime(2026, 4, 10),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=24),
    tags=["klin", "batch", "docker"],
) as dag:
    run_batch = DockerOperator(
        task_id="run_batch_for_date",
        image=os.environ.get("KLIN_BATCH_RUNNER_IMAGE", "klin-batch-runner:latest"),
        command="python -m app.batch.run_batch --date {{ ds }}",
        docker_url=os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
        network_mode="klin-web",
        auto_remove="success",
        mount_tmp_dir=False,
        force_pull=False,
        private_environment=BATCH_ENV,
    )
