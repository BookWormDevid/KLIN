"""Airflow DAG for date-partitioned KLIN batch processing from UI-managed secrets."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.exceptions import AirflowException
from airflow.models.dag import DAG
from airflow.models.variable import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import BaseHook


DAG_ID = "klin_batch_s3"
DOCKER_CONN_ID = "klin_batch_docker"


def _get_required_variable(*keys: str) -> str:
    """Load the first non-empty Airflow variable from the provided aliases."""

    for key in keys:
        try:
            value = str(Variable.get(key)).strip()
        except KeyError:
            continue
        if value:
            return value

    keys_list = ", ".join(f"'{key}'" for key in keys)
    raise AirflowException(
        f"One of Airflow Variables {keys_list} is required for DAG '{DAG_ID}'."
    )


def _get_optional_variable(default: str, *keys: str) -> str:
    """Load the first non-empty Airflow variable from aliases or return default."""

    for key in keys:
        value = str(Variable.get(key, default_var="")).strip()
        if value:
            return value
    return default


def _require_connection(conn_id: str) -> str:
    """Validate that the configured Docker connection exists in Airflow."""

    try:
        BaseHook.get_connection(conn_id)
    except Exception as exc:  # pragma: no cover - depends on Airflow metastore state
        raise AirflowException(
            f"Airflow Connection '{conn_id}' is required for DAG '{DAG_ID}'."
        ) from exc
    return conn_id


def _build_batch_environment() -> dict[str, str]:
    """Assemble the hidden runtime environment for the batch container."""

    return {
        "DATABASE_URL": _get_required_variable(
            "DATABASE_URL",
            "klin_batch_database_url",
        ),
        "S3_ENDPOINT_URL": _get_required_variable(
            "S3_ENDPOINT_URL",
            "klin_batch_s3_endpoint_url",
        ),
        "S3_BUCKET_NAME": _get_required_variable(
            "S3_BUCKET_NAME",
            "klin_batch_s3_bucket_name",
        ),
        "S3_ACCESS_KEY_ID": _get_required_variable(
            "S3_ACCESS_KEY_ID",
            "klin_batch_s3_access_key_id",
        ),
        "S3_SECRET_ACCESS_KEY": _get_required_variable(
            "S3_SECRET_ACCESS_KEY", "klin_batch_s3_secret_access_key"
        ),
        "S3_REGION": _get_optional_variable(
            "us-east-1",
            "S3_REGION",
            "klin_batch_s3_region",
        ),
        "S3_ADDRESSING_STYLE": _get_optional_variable(
            "path",
            "S3_ADDRESSING_STYLE",
            "klin_batch_s3_addressing_style",
        ),
        "TRITON_GRPC_URL": _get_required_variable(
            "TRITON_GRPC_URL",
            "klin_batch_triton_grpc_url",
        ),
        "KEEP_S3_SOURCE_OBJECTS": _get_optional_variable(
            "true",
            "klin_batch_keep_s3_source_objects",
        ),
        "KLIN_BATCH_S3_PREFIX": _get_optional_variable(
            "klin/batch",
            "klin_batch_s3_prefix",
        ),
        "KLIN_BATCH_FILE_EXTENSIONS": _get_optional_variable(
            ".mp4,.avi,.mov,.mkv,.wmv,.webm",
            "klin_batch_file_extensions",
        ),
        "DB_CONNECT_TIMEOUT": _get_optional_variable(
            "30",
            "klin_batch_db_connect_timeout",
        ),
        "MAX_RETRY_ATTEMPTS": _get_optional_variable(
            "1",
            "klin_batch_max_retry_attempts",
        ),
    }


BATCH_ENV = _build_batch_environment()
BATCH_IMAGE = _get_required_variable(
    "KLIN_BATCH_RUNNER_IMAGE",
    "klin_batch_runner_image",
)
DOCKER_NETWORK = _get_optional_variable("klin-web", "klin_batch_docker_network")
VALIDATED_DOCKER_CONN_ID = _require_connection(DOCKER_CONN_ID)


with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2026, 4, 10),
    schedule="0 1 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=24),
    tags=["klin", "batch", "docker"],
) as dag:
    inspect_batch_inputs = DockerOperator(
        task_id="inspect_batch_inputs_for_date",
        image=BATCH_IMAGE,
        command="python -m app.batch.inspect_batch_inputs --date {{ ds }}",
        docker_conn_id=VALIDATED_DOCKER_CONN_ID,
        network_mode=DOCKER_NETWORK,
        auto_remove="success",
        mount_tmp_dir=False,
        force_pull=False,
        private_environment=BATCH_ENV,
    )
    run_batch = DockerOperator(
        task_id="run_batch_for_date",
        image=BATCH_IMAGE,
        command="python -m app.batch.run_batch --date {{ ds }}",
        docker_conn_id=VALIDATED_DOCKER_CONN_ID,
        network_mode=DOCKER_NETWORK,
        auto_remove="success",
        mount_tmp_dir=False,
        force_pull=False,
        private_environment=BATCH_ENV,
    )

    inspect_batch_inputs >> run_batch
