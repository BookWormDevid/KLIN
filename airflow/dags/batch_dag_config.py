"""Shared configuration helpers for KLIN Docker-based batch DAGs."""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from airflow.exceptions import AirflowException
from airflow.models.variable import Variable
from airflow.sdk import BaseHook
from sqlalchemy.engine.url import make_url


DEFAULT_BATCH_S3_PREFIX = "klin/batch"
DEFAULT_BATCH_FILE_EXTENSIONS = ".mp4,.avi,.mov,.mkv,.wmv,.webm"
DEFAULT_DOCKER_NETWORK = "klin-web"
DEFAULT_DOCKER_URL = "unix:///var/run/docker.sock"
APP_DB_CONN_ID = "klin_app_db"


class CommonDockerOperatorArgs(TypedDict):
    """Typed common kwargs passed into DockerOperator."""

    image: str
    docker_url: str
    network_mode: str
    auto_remove: Literal["success"]
    mount_tmp_dir: bool
    force_pull: bool


def _read_variable(*keys: str) -> str | None:
    """Return the first non-empty Airflow Variable value from the given aliases."""

    for key in keys:
        value = str(Variable.get(key, default_var="")).strip()
        if value:
            return value
    return None


def required_variable(dag_id: str, *keys: str) -> str:
    """Load one required Airflow Variable using aliases."""

    value = _read_variable(*keys)
    if value:
        return value

    keys_list = ", ".join(f"'{key}'" for key in keys)
    raise AirflowException(
        f"One of Airflow Variables {keys_list} is required for DAG '{dag_id}'."
    )


def optional_variable(default: str, *keys: str) -> str:
    """Load one optional Airflow Variable using aliases."""

    return _read_variable(*keys) or default


def docker_url() -> str:
    """Return the Docker daemon URL used by DockerOperator."""

    return os.environ.get("DOCKER_HOST", DEFAULT_DOCKER_URL).strip()


def docker_network(host: bool) -> str:
    """Return the Docker network used for batch containers."""

    if host:
        return "host"
    return optional_variable(
        DEFAULT_DOCKER_NETWORK,
    )


def batch_image(dag_id: str) -> str:
    """Return the batch runner image name."""

    return required_variable(
        dag_id,
        "KLIN_BATCH_RUNNER_IMAGE",
        "klin_batch_runner_image",
    )


def database_url_from_connection(dag_id: str) -> str:
    """Build asyncpg DATABASE_URL from Airflow connection klin_app_db."""

    try:
        connection = BaseHook.get_connection(APP_DB_CONN_ID)
    except Exception as exc:  # pragma: no cover - depends on Airflow metastore state
        raise AirflowException(
            f"Airflow Connection '{APP_DB_CONN_ID}' is required for DAG '{dag_id}'."
        ) from exc

    url = make_url(connection.get_uri())
    asyncpg_url = url.set(drivername="postgresql+asyncpg")
    return asyncpg_url.render_as_string(hide_password=False)


def build_batch_runtime_env(dag_id: str) -> dict[str, str]:
    """Return runtime env for the main batch processing DAG."""

    return {
        "DATABASE_URL": database_url_from_connection(dag_id),
        "S3_ENDPOINT_URL": required_variable(
            dag_id,
            "S3_ENDPOINT_URL",
            "klin_batch_s3_endpoint_url",
        ),
        "S3_BUCKET_NAME": required_variable(
            dag_id,
            "S3_BUCKET_NAME",
            "klin_batch_s3_bucket_name",
        ),
        "S3_ACCESS_KEY_ID": required_variable(
            dag_id,
            "S3_ACCESS_KEY_ID",
            "klin_batch_s3_access_key_id",
        ),
        "S3_SECRET_ACCESS_KEY": required_variable(
            dag_id,
            "S3_SECRET_ACCESS_KEY",
            "klin_batch_s3_secret_access_key",
        ),
        "S3_REGION": optional_variable(
            "us-east-1",
            "S3_REGION",
            "klin_batch_s3_region",
        ),
        "S3_ADDRESSING_STYLE": optional_variable(
            "path",
            "S3_ADDRESSING_STYLE",
            "klin_batch_s3_addressing_style",
        ),
        "TRITON_GRPC_URL": required_variable(
            dag_id,
            "TRITON_GRPC_URL",
            "klin_batch_triton_grpc_url",
        ),
        "KEEP_S3_SOURCE_OBJECTS": optional_variable(
            "true",
            "KEEP_S3_SOURCE_OBJECTS",
            "klin_batch_keep_s3_source_objects",
        ),
        "KLIN_BATCH_S3_PREFIX": optional_variable(
            DEFAULT_BATCH_S3_PREFIX,
            "KLIN_BATCH_S3_PREFIX",
            "klin_batch_s3_prefix",
        ),
        "KLIN_BATCH_FILE_EXTENSIONS": optional_variable(
            DEFAULT_BATCH_FILE_EXTENSIONS,
            "KLIN_BATCH_FILE_EXTENSIONS",
            "klin_batch_file_extensions",
        ),
        "DB_CONNECT_TIMEOUT": optional_variable(
            "30",
            "DB_CONNECT_TIMEOUT",
            "klin_batch_db_connect_timeout",
        ),
        "MAX_RETRY_ATTEMPTS": optional_variable(
            "1",
            "MAX_RETRY_ATTEMPTS",
            "klin_batch_max_retry_attempts",
        ),
    }


def build_seed_runtime_env(dag_id: str, source_bucket: str) -> dict[str, str]:
    """Return runtime env for the batch seeding DAG."""

    return {
        "S3_ENDPOINT_URL": required_variable(
            dag_id,
            "S3_ENDPOINT_URL",
            "klin_batch_s3_endpoint_url",
        ),
        "S3_BUCKET_NAME": required_variable(
            dag_id,
            "S3_BUCKET_NAME",
            "klin_batch_s3_bucket_name",
        ),
        "S3_ACCESS_KEY_ID": required_variable(
            dag_id,
            "S3_ACCESS_KEY_ID",
            "klin_batch_s3_access_key_id",
        ),
        "S3_SECRET_ACCESS_KEY": required_variable(
            dag_id,
            "S3_SECRET_ACCESS_KEY",
            "klin_batch_s3_secret_access_key",
        ),
        "S3_REGION": optional_variable(
            "us-east-1",
            "S3_REGION",
            "klin_batch_s3_region",
        ),
        "S3_ADDRESSING_STYLE": optional_variable(
            "path",
            "S3_ADDRESSING_STYLE",
            "klin_batch_s3_addressing_style",
        ),
        "KLIN_BATCH_S3_PREFIX": optional_variable(
            DEFAULT_BATCH_S3_PREFIX,
            "KLIN_BATCH_S3_PREFIX",
            "klin_batch_s3_prefix",
        ),
        "KLIN_BATCH_FILE_EXTENSIONS": optional_variable(
            DEFAULT_BATCH_FILE_EXTENSIONS,
            "KLIN_BATCH_FILE_EXTENSIONS",
            "klin_batch_file_extensions",
        ),
        "KLIN_BATCH_SEED_SOURCE_BUCKET": source_bucket,
        "KLIN_BATCH_SEED_SOURCE_PREFIX": optional_variable(
            "",
            "KLIN_BATCH_SEED_SOURCE_PREFIX",
            "klin_batch_seed_source_prefix",
        ),
        "KLIN_BATCH_SEED_COUNT": optional_variable(
            "5",
            "KLIN_BATCH_SEED_COUNT",
            "klin_batch_seed_count",
        ),
    }


def common_docker_operator_args(
    dag_id: str, host_network: bool = False
) -> CommonDockerOperatorArgs:
    """Return the common DockerOperator arguments for all batch DAG tasks."""

    image = batch_image(dag_id)
    return {
        "image": image,
        "docker_url": docker_url(),
        "network_mode": docker_network(host_network),
        "auto_remove": "success",
        "mount_tmp_dir": False,
        "force_pull": True,
    }
