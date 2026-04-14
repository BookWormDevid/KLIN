"""Shared S3 and env helpers for batch inspection and fake seeding."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any, cast

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_BATCH_S3_PREFIX = "klin/batch"
DEFAULT_BATCH_FILE_EXTENSIONS = ".mp4,.avi,.mov,.mkv,.wmv,.webm"
DEFAULT_S3_REGION = "us-east-1"
DEFAULT_S3_ADDRESSING_STYLE = "path"


def add_date_argument(parser: Any) -> None:
    """Attach the common batch date argument to an argparse parser."""

    parser.add_argument(
        "--date",
        required=True,
        help="Date partition in YYYY-MM-DD format.",
    )


def configure_batch_logging() -> None:
    """Configure consistent CLI logging for batch utilities."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_required_env(name: str) -> str:
    """Read one required environment variable."""

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable '{name}' is required.")
    return value


def get_optional_env(name: str, default: str) -> str:
    """Read one optional environment variable with a default."""

    return os.environ.get(name, default).strip()


def build_partition_prefix(batch_date: str, base_prefix: str) -> str:
    """Build a normalized date-partitioned S3 prefix."""

    normalized_base = base_prefix.strip().strip("/")
    return f"{normalized_base}/{batch_date}/" if normalized_base else f"{batch_date}/"


def parse_allowed_extensions(raw_value: str) -> tuple[str, ...]:
    """Parse and normalize allowed file extensions."""

    extensions = tuple(
        part.strip().lower() for part in raw_value.split(",") if part.strip()
    )
    if not extensions:
        raise ValueError("At least one allowed extension is required.")
    return extensions


def has_allowed_extension(object_key: str, allowed_extensions: tuple[str, ...]) -> bool:
    """Check whether the object key has an allowed video extension."""

    return PurePosixPath(object_key).suffix.lower() in allowed_extensions


def build_s3_uri(bucket_name: str, object_key: str) -> str:
    """Build one S3 URI."""

    return f"s3://{bucket_name}/{object_key}"


def build_s3_client() -> Any:
    """Build a boto3 S3 client from runtime environment variables."""

    addressing_style = get_optional_env(
        "S3_ADDRESSING_STYLE",
        DEFAULT_S3_ADDRESSING_STYLE,
    )
    s3_config = cast(Any, {"addressing_style": addressing_style})

    return boto3.client(
        "s3",
        endpoint_url=get_required_env("S3_ENDPOINT_URL"),
        aws_access_key_id=get_required_env("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=get_required_env("S3_SECRET_ACCESS_KEY"),
        region_name=get_optional_env("S3_REGION", DEFAULT_S3_REGION),
        config=Config(signature_version="s3v4", s3=s3_config),
    )


async def run_sync(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a blocking boto3 call in the default executor."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def ensure_bucket_exists(
    client: Any,
    bucket_name: str,
    region_name: str,
) -> None:
    """Ensure the destination bucket exists before copying seed objects."""

    try:
        await run_sync(client.head_bucket, Bucket=bucket_name)
    except ClientError as exc:
        error_code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if error_code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

        create_kwargs: dict[str, Any] = {"Bucket": bucket_name}
        if region_name and region_name != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": region_name
            }
        await run_sync(client.create_bucket, **create_kwargs)


async def list_bucket_object_keys(
    client: Any,
    *,
    bucket_name: str,
    prefix: str,
) -> list[str]:
    """List object keys in one bucket for the given prefix."""

    normalized_prefix = prefix.strip().lstrip("/")

    def _list() -> list[str]:
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=normalized_prefix):
            for item in page.get("Contents", []):
                object_key = item.get("Key")
                if not object_key or object_key.endswith("/"):
                    continue
                keys.append(str(object_key))
        return keys

    return cast(list[str], await run_sync(_list))


async def copy_object(
    client: Any,
    *,
    source_bucket: str,
    source_key: str,
    destination_bucket: str,
    destination_key: str,
) -> None:
    """Copy one object between buckets on the same S3 endpoint."""

    await run_sync(
        client.copy_object,
        CopySource={"Bucket": source_bucket, "Key": source_key},
        Bucket=destination_bucket,
        Key=destination_key,
    )
