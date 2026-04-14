"""Seed date-partitioned batch inputs from a dataset bucket when needed."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from app.batch.s3_batch_support import (
    DEFAULT_BATCH_FILE_EXTENSIONS,
    DEFAULT_BATCH_S3_PREFIX,
    DEFAULT_S3_REGION,
    add_date_argument,
    build_partition_prefix,
    build_s3_client,
    build_s3_uri,
    configure_batch_logging,
    copy_object,
    ensure_bucket_exists,
    get_optional_env,
    get_required_env,
    has_allowed_extension,
    list_bucket_object_keys,
    parse_allowed_extensions,
)


logger = logging.getLogger(__name__)

DEFAULT_SEED_SOURCE_BUCKET = "ufc-crime-klin-dataset"
DEFAULT_SEED_SOURCE_PREFIX = ""
DEFAULT_SEED_COUNT = 5


@dataclass(frozen=True)
class SeedBatchConfig:
    """Resolved runtime configuration for one fake batch seed run."""

    target_bucket: str
    region_name: str
    batch_prefix: str
    allowed_extensions: tuple[str, ...]
    source_bucket: str
    source_prefix: str
    sample_size: int


def parse_args() -> argparse.Namespace:
    """Parse one seed run from the command line."""

    parser = argparse.ArgumentParser(
        description="Seed fake KLIN batch inputs from a dataset bucket."
    )
    add_date_argument(parser)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Optional override for how many random videos to seed.",
    )
    parser.add_argument(
        "--source-bucket",
        default="",
        help="Optional override for the source dataset bucket.",
    )
    parser.add_argument(
        "--source-prefix",
        default="",
        help="Optional override for the source dataset prefix.",
    )
    return parser.parse_args()


def compute_seed_value(
    *,
    batch_date: str,
    source_bucket: str,
    source_prefix: str,
    sample_size: int,
) -> int:
    """Build a deterministic seed for one daily fake batch."""

    digest = hashlib.sha256(
        f"{batch_date}:{source_bucket}:{source_prefix}:{sample_size}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def select_source_keys(
    source_keys: list[str],
    *,
    sample_size: int,
    seed_value: int,
) -> list[str]:
    """Select a deterministic random sample from the source dataset."""

    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    normalized_keys = sorted(source_keys)
    if not normalized_keys:
        return []

    rng = random.Random(seed_value)
    selection_size = min(sample_size, len(normalized_keys))
    return rng.sample(normalized_keys, k=selection_size)


def build_destination_key(
    *,
    batch_date: str,
    base_prefix: str,
    source_key: str,
    ordinal: int,
) -> str:
    """Build a collision-resistant destination key for one copied seed object."""

    source_hash = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
    file_name = PurePosixPath(source_key).name
    target_prefix = build_partition_prefix(batch_date, base_prefix)
    return f"{target_prefix}seed-{ordinal:03d}-{source_hash}-{file_name}"


def _build_seed_config(args: argparse.Namespace) -> SeedBatchConfig:
    """Resolve one normalized seed runtime config from env and CLI args."""

    target_bucket = get_required_env("S3_BUCKET_NAME")
    region_name = get_optional_env("S3_REGION", DEFAULT_S3_REGION)
    batch_prefix = get_optional_env("KLIN_BATCH_S3_PREFIX", DEFAULT_BATCH_S3_PREFIX)
    allowed_extensions = parse_allowed_extensions(
        get_optional_env(
            "KLIN_BATCH_FILE_EXTENSIONS",
            DEFAULT_BATCH_FILE_EXTENSIONS,
        )
    )
    source_bucket = args.source_bucket.strip() or get_optional_env(
        "KLIN_BATCH_SEED_SOURCE_BUCKET",
        DEFAULT_SEED_SOURCE_BUCKET,
    )
    source_prefix = (
        (
            args.source_prefix.strip()
            or get_optional_env(
                "KLIN_BATCH_SEED_SOURCE_PREFIX",
                DEFAULT_SEED_SOURCE_PREFIX,
            )
        )
        .strip()
        .strip("/")
    )
    sample_size = args.count or int(
        get_optional_env("KLIN_BATCH_SEED_COUNT", str(DEFAULT_SEED_COUNT))
    )

    return SeedBatchConfig(
        target_bucket=target_bucket,
        region_name=region_name,
        batch_prefix=batch_prefix,
        allowed_extensions=allowed_extensions,
        source_bucket=source_bucket,
        source_prefix=source_prefix,
        sample_size=sample_size,
    )


async def _copy_selected_seed_objects(
    *,
    client: Any,
    args: argparse.Namespace,
    config: SeedBatchConfig,
    source_keys: list[str],
    target_prefix: str,
) -> tuple[int, list[str], list[str]]:
    """Copy the deterministic selection of source objects into the batch partition."""

    seed_value = compute_seed_value(
        batch_date=args.date,
        source_bucket=config.source_bucket,
        source_prefix=config.source_prefix,
        sample_size=config.sample_size,
    )
    selected_source_keys = select_source_keys(
        source_keys,
        sample_size=config.sample_size,
        seed_value=seed_value,
    )
    copied_destinations: list[str] = []

    logger.info(
        "Seeding fake batch data with seed=%d source_bucket=%s source_prefix=%s "
        "target_bucket=%s target_prefix=%s available=%d selected=%d",
        seed_value,
        config.source_bucket,
        config.source_prefix,
        config.target_bucket,
        target_prefix,
        len(source_keys),
        len(selected_source_keys),
    )

    for ordinal, source_key in enumerate(selected_source_keys, start=1):
        destination_key = build_destination_key(
            batch_date=args.date,
            base_prefix=config.batch_prefix,
            source_key=source_key,
            ordinal=ordinal,
        )
        destination_uri = build_s3_uri(config.target_bucket, destination_key)
        source_uri = build_s3_uri(config.source_bucket, source_key)
        logger.info(
            "Copying seed object source=%s destination=%s",
            source_uri,
            destination_uri,
        )
        await copy_object(
            client,
            source_bucket=config.source_bucket,
            source_key=source_key,
            destination_bucket=config.target_bucket,
            destination_key=destination_key,
        )
        copied_destinations.append(destination_uri)

    return seed_value, selected_source_keys, copied_destinations


async def process_seed_batch(args: argparse.Namespace) -> int:
    """Seed fake batch inputs only when the destination partition is empty."""

    client = build_s3_client()
    config = _build_seed_config(args)
    target_prefix = build_partition_prefix(args.date, config.batch_prefix)

    await ensure_bucket_exists(client, config.target_bucket, config.region_name)

    existing_target_keys = await list_bucket_object_keys(
        client,
        bucket_name=config.target_bucket,
        prefix=target_prefix,
    )
    existing_target_keys = sorted(
        key
        for key in existing_target_keys
        if has_allowed_extension(key, config.allowed_extensions)
    )

    if existing_target_keys:
        logger.info(
            "Skipping seed because destination already contains %d eligible object(s). "
            "bucket=%s prefix=%s",
            len(existing_target_keys),
            config.target_bucket,
            target_prefix,
        )
        for object_key in existing_target_keys:
            logger.info(
                "Existing destination object: %s",
                build_s3_uri(config.target_bucket, object_key),
            )

        print(
            json.dumps(
                {
                    "date": args.date,
                    "seeded": False,
                    "reason": "existing_destination_data",
                    "target_bucket": config.target_bucket,
                    "target_prefix": target_prefix,
                    "existing_objects": [
                        build_s3_uri(config.target_bucket, object_key)
                        for object_key in existing_target_keys
                    ],
                },
                ensure_ascii=True,
            )
        )
        return 0

    source_keys = await list_bucket_object_keys(
        client,
        bucket_name=config.source_bucket,
        prefix=config.source_prefix,
    )
    source_keys = [
        key
        for key in source_keys
        if has_allowed_extension(key, config.allowed_extensions)
    ]

    if not source_keys:
        logger.error(
            "No eligible source objects found for fake seed. bucket=%s prefix=%s",
            config.source_bucket,
            config.source_prefix,
        )
        print(
            json.dumps(
                {
                    "date": args.date,
                    "seeded": False,
                    "reason": "no_source_objects",
                    "source_bucket": config.source_bucket,
                    "source_prefix": config.source_prefix,
                },
                ensure_ascii=True,
            )
        )
        return 1

    (
        seed_value,
        selected_source_keys,
        copied_destinations,
    ) = await _copy_selected_seed_objects(
        client=client,
        args=args,
        config=config,
        source_keys=source_keys,
        target_prefix=target_prefix,
    )

    print(
        json.dumps(
            {
                "date": args.date,
                "seeded": True,
                "seed_value": seed_value,
                "source_bucket": config.source_bucket,
                "source_prefix": config.source_prefix,
                "target_bucket": config.target_bucket,
                "target_prefix": target_prefix,
                "selected_sources": [
                    build_s3_uri(config.source_bucket, key)
                    for key in selected_source_keys
                ],
                "copied_destinations": copied_destinations,
            },
            ensure_ascii=True,
        )
    )
    return 0


def main() -> None:
    """Run the fake daily batch seeding entrypoint."""

    configure_batch_logging()
    args = parse_args()
    raise SystemExit(asyncio.run(process_seed_batch(args)))


if __name__ == "__main__":
    main()
