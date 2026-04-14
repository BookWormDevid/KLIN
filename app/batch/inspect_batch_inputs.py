"""Inspect one daily batch partition in S3 and log eligible objects."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from app.batch.s3_batch_support import (
    DEFAULT_BATCH_FILE_EXTENSIONS,
    DEFAULT_BATCH_S3_PREFIX,
    add_date_argument,
    build_partition_prefix,
    build_s3_client,
    build_s3_uri,
    configure_batch_logging,
    get_optional_env,
    get_required_env,
    has_allowed_extension,
    list_bucket_object_keys,
    parse_allowed_extensions,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse one inspection run from the command line."""

    parser = argparse.ArgumentParser(
        description="Inspect daily KLIN batch inputs in S3."
    )
    add_date_argument(parser)
    parser.add_argument(
        "--bucket",
        default="",
        help="Optional override for the inspected bucket.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional override for the base prefix.",
    )
    return parser.parse_args()


async def inspect_batch_inputs(args: argparse.Namespace) -> int:
    """Inspect and log all eligible objects for one date partition."""

    client = build_s3_client()
    target_bucket = args.bucket.strip() or get_required_env("S3_BUCKET_NAME")
    batch_prefix = args.prefix.strip() or get_optional_env(
        "KLIN_BATCH_S3_PREFIX",
        DEFAULT_BATCH_S3_PREFIX,
    )
    allowed_extensions = parse_allowed_extensions(
        get_optional_env(
            "KLIN_BATCH_FILE_EXTENSIONS",
            DEFAULT_BATCH_FILE_EXTENSIONS,
        )
    )
    target_prefix = build_partition_prefix(args.date, batch_prefix)

    object_keys = await list_bucket_object_keys(
        client,
        bucket_name=target_bucket,
        prefix=target_prefix,
    )
    eligible_uris = sorted(
        build_s3_uri(target_bucket, key)
        for key in object_keys
        if has_allowed_extension(key, allowed_extensions)
    )

    logger.info(
        "Inspected bucket=%s prefix=%s and found %d eligible object(s).",
        target_bucket,
        target_prefix,
        len(eligible_uris),
    )
    for uri in eligible_uris:
        logger.info("Eligible batch object: %s", uri)

    print(
        json.dumps(
            {
                "date": args.date,
                "bucket": target_bucket,
                "prefix": target_prefix,
                "count": len(eligible_uris),
                "objects": eligible_uris,
            },
            ensure_ascii=True,
        )
    )
    return 0


def main() -> None:
    """Run the batch partition inspection entrypoint."""

    configure_batch_logging()
    args = parse_args()
    raise SystemExit(asyncio.run(inspect_batch_inputs(args)))


if __name__ == "__main__":
    main()
