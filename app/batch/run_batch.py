"""Date-partitioned offline batch runner for KLIN S3 inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import PurePosixPath
from urllib.parse import urlparse
from uuid import UUID

from dishka import make_container

from app.application.interfaces import IKlinTaskRepository, IKlinVideoStorage
from app.application.services import KlinService
from app.config import app_settings
from app.ioc import get_worker_providers
from app.models import KlinModel, ProcessingState


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one batch run."""

    parser = argparse.ArgumentParser(
        description="Run date-partitioned KLIN batch processing from S3."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date partition in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional override for the base S3 prefix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of files to process. 0 means no limit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing the rest of the batch after a failure.",
    )
    return parser.parse_args()


def build_partition_prefix(batch_date: str, base_prefix: str) -> str:
    normalized_base = base_prefix.strip().strip("/")
    return f"{normalized_base}/{batch_date}/" if normalized_base else f"{batch_date}/"


def has_allowed_extension(source_uri: str) -> bool:
    suffix = PurePosixPath(urlparse(source_uri).path).suffix.lower()
    return suffix in app_settings.batch_file_extensions


def build_result_row(
    *,
    klin_id: UUID,
    source_uri: str,
    state: ProcessingState,
    action: str,
) -> dict[str, str]:
    """Build one JSON-serializable result row for the batch summary."""

    return {
        "klin_id": str(klin_id),
        "source_uri": source_uri,
        "state": state.value,
        "action": action,
    }


async def prepare_task_for_source_uri(
    repository: IKlinTaskRepository,
    source_uri: str,
) -> tuple[KlinModel, str]:
    """Prepare an idempotent offline task for one S3 source object."""

    existing = await repository.get_latest_by_video_path(source_uri)
    if existing is None:
        klin = KlinModel(
            response_url=None,
            video_path=source_uri,
            state=ProcessingState.PENDING,
        )
        created = await repository.create(klin)
        return created, "created"

    if existing.state == ProcessingState.ERROR:
        existing.state = ProcessingState.PENDING
        existing.x3d = None
        existing.mae = None
        existing.yolo = None
        existing.objects = []
        existing.all_classes = []
        await repository.update(existing)
        return existing, "retried"

    if existing.state == ProcessingState.FINISHED:
        return existing, "skipped_finished"

    return existing, "skipped_inflight"


async def discover_source_uris(
    args: argparse.Namespace,
    storage: IKlinVideoStorage,
) -> list[str]:
    """Discover eligible S3 objects for the requested batch date."""

    base_prefix = args.prefix.strip() or app_settings.batch_s3_prefix
    target_prefix = build_partition_prefix(args.date, base_prefix)
    source_uris = await storage.list_objects(target_prefix)
    source_uris = [uri for uri in source_uris if has_allowed_extension(uri)]

    if args.limit > 0:
        source_uris = source_uris[: args.limit]

    logger.info(
        "Discovered %d batch files under prefix=%s",
        len(source_uris),
        target_prefix,
    )
    return source_uris


async def process_source_uri(
    *,
    repository: IKlinTaskRepository,
    klin_service: KlinService,
    source_uri: str,
    continue_on_error: bool,
) -> tuple[dict[str, str], bool, bool]:
    """Process one discovered source URI and return summary flags."""

    klin, action = await prepare_task_for_source_uri(repository, source_uri)

    if action.startswith("skipped"):
        return (
            build_result_row(
                klin_id=klin.id,
                source_uri=source_uri,
                state=klin.state,
                action=action,
            ),
            False,
            False,
        )

    try:
        await klin_service.perform_klin(klin.id)
        refreshed = await repository.get_by_id(klin.id)
        failed = refreshed.state == ProcessingState.ERROR
        return (
            build_result_row(
                klin_id=refreshed.id,
                source_uri=source_uri,
                state=refreshed.state,
                action=action,
            ),
            failed,
            failed and not continue_on_error,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(
            "Batch object failed. klin_id=%s source_uri=%s error=%s",
            klin.id,
            source_uri,
            exc,
        )
        return (
            build_result_row(
                klin_id=klin.id,
                source_uri=source_uri,
                state=ProcessingState.ERROR,
                action=action,
            ),
            True,
            not continue_on_error,
        )


async def process_batch(args: argparse.Namespace) -> int:
    """Process all discovered S3 objects for the requested date partition."""

    container = make_container(*get_worker_providers())
    storage = container.get(IKlinVideoStorage)
    repository = container.get(IKlinTaskRepository)
    klin_service = container.get(KlinService)
    source_uris = await discover_source_uris(args, storage)
    failures = 0
    results: list[dict[str, str]] = []

    for source_uri in source_uris:
        result_row, failed, should_stop = await process_source_uri(
            repository=repository,
            klin_service=klin_service,
            source_uri=source_uri,
            continue_on_error=args.continue_on_error,
        )
        results.append(result_row)
        failures += int(failed)
        if should_stop:
            break

    print(json.dumps({"date": args.date, "results": results}, ensure_ascii=True))
    return 1 if failures else 0


def main() -> None:
    """Run the batch processor from the command line."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    raise SystemExit(asyncio.run(process_batch(args)))


if __name__ == "__main__":
    main()
