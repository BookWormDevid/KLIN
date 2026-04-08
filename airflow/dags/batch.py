# dags/batch_video_inference_dag.py

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import boto3
from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, task
from botocore.config import Config

from app.application.interfaces import IKlinTaskRepository
from app.application.services import KlinService
from app.bootstrap import create_worker_container
from app.config import app_settings
from app.models import KlinModel, ProcessingState


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run_async(coro):
    return asyncio.run(coro)


async def _resolve_from_container(type_: Any):
    """
    This code assumes standard Dishka async-container usage.
    I cannot fully confirm exact runtime API without the actual Dishka version,
    but provider wiring itself is confirmed by your IOC module.
    """
    container = create_worker_container()

    async with container as c:
        # most likely API
        if hasattr(c, "get"):
            value = c.get(type_)
            if asyncio.iscoroutine(value):
                return await value
            return value

        raise RuntimeError("Dishka container API mismatch: expected async get(...)")


def _build_s3_client():
    s3_config = cast(
        Any,
        {"addressing_style": app_settings.s3_addressing_style},
    )
    return boto3.client(
        "s3",
        endpoint_url=app_settings.s3_endpoint_url,
        aws_access_key_id=app_settings.s3_access_key_id,
        aws_secret_access_key=app_settings.s3_secret_access_key,
        region_name=app_settings.s3_region,
        config=Config(signature_version="s3v4", s3=s3_config),
    )


def _list_local_videos(input_dir: str) -> list[str]:
    root = Path(input_dir)
    if not root.exists():
        raise AirflowFailException(f"Directory does not exist: {input_dir}")
    if not root.is_dir():
        raise AirflowFailException(f"Not a directory: {input_dir}")

    return [
        str(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]


def _list_s3_videos(prefix: str) -> list[str]:
    client = _build_s3_client()
    bucket = app_settings.s3_bucket_name

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    videos: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if Path(key).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(f"s3://{bucket}/{key}")

    return sorted(videos)


def _resolve_videos(
    *,
    video_paths: list[str] | None,
    input_dir: str | None,
    s3_prefix: str | None,
    run_date: str | None,
    s3_date_template: str,
) -> list[str]:
    if video_paths:
        return video_paths

    if input_dir:
        return _list_local_videos(input_dir)

    if s3_prefix:
        return _list_s3_videos(s3_prefix)

    if run_date:
        dt = datetime.strptime(run_date, "%Y-%m-%d")
        prefix = dt.strftime(s3_date_template)
        return _list_s3_videos(prefix)

    raise AirflowFailException(
        "Provide one of: video_paths, input_dir, s3_prefix, run_date"
    )


async def _create_klin_task(video_path: str) -> str:
    repo: IKlinTaskRepository = await _resolve_from_container(IKlinTaskRepository)

    model = KlinModel(
        response_url=None,
        video_path=video_path,
        state=ProcessingState.PENDING,
    )
    created = await repo.create(model)
    return str(created.id)


async def _run_klin(klin_id: str) -> dict[str, Any]:
    service: KlinService = await _resolve_from_container(KlinService)
    repo: IKlinTaskRepository = await _resolve_from_container(IKlinTaskRepository)

    klin_uuid = uuid.UUID(klin_id)
    await service.perform_klin(klin_uuid)
    model = await repo.get_by_id(klin_uuid)

    return {
        "klin_id": str(model.id),
        "video_path": model.video_path,
        "state": str(model.state),
        "objects_count": len(model.objects or []),
        "classes_count": len(model.all_classes or []),
    }


@dag(
    dag_id="batch_video_inference_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "video_paths": Param(default=None, type=["null", "array"]),
        "input_dir": Param(default=None, type=["null", "string"]),
        "s3_prefix": Param(default=None, type=["null", "string"]),
        "run_date": Param(default=None, type=["null", "string"]),
        "s3_date_template": Param(default="incoming/videos/%Y/%m/%d/", type="string"),
    },
    tags=["klin", "batch", "video", "inference"],
)
def batch_video_inference_dag():
    @task
    def discover(**context) -> list[str]:
        params = context["params"]
        videos = _resolve_videos(
            video_paths=params.get("video_paths"),
            input_dir=params.get("input_dir"),
            s3_prefix=params.get("s3_prefix"),
            run_date=params.get("run_date"),
            s3_date_template=params.get("s3_date_template"),
        )
        if not videos:
            raise AirflowFailException("No videos found")
        return videos

    @task
    def create_task_record(video_path: str) -> str:
        return run_async(_create_klin_task(video_path))

    @task
    def process_video(klin_id: str) -> dict[str, Any]:
        return run_async(_run_klin(klin_id))

    @task
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        finished = sum(1 for x in items if "FINISHED" in x["state"])
        failed = sum(1 for x in items if "ERROR" in x["state"])
        return {
            "total": total,
            "finished": finished,
            "failed": failed,
            "items": items,
        }

    videos = discover()
    klin_ids = create_task_record.expand(video_path=videos)
    results = process_video.expand(klin_id=klin_ids)
    summarize(results)


dag = batch_video_inference_dag()
