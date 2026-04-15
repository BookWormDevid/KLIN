import argparse
import json
import uuid
from unittest.mock import AsyncMock

import pytest

from app.batch.run_batch import prepare_task_for_source_uri, process_batch
from app.config import app_settings
from app.models import KlinModel, ProcessingState


class FakeContainer:
    def __init__(self, mapping: dict[object, object]) -> None:
        self._mapping = mapping

    def get(self, key: object) -> object:
        return self._mapping[key]


@pytest.mark.anyio
async def test_prepare_task_for_source_uri_creates_new_task() -> None:
    repository = AsyncMock()
    created = KlinModel(
        id=uuid.uuid4(),
        response_url=None,
        video_path="s3://klin-videos/klin/batch/2026-04-10/a.mp4",
        state=ProcessingState.PENDING,
    )
    repository.get_latest_by_video_path.return_value = None
    repository.create.return_value = created

    klin, action = await prepare_task_for_source_uri(repository, created.video_path)

    assert klin is created
    assert action == "created"
    repository.create.assert_awaited_once()
    repository.update.assert_not_awaited()


@pytest.mark.anyio
async def test_prepare_task_for_source_uri_retries_error_task() -> None:
    failed = KlinModel(
        id=uuid.uuid4(),
        response_url=None,
        video_path="s3://klin-videos/klin/batch/2026-04-10/b.mp4",
        state=ProcessingState.ERROR,
        x3d="old",
        mae="old",
        yolo="old",
        objects=["person"],
        all_classes=["fight"],
    )
    repository = AsyncMock()
    repository.get_latest_by_video_path.return_value = failed

    klin, action = await prepare_task_for_source_uri(repository, failed.video_path)

    assert klin is failed
    assert action == "retried"
    assert klin.state == ProcessingState.PENDING
    assert klin.x3d is None
    assert klin.mae is None
    assert klin.yolo is None
    assert klin.objects == []
    assert klin.all_classes == []
    repository.create.assert_not_awaited()
    repository.update.assert_awaited_once_with(failed)


@pytest.mark.anyio
async def test_process_batch_skips_finished_existing_task(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_uri = "s3://klin-videos/klin/batch/2026-04-10/c.mp4"
    finished = KlinModel(
        id=uuid.uuid4(),
        response_url=None,
        video_path=source_uri,
        state=ProcessingState.FINISHED,
    )

    repository = AsyncMock()
    repository.get_latest_by_video_path.return_value = finished
    storage = AsyncMock()
    storage.list_objects.return_value = [source_uri]
    klin_service = AsyncMock()

    from app.application.interfaces import IKlinTaskRepository, IKlinVideoStorage

    container = FakeContainer(
        {
            IKlinVideoStorage: storage,
            IKlinTaskRepository: repository,
        }
    )

    monkeypatch.setattr("app.batch.run_batch.make_container", lambda *args: container)
    monkeypatch.setattr("app.batch.run_batch.get_worker_providers", lambda: ("worker",))
    monkeypatch.setattr(
        "app.batch.run_batch._build_batch_klin_service",
        lambda _container: klin_service,
    )
    monkeypatch.setattr(
        "app.batch.run_batch._verify_database_connectivity",
        AsyncMock(),
    )

    monkeypatch.setenv("KLIN_BATCH_S3_PREFIX", "klin/batch")
    monkeypatch.setenv("KLIN_BATCH_FILE_EXTENSIONS", ".mp4")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:pass@postgresql:5432/klin",
    )
    for key in ("KLIN_BATCH_S3_PREFIX", "KLIN_BATCH_FILE_EXTENSIONS", "DATABASE_URL"):
        app_settings.env_properties.pop(key, None)

    args = argparse.Namespace(
        date="2026-04-10",
        prefix="",
        limit=0,
        continue_on_error=False,
    )

    exit_code = await process_batch(args)

    assert exit_code == 0
    klin_service.perform_klin.assert_not_awaited()

    output = json.loads(capsys.readouterr().out.strip())
    assert output["results"] == [
        {
            "klin_id": str(finished.id),
            "source_uri": source_uri,
            "state": ProcessingState.FINISHED.value,
            "action": "skipped_finished",
        }
    ]


@pytest.mark.anyio
async def test_process_batch_handles_string_state_from_repository(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_uri = "s3://klin-videos/klin/batch/2026-04-10/d.mp4"
    finished = KlinModel(
        id=uuid.uuid4(),
        response_url=None,
        video_path=source_uri,
        state=ProcessingState.FINISHED,
    )
    object.__setattr__(finished, "state", "FINISHED")

    repository = AsyncMock()
    repository.get_latest_by_video_path.return_value = finished
    storage = AsyncMock()
    storage.list_objects.return_value = [source_uri]
    klin_service = AsyncMock()

    from app.application.interfaces import IKlinTaskRepository, IKlinVideoStorage

    container = FakeContainer(
        {
            IKlinVideoStorage: storage,
            IKlinTaskRepository: repository,
        }
    )

    monkeypatch.setattr("app.batch.run_batch.make_container", lambda *args: container)
    monkeypatch.setattr("app.batch.run_batch.get_worker_providers", lambda: ("worker",))
    monkeypatch.setattr(
        "app.batch.run_batch._build_batch_klin_service",
        lambda _container: klin_service,
    )
    monkeypatch.setattr(
        "app.batch.run_batch._verify_database_connectivity",
        AsyncMock(),
    )

    monkeypatch.setenv("KLIN_BATCH_S3_PREFIX", "klin/batch")
    monkeypatch.setenv("KLIN_BATCH_FILE_EXTENSIONS", ".mp4")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:pass@postgresql:5432/klin",
    )
    for key in ("KLIN_BATCH_S3_PREFIX", "KLIN_BATCH_FILE_EXTENSIONS", "DATABASE_URL"):
        app_settings.env_properties.pop(key, None)

    args = argparse.Namespace(
        date="2026-04-10",
        prefix="",
        limit=0,
        continue_on_error=False,
    )

    exit_code = await process_batch(args)

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out.strip())
    assert output["results"] == [
        {
            "klin_id": str(finished.id),
            "source_uri": source_uri,
            "state": "FINISHED",
            "action": "skipped_finished",
        }
    ]
