import argparse
import json
from unittest.mock import AsyncMock

import pytest

from app.batch.seed_fake_batch import (
    build_destination_key,
    compute_seed_value,
    process_seed_batch,
    select_source_keys,
)


def test_select_source_keys_is_deterministic() -> None:
    source_keys = [
        "dataset/c.mp4",
        "dataset/a.mp4",
        "dataset/b.mp4",
    ]
    seed_value = compute_seed_value(
        batch_date="2026-04-14",
        source_bucket="ufc-crime-klin-dataset",
        source_prefix="",
        sample_size=2,
    )

    first = select_source_keys(source_keys, sample_size=2, seed_value=seed_value)
    second = select_source_keys(
        list(reversed(source_keys)),
        sample_size=2,
        seed_value=seed_value,
    )

    assert first == second


def test_build_destination_key_keeps_partition_and_uniqueness() -> None:
    destination_key = build_destination_key(
        batch_date="2026-04-14",
        base_prefix="klin/batch",
        source_key="dataset/fights/clip.mp4",
        ordinal=3,
    )

    assert destination_key.startswith("klin/batch/2026-04-14/seed-003-")
    assert destination_key.endswith("-clip.mp4")


@pytest.mark.anyio
async def test_process_seed_batch_skips_when_destination_has_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        date="2026-04-14",
        count=0,
        source_bucket="",
        source_prefix="",
    )
    list_objects_mock = AsyncMock(return_value=["klin/batch/2026-04-14/existing.mp4"])
    copy_object_mock = AsyncMock()

    monkeypatch.setenv("S3_BUCKET_NAME", "klin-videos")
    monkeypatch.setattr("app.batch.seed_fake_batch.build_s3_client", lambda: object())
    monkeypatch.setattr(
        "app.batch.seed_fake_batch.ensure_bucket_exists",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "app.batch.seed_fake_batch.list_bucket_object_keys",
        list_objects_mock,
    )
    monkeypatch.setattr(
        "app.batch.seed_fake_batch.copy_object",
        copy_object_mock,
    )

    exit_code = await process_seed_batch(args)

    assert exit_code == 0
    copy_object_mock.assert_not_awaited()
    output = json.loads(capsys.readouterr().out.strip())
    assert output["seeded"] is False
    assert output["reason"] == "existing_destination_data"


@pytest.mark.anyio
async def test_process_seed_batch_copies_random_source_objects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        date="2026-04-14",
        count=2,
        source_bucket="",
        source_prefix="",
    )
    list_objects_mock = AsyncMock(
        side_effect=[
            [],
            [
                "dataset/a.mp4",
                "dataset/b.avi",
                "dataset/ignore.txt",
            ],
        ]
    )
    copy_object_mock = AsyncMock()

    monkeypatch.setenv("S3_BUCKET_NAME", "klin-videos")
    monkeypatch.setenv("KLIN_BATCH_SEED_SOURCE_BUCKET", "ufc-crime-klin-dataset")
    monkeypatch.setattr("app.batch.seed_fake_batch.build_s3_client", lambda: object())
    monkeypatch.setattr(
        "app.batch.seed_fake_batch.ensure_bucket_exists",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "app.batch.seed_fake_batch.list_bucket_object_keys",
        list_objects_mock,
    )
    monkeypatch.setattr(
        "app.batch.seed_fake_batch.copy_object",
        copy_object_mock,
    )

    exit_code = await process_seed_batch(args)

    assert exit_code == 0
    assert copy_object_mock.await_count == 2
    output = json.loads(capsys.readouterr().out.strip())
    assert output["seeded"] is True
    assert output["source_bucket"] == "ufc-crime-klin-dataset"
    assert len(output["selected_sources"]) == 2
    assert len(output["copied_destinations"]) == 2
