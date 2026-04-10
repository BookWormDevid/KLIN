import io
import uuid
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from app.config import app_settings
from app.infrastructure.services.callback_sender import KlinCallbackSender
from app.infrastructure.services.s3_storage import S3ObjectStorage
from app.models import KlinModel, ProcessingState


class DummyAsyncContextManager:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeResponseContext:
    def __init__(self, status: int, body: str = "ok") -> None:
        self.status = status
        self._body = body

    async def __aenter__(self) -> "FakeResponseContext":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return self._body


class FakeSession:
    def __init__(self, responses: list[FakeResponseContext], seen_calls: list[dict]):
        self._responses = responses
        self._seen_calls = seen_calls

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url, *, data, headers, timeout):
        self._seen_calls.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self._responses.pop(0)


@pytest.fixture(name="klin_model")
def fixture_klin_model() -> KlinModel:
    return KlinModel(
        id=uuid.uuid4(),
        response_url="https://callback.example/result",
        video_path="s3://klin-videos/uploads/clip.mp4",
        state=ProcessingState.FINISHED,
        x3d='{"violence": 0.1}',
        mae='[{"label": "fight"}]',
        yolo='{"0": []}',
        objects=["person"],
        all_classes=["person", "fight"],
    )


def make_s3_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[S3ObjectStorage, MagicMock]:
    client = MagicMock()

    async def immediate_run_sync(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "app.infrastructure.services.s3_storage.boto3.client",
        MagicMock(return_value=client),
    )
    monkeypatch.setattr(S3ObjectStorage, "_run_sync", immediate_run_sync)

    monkeypatch.setenv("S3_ENDPOINT_URL", "http://s3.local")
    monkeypatch.setenv("S3_BUCKET_NAME", "klin-videos")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("S3_REGION", "us-east-1")
    monkeypatch.setenv("S3_ADDRESSING_STYLE", "path")

    for key in (
        "S3_ENDPOINT_URL",
        "S3_BUCKET_NAME",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_REGION",
        "S3_ADDRESSING_STYLE",
    ):
        app_settings.env_properties.pop(key, None)

    return S3ObjectStorage(), client


def make_client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code}}, "HeadBucket")


@pytest.mark.anyio
async def test_post_consumer_skips_models_without_callback_url(
    monkeypatch: pytest.MonkeyPatch,
    klin_model: KlinModel,
) -> None:
    sender = KlinCallbackSender()
    klin_model.response_url = None
    session_factory = MagicMock()
    monkeypatch.setattr(
        "app.infrastructure.services.callback_sender.aiohttp.ClientSession",
        session_factory,
    )

    await sender.post_consumer(klin_model)

    session_factory.assert_not_called()


@pytest.mark.anyio
async def test_post_consumer_retries_failed_callbacks_until_success(
    monkeypatch: pytest.MonkeyPatch,
    klin_model: KlinModel,
) -> None:
    sender = KlinCallbackSender()
    seen_calls: list[dict] = []
    responses = [FakeResponseContext(500, "boom"), FakeResponseContext(201, "ok")]
    monkeypatch.setattr(
        "app.infrastructure.services.callback_sender.async_timeout.timeout",
        lambda _seconds: DummyAsyncContextManager(),
    )
    monkeypatch.setattr(
        "app.infrastructure.services.callback_sender.aiohttp.ClientSession",
        lambda: FakeSession(responses, seen_calls),
    )

    await sender.post_consumer(klin_model)

    assert len(seen_calls) == 2
    assert seen_calls[0]["url"] == klin_model.response_url
    assert seen_calls[0]["headers"]["Content-Type"] == "application/json"


@pytest.mark.anyio
async def test_post_consumer_logs_after_final_failure(
    monkeypatch: pytest.MonkeyPatch,
    klin_model: KlinModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sender = KlinCallbackSender()
    seen_calls: list[dict] = []
    responses = [
        FakeResponseContext(500, "boom-1"),
        FakeResponseContext(500, "boom-2"),
        FakeResponseContext(500, "boom-3"),
    ]
    monkeypatch.setattr(
        "app.infrastructure.services.callback_sender.async_timeout.timeout",
        lambda _seconds: DummyAsyncContextManager(),
    )
    monkeypatch.setattr(
        "app.infrastructure.services.callback_sender.aiohttp.ClientSession",
        lambda: FakeSession(responses, seen_calls),
    )

    await sender.post_consumer(klin_model)

    assert len(seen_calls) == 3
    assert "Callback failed after 3 attempts" in caplog.text


@pytest.mark.anyio
async def test_upload_fileobj_uploads_content_and_returns_s3_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client = make_s3_storage(monkeypatch)
    fileobj = io.BytesIO(b"video-bytes")

    uri = await storage.upload_fileobj(
        fileobj=fileobj,
        object_key="uploads/clip.mp4",
        content_type="video/mp4",
        max_size_bytes=100,
    )

    assert uri.endswith("/uploads/clip.mp4")
    client.head_bucket.assert_called_once()
    client.upload_fileobj.assert_called_once()
    upload_kwargs = client.upload_fileobj.call_args.kwargs
    assert upload_kwargs["Bucket"] == storage._bucket_name
    assert upload_kwargs["Key"] == "uploads/clip.mp4"
    assert upload_kwargs["ExtraArgs"] == {"ContentType": "video/mp4"}


@pytest.mark.anyio
async def test_upload_fileobj_rejects_empty_and_large_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client = make_s3_storage(monkeypatch)

    with pytest.raises(ValueError, match="Uploaded file is empty"):
        await storage.upload_fileobj(
            fileobj=io.BytesIO(b""),
            object_key="uploads/empty.mp4",
        )

    with pytest.raises(ValueError, match="File too large"):
        await storage.upload_fileobj(
            fileobj=io.BytesIO(b"12345"),
            object_key="uploads/large.mp4",
            max_size_bytes=4,
        )

    client.upload_fileobj.assert_not_called()


@pytest.mark.anyio
async def test_download_and_delete_delegate_to_boto_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client = make_s3_storage(monkeypatch)
    source_uri = "s3://klin-videos/uploads/clip.mp4"

    await storage.download_to_path(
        source_uri=source_uri,
        destination_path="C:/tmp/clip.mp4",
    )
    await storage.delete(source_uri)

    client.download_file.assert_called_once_with(
        "klin-videos",
        "uploads/clip.mp4",
        "C:/tmp/clip.mp4",
    )
    client.delete_object.assert_called_once_with(
        Bucket="klin-videos",
        Key="uploads/clip.mp4",
    )


@pytest.mark.anyio
async def test_list_objects_returns_s3_uris_for_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client = make_s3_storage(monkeypatch)
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "klin/batch/2026-04-10/a.mp4"}]},
        {
            "Contents": [
                {"Key": "klin/batch/2026-04-10/subdir/"},
                {"Key": "klin/batch/2026-04-10/b.avi"},
            ]
        },
    ]
    client.get_paginator.return_value = paginator

    uris = await storage.list_objects("klin/batch/2026-04-10/")

    assert uris == [
        "s3://klin-videos/klin/batch/2026-04-10/a.mp4",
        "s3://klin-videos/klin/batch/2026-04-10/b.avi",
    ]
    client.get_paginator.assert_called_once_with("list_objects_v2")


@pytest.mark.anyio
async def test_ensure_bucket_exists_creates_missing_bucket_with_region_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client = make_s3_storage(monkeypatch)
    storage._region_name = "eu-central-1"
    client.head_bucket.side_effect = make_client_error("404")

    await storage._ensure_bucket_exists()

    client.create_bucket.assert_called_once_with(
        Bucket=storage._bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
    )
    assert storage._bucket_ready is True


@pytest.mark.anyio
async def test_ensure_bucket_exists_tolerates_already_owned_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client = make_s3_storage(monkeypatch)
    storage._region_name = "us-east-1"
    client.head_bucket.side_effect = make_client_error("NoSuchBucket")
    client.create_bucket.side_effect = make_client_error("BucketAlreadyOwnedByYou")

    await storage._ensure_bucket_exists()

    client.create_bucket.assert_called_once_with(Bucket=storage._bucket_name)
    assert storage._bucket_ready is True


@pytest.mark.anyio
async def test_ensure_bucket_exists_reraises_unexpected_head_bucket_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client = make_s3_storage(monkeypatch)
    client.head_bucket.side_effect = make_client_error("403")

    with pytest.raises(ClientError):
        await storage._ensure_bucket_exists()

    client.create_bucket.assert_not_called()


def test_s3_uri_helpers_cover_valid_and_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = make_s3_storage(monkeypatch)
    fileobj = io.BytesIO(b"abcdef")
    fileobj.seek(2)

    assert storage.build_uri(bucket_name="klin-videos", object_key="uploads/a.mp4") == (
        "s3://klin-videos/uploads/a.mp4"
    )
    assert storage.parse_uri("s3://klin-videos/uploads/a.mp4") == (
        "klin-videos",
        "uploads/a.mp4",
    )
    assert storage._get_file_size(fileobj) == 6
    assert fileobj.tell() == 2
    assert storage._extract_error_code(make_client_error("404")) == "404"

    with pytest.raises(ValueError, match="Unsupported S3 URI"):
        storage.parse_uri("https://example.com/video.mp4")
