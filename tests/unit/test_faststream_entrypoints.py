import importlib
import importlib.metadata as importlib_metadata
import sys
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock

import msgspec
import pytest

from app.application.dto import KlinProcessDto, StreamEventDto, StreamProcessDto


class FakeBroker:
    def subscriber(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


class FakeCounter:
    def __init__(self, *args, **kwargs) -> None:
        self.count = 0
        self.labels_calls: list[dict[str, str]] = []

    def inc(self) -> None:
        self.count += 1

    def labels(self, **kwargs):
        self.labels_calls.append(kwargs)
        return self


class FakeTimer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        return False


class FakeHistogram:
    def __init__(self, *args, **kwargs) -> None:
        self.calls = 0

    def time(self) -> FakeTimer:
        self.calls += 1
        return FakeTimer()


class FakeContainer:
    def __init__(self, mapping: dict[str, object]) -> None:
        self.mapping = mapping

    def get(self, key: type[Any]) -> object:
        return self.mapping[key.__name__]


def load_faststream_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    mapping: dict[str, object],
):
    real_distribution: Callable[[str], Any] = importlib_metadata.distribution

    def safe_distribution(name: str) -> Any:
        if name == "attrs":
            return SimpleNamespace(version="21.3.0")
        distribution = real_distribution(name)
        version = getattr(distribution, "version", None)
        if not isinstance(version, str) or not version:
            return SimpleNamespace(version="0")
        return distribution

    monkeypatch.setattr(importlib_metadata, "distribution", safe_distribution)
    dishka = importlib.import_module("dishka")
    faststream_module = importlib.import_module("faststream")
    prometheus_client = importlib.import_module("prometheus_client")
    ioc_module = importlib.import_module("app.ioc")
    health_module = importlib.import_module("app.infrastructure.database.health")

    ping_mock = AsyncMock()
    monkeypatch.setattr(dishka, "make_container", lambda *args: FakeContainer(mapping))
    monkeypatch.setattr(ioc_module, "get_worker_providers", lambda: ("worker",))
    monkeypatch.setattr(
        faststream_module,
        "FastStream",
        lambda broker, **kwargs: SimpleNamespace(broker=broker, kwargs=kwargs),
    )
    monkeypatch.setattr(prometheus_client, "Counter", FakeCounter)
    monkeypatch.setattr(prometheus_client, "Histogram", FakeHistogram)
    monkeypatch.setattr(prometheus_client, "start_http_server", lambda port: None)
    monkeypatch.setattr(health_module, "ping_database", ping_mock)
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return module, ping_mock


@pytest.mark.anyio
async def test_faststream_worker_bootstrap_and_base_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker()
    klin_service = AsyncMock()
    module, ping_mock = load_faststream_module(
        monkeypatch,
        "app.presentation.faststream.app",
        {
            "RabbitBroker": broker,
            "AsyncEngine": object(),
            "KlinService": klin_service,
        },
    )
    message = SimpleNamespace(
        body=msgspec.json.encode(KlinProcessDto(klin_id=uuid.uuid4()))
    )

    await module.verify_worker_database()
    await module.base_handler(message)

    ping_mock.assert_awaited_once_with(module.db_engine)
    klin_service.perform_klin.assert_awaited_once()
    assert module.KLIN_PROCESSED.count == 1
    assert module.KLIN_PROCESSING_TIME.calls == 1


@pytest.mark.anyio
async def test_faststream_stream_worker_handles_events_and_starts_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker()
    stream_service = AsyncMock()
    stream_event_consumer = AsyncMock()
    module, ping_mock = load_faststream_module(
        monkeypatch,
        "app.presentation.faststream_stream.app",
        {
            "RabbitBroker": broker,
            "AsyncEngine": object(),
            "StreamService": stream_service,
            "IKlinStreamEventConsumer": stream_event_consumer,
        },
    )
    event = StreamEventDto(
        id="event-1",
        stream_id=uuid.uuid4(),
        camera_id="cam-1",
        type="YOLO",
        payload={"timestamp": 1.0, "detections": []},
    )
    event_message = SimpleNamespace(body=msgspec.json.encode(event))
    dto = StreamProcessDto(stream_id=uuid.uuid4())
    start_message = SimpleNamespace(body=msgspec.json.encode(dto))

    await module.verify_worker_database()
    await module.event_handler(event_message)
    await module.stream_start_handler(start_message)

    ping_mock.assert_awaited_once_with(module.db_engine)
    stream_event_consumer.handle.assert_awaited_once()
    stream_service.perform_stream.assert_awaited_once_with(stream_id=dto.stream_id)
    assert {"status": "success"} in module.STREAM_PROCESSED.labels_calls
    assert module.STREAM_PROCESSING_TIME.calls == 1
