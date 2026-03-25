import asyncio
import uuid
from unittest.mock import AsyncMock

import msgspec
import pytest

from app.application.consumers.stream_event_consumer import StreamEventConsumer
from app.application.dto import KlinProcessDto, StreamEventDto, StreamProcessDto
from app.config import app_settings
from app.infrastructure.producers import KlinEventProducer, KlinProcessProducer


def test_klin_process_producer_routes_offline_jobs_to_klin_queue() -> None:
    broker = AsyncMock()
    producer = KlinProcessProducer(_rabbit_broker=broker)

    payload = KlinProcessDto(klin_id=uuid.UUID(int=0))

    asyncio.run(producer.send(payload))

    broker.publish.assert_awaited_once()
    assert broker.publish.await_args.kwargs == {
        "queue": app_settings.Klin_queue,
    }


def test_klin_process_producer_routes_stream_jobs_to_stream_queue() -> None:
    broker = AsyncMock()
    producer = KlinProcessProducer(_rabbit_broker=broker)

    payload = StreamProcessDto(stream_id=uuid.UUID(int=0))

    asyncio.run(producer.send_stream(payload))

    broker.publish.assert_awaited_once()
    assert broker.publish.await_args.kwargs == {
        "queue": app_settings.Klin_stream_queue,
    }


def test_stream_commands_and_events_use_distinct_queues() -> None:
    assert app_settings.Klin_stream_queue != app_settings.Klin_stream_event_queue


def test_klin_event_producer_publishes_stream_event() -> None:
    broker = AsyncMock()
    producer = KlinEventProducer(_rabbit_broker=broker)

    event = StreamEventDto(
        id="1",
        stream_id=uuid.UUID(int=0),
        camera_id="cam-1",
        type="YOLO",
        payload={"detections": [], "timestamp": 123.0},
    )

    asyncio.run(producer.send_event(event))

    broker.publish.assert_awaited_once()
    encoded_event = broker.publish.await_args.args[0]

    assert msgspec.json.decode(encoded_event, type=StreamEventDto) == event
    assert broker.publish.await_args.kwargs == {
        "queue": app_settings.Klin_stream_event_queue,
    }


def test_stream_event_consumer_dispatches_yolo_event() -> None:
    repository = AsyncMock()
    repository.get_by_id_stream.return_value = None
    consumer = StreamEventConsumer(repository=repository)

    event = StreamEventDto(
        id="1",
        stream_id=uuid.UUID(int=0),
        camera_id="cam-1",
        type="YOLO",
        payload={"detections": [], "timestamp": 123.0},
    )

    asyncio.run(consumer.handle(event))

    repository.save_yolo.assert_awaited_once()
    saved_result = repository.save_yolo.await_args.args[0]

    assert saved_result.event_id == event.id
    assert saved_result.stream_id == event.stream_id
    assert saved_result.camera_id == event.camera_id
    assert saved_result.ts == event.payload["timestamp"]
    assert saved_result.detections == event.payload["detections"]


def test_stream_event_consumer_rejects_unknown_event() -> None:
    repository = AsyncMock()
    consumer = StreamEventConsumer(repository=repository)

    event = StreamEventDto(
        id="1",
        stream_id=uuid.UUID(int=0),
        camera_id="cam-1",
        type="UNKNOWN",
        payload={},
    )

    with pytest.raises(ValueError, match="Unsupported event type"):
        asyncio.run(consumer.handle(event))
