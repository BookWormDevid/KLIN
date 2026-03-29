import asyncio
import uuid
from unittest.mock import AsyncMock

import msgspec

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
        "queue": app_settings.Klin_process_queue,
    }


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
