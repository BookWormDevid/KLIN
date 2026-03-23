import uuid
from unittest.mock import AsyncMock

import pytest

from app.application.dto import StreamEventDto, StreamProcessDto
from app.infrastructure.producers import KlinEventProducer, KlinProcessProducer


@pytest.mark.asyncio
async def test_klin_process_producer_supports_send_stream() -> None:
    broker = AsyncMock()
    producer = KlinProcessProducer(_rabbit_broker=broker)

    payload = StreamProcessDto(stream_id=uuid.UUID(int=0))

    await producer.send_stream(payload)

    broker.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_klin_event_producer_dispatches_yolo_event() -> None:
    repository = AsyncMock()
    producer = KlinEventProducer(_repository=repository)

    event = StreamEventDto(
        id="1",
        stream_id=uuid.UUID(int=0),
        camera_id="cam-1",
        type="YOLO",
        payload={"detections": []},
    )

    await producer.send_event(event)

    repository.save_yolo.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_klin_event_producer_rejects_unknown_event() -> None:
    repository = AsyncMock()
    producer = KlinEventProducer(_repository=repository)

    event = StreamEventDto(
        id="1",
        stream_id=uuid.UUID(int=0),
        camera_id="cam-1",
        type="UNKNOWN",
        payload={},
    )

    with pytest.raises(ValueError, match="Unsupported stream event type"):
        await producer.send_event(event)
