import uuid
from typing import cast
from unittest.mock import AsyncMock

import pytest

from app.application.consumers.stream_event_consumer import StreamEventConsumer
from app.application.consumers.stream_event_service import StreamEventService
from app.application.dto import StreamEventDto
from app.models import KlinStreamState, ProcessingState


def make_stream_event(event_type: str, payload: dict) -> StreamEventDto:
    return StreamEventDto(
        id="event-1",
        stream_id=uuid.uuid4(),
        camera_id="cam-1",
        type=event_type,
        payload=payload,
    )


@pytest.fixture(name="stream_event_repository")
def fixture_stream_event_repository() -> AsyncMock:
    return AsyncMock()


@pytest.fixture(name="stream_state_repository")
def fixture_stream_state_repository() -> AsyncMock:
    repository = AsyncMock()
    repository.get_by_id_stream.return_value = None
    return repository


@pytest.fixture(name="stream_processor")
def fixture_stream_processor() -> AsyncMock:
    return AsyncMock()


@pytest.fixture(name="stream_event_service")
def fixture_stream_event_service(
    stream_event_repository: AsyncMock,
    stream_state_repository: AsyncMock,
    stream_processor: AsyncMock,
) -> StreamEventService:
    return StreamEventService(
        repository_event=stream_event_repository,
        repository_id=stream_state_repository,
        stream_processor=stream_processor,
    )


@pytest.fixture(name="sample_stream_state")
def fixture_sample_stream_state() -> KlinStreamState:
    return KlinStreamState(
        id=uuid.uuid4(),
        camera_id="cam-1",
        camera_url="rtsp://example/cam-1",
        state=ProcessingState.PROCESSING,
    )


@pytest.mark.anyio
async def test_handle_mae_event_builds_expected_result(
    stream_event_service: StreamEventService,
    stream_event_repository: AsyncMock,
) -> None:
    event = make_stream_event(
        "MAE",
        {
            "label": "fight",
            "confidence": 0.94,
            "probs": [{"class_name": "fight", "probability": 0.94}],
            "start_ts": 1.0,
            "end_ts": 2.5,
        },
    )

    await stream_event_service.process(event)

    stream_event_repository.save_mae.assert_awaited_once()
    saved_event = stream_event_repository.save_mae.await_args.args[0]
    assert saved_event.label == "fight"
    assert saved_event.confidence == 0.94
    assert saved_event.start_ts == 1.0


@pytest.mark.anyio
async def test_handle_x3d_event_builds_expected_result(
    stream_event_service: StreamEventService,
    stream_event_repository: AsyncMock,
) -> None:
    event = make_stream_event(
        "X3D_VIOLENCE",
        {"timestamp": 12.0, "label": "violence", "prob": 0.77},
    )

    await stream_event_service.process(event)

    stream_event_repository.save_x3d.assert_awaited_once()
    saved_event = stream_event_repository.save_x3d.await_args.args[0]
    assert saved_event.ts == 12.0
    assert saved_event.label == "violence"


@pytest.mark.anyio
async def test_consumer_delegates_to_service() -> None:
    service = AsyncMock()
    consumer = StreamEventConsumer(service=cast(StreamEventService, service))
    event = make_stream_event("YOLO", {"detections": [], "timestamp": 1.0})

    await consumer.handle(event)

    service.process.assert_awaited_once_with(event)


@pytest.mark.anyio
async def test_handle_stop_requests_processor_shutdown(
    stream_event_service: StreamEventService,
    stream_processor: AsyncMock,
) -> None:
    event = make_stream_event("STOP_STREAM", {})
    stream_processor.wait_stopped.return_value = True

    await stream_event_service.process(event)

    stream_processor.stop.assert_awaited_once_with(event.camera_id)
    stream_processor.wait_stopped.assert_awaited_once_with(event.camera_id)


@pytest.mark.anyio
async def test_handle_stream_stopped_marks_stream_stopped(
    stream_event_service: StreamEventService,
    stream_state_repository: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    event = make_stream_event("STREAM_STOPPED", {"timestamp": "2026-03-29T00:00:00Z"})
    sample_stream_state.id = event.stream_id
    stream_state_repository.get_by_id_stream.return_value = sample_stream_state

    await stream_event_service.process(event)

    assert sample_stream_state.state == ProcessingState.STOPPED
    stream_state_repository.update.assert_awaited_once_with(sample_stream_state)


@pytest.mark.anyio
async def test_handle_raises_for_unsupported_event_type(
    stream_event_service: StreamEventService,
) -> None:
    event = make_stream_event("UNKNOWN", {})

    with pytest.raises(ValueError, match="Unsupported event type: UNKNOWN"):
        await stream_event_service.process(event)
