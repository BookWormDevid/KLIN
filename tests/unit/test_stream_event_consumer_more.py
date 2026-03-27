import uuid
from unittest.mock import AsyncMock

import pytest

from app.application.consumers.stream_event_consumer import StreamEventConsumer
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


@pytest.fixture(name="stream_event_consumer")
def fixture_stream_event_consumer(
    stream_event_repository: AsyncMock,
) -> StreamEventConsumer:
    return StreamEventConsumer(repository=stream_event_repository)


@pytest.fixture(name="stream_state")
def fixture_stream_state() -> KlinStreamState:
    return KlinStreamState(
        id=uuid.uuid4(),
        camera_id="cam-1",
        camera_url="rtsp://example/cam-1",
        state=ProcessingState.PROCESSING,
    )


@pytest.mark.anyio
async def test_handle_mae_event_updates_last_prediction(
    stream_event_consumer: StreamEventConsumer,
    stream_event_repository: AsyncMock,
    stream_state: KlinStreamState,
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
    stream_event_repository.get_by_id_stream.return_value = stream_state

    await stream_event_consumer.handle(event)

    stream_event_repository.save_mae.assert_awaited_once()
    saved_event = stream_event_repository.save_mae.await_args.args[0]
    assert saved_event.label == "fight"
    assert saved_event.confidence == 0.94
    assert saved_event.start_ts == 1.0
    assert stream_state.last_mae_label == "fight"
    assert stream_state.last_mae_confidence == 0.94


@pytest.mark.anyio
async def test_handle_x3d_event_updates_last_prediction(
    stream_event_consumer: StreamEventConsumer,
    stream_event_repository: AsyncMock,
    stream_state: KlinStreamState,
) -> None:
    event = make_stream_event(
        "X3D_VIOLENCE",
        {"timestamp": 12.0, "label": "violence", "prob": 0.77},
    )
    stream_event_repository.get_by_id_stream.return_value = stream_state

    await stream_event_consumer.handle(event)

    stream_event_repository.save_x3d.assert_awaited_once()
    saved_event = stream_event_repository.save_x3d.await_args.args[0]
    assert saved_event.ts == 12.0
    assert saved_event.label == "violence"
    assert stream_state.last_x3d_label == "violence"
    assert stream_state.last_x3d_confidence == 0.77


@pytest.mark.anyio
async def test_handle_x3d_event_skips_missing_timestamps(
    stream_event_consumer: StreamEventConsumer,
    stream_event_repository: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = make_stream_event("X3D_VIOLENCE", {"label": "violence"})

    await stream_event_consumer._handle_x3d(event)

    assert "missing timestamp payload" in caplog.text
    stream_event_repository.save_x3d.assert_not_awaited()


@pytest.mark.anyio
async def test_handle_retries_once_then_succeeds(
    stream_event_consumer: StreamEventConsumer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_mock = AsyncMock()
    event = make_stream_event("YOLO", {"detections": [], "timestamp": 1.0})
    retrying_handler = AsyncMock(side_effect=[RuntimeError("boom"), None])
    monkeypatch.setattr(
        "app.application.consumers.stream_event_consumer.asyncio.sleep",
        sleep_mock,
    )
    monkeypatch.setattr(stream_event_consumer, "_handle_yolo", retrying_handler)

    await stream_event_consumer.handle(event)

    assert retrying_handler.await_count == 2
    sleep_mock.assert_awaited_once_with(0.5)


@pytest.mark.anyio
async def test_handle_raises_after_all_retries(
    stream_event_consumer: StreamEventConsumer,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sleep_mock = AsyncMock()
    event = make_stream_event("YOLO", {"detections": [], "timestamp": 1.0})
    failing_handler = AsyncMock(side_effect=RuntimeError("still broken"))
    monkeypatch.setattr(
        "app.application.consumers.stream_event_consumer.asyncio.sleep",
        sleep_mock,
    )
    monkeypatch.setattr(stream_event_consumer, "_handle_yolo", failing_handler)

    with pytest.raises(RuntimeError, match="still broken"):
        await stream_event_consumer.handle(event)

    assert failing_handler.await_count == 3
    assert sleep_mock.await_count == 2
    assert "Failed to save YOLO event" in caplog.text
