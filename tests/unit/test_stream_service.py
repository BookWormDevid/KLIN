import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from app.application.dto import StreamReadDto, StreamUploadDto
from app.application.services.stream import StreamService
from app.models import KlinStreamState, ProcessingState


@pytest.fixture(name="stream_repository")
def fixture_stream_repository() -> AsyncMock:
    repository = AsyncMock()
    repository.get_by_id_camera.return_value = None
    repository.get_by_id_stream.return_value = None
    repository.claim_for_processing_stream.return_value = None
    return repository


@pytest.fixture(name="stream_runner")
def fixture_stream_runner() -> AsyncMock:
    return AsyncMock()


@pytest.fixture(name="stream_process_producer")
def fixture_stream_process_producer() -> AsyncMock:
    return AsyncMock()


@pytest.fixture(name="stream_event_producer")
def fixture_stream_event_producer() -> AsyncMock:
    return AsyncMock()


@pytest.fixture(name="stream_service")
def fixture_stream_service(
    stream_repository: AsyncMock,
    stream_runner: AsyncMock,
    stream_process_producer: AsyncMock,
    stream_event_producer: AsyncMock,
) -> StreamService:
    return StreamService(
        klin_repository=stream_repository,
        klin_stream=stream_runner,
        klin_process_producer=stream_process_producer,
        klin_event_producer=stream_event_producer,
    )


@pytest.fixture(name="sample_stream_state")
def fixture_sample_stream_state() -> KlinStreamState:
    return KlinStreamState(
        id=uuid.uuid4(),
        camera_id="cam-1",
        camera_url="rtsp://example/cam-1",
        state=ProcessingState.PENDING,
    )


@pytest.mark.anyio
async def test_start_stream_creates_state_and_publishes_job(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_process_producer: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    upload = StreamUploadDto(camera_id="cam-1", camera_url="rtsp://example/cam-1")
    stream_repository.create.return_value = sample_stream_state

    result = await stream_service.start_stream(upload)

    assert result is sample_stream_state
    stream_repository.create.assert_awaited_once()
    created_model = stream_repository.create.await_args.args[0]
    assert created_model.camera_id == upload.camera_id
    assert created_model.camera_url == upload.camera_url
    assert created_model.state == ProcessingState.PENDING
    stream_process_producer.send_stream.assert_awaited_once()
    payload = stream_process_producer.send_stream.await_args.args[0]
    assert payload.stream_id == sample_stream_state.id


@pytest.mark.anyio
async def test_start_stream_reuses_stopped_stream_and_publishes_job(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_process_producer: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    upload = StreamUploadDto(camera_id="cam-1", camera_url="rtsp://example/new")
    sample_stream_state.state = ProcessingState.STOPPED
    stream_repository.get_by_id_camera.return_value = sample_stream_state

    result = await stream_service.start_stream(upload)

    assert result is sample_stream_state
    assert sample_stream_state.state == ProcessingState.PENDING
    assert sample_stream_state.camera_url == upload.camera_url
    stream_repository.create.assert_not_awaited()
    stream_repository.update.assert_awaited_once_with(sample_stream_state)
    stream_process_producer.send_stream.assert_awaited_once()
    payload = stream_process_producer.send_stream.await_args.args[0]
    assert payload.stream_id == sample_stream_state.id


@pytest.mark.anyio
async def test_start_stream_marks_state_error_when_queue_publish_fails(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_process_producer: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    upload = StreamUploadDto(camera_id="cam-1", camera_url="rtsp://example/cam-1")
    stream_repository.create.return_value = sample_stream_state
    stream_process_producer.send_stream.side_effect = RuntimeError("broker down")

    with pytest.raises(RuntimeError, match="broker down"):
        await stream_service.start_stream(upload)

    assert sample_stream_state.state == ProcessingState.ERROR
    assert sample_stream_state.last_mae_label == "broker down"
    stream_repository.update.assert_awaited_once_with(sample_stream_state)
    stream_process_producer.send_stream.assert_awaited_once()


@pytest.mark.anyio
async def test_perform_stream_skips_when_claim_is_not_acquired(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_runner: AsyncMock,
) -> None:
    stream_id = uuid.uuid4()
    stream_repository.claim_for_processing_stream.return_value = None

    await stream_service.perform_stream(stream_id)

    stream_runner.streaming_analyze.assert_not_awaited()
    stream_repository.update.assert_not_awaited()


@pytest.mark.anyio
async def test_perform_stream_marks_processing_before_analysis(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_runner: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    stream_repository.claim_for_processing_stream.return_value = sample_stream_state

    await stream_service.perform_stream(sample_stream_state.id)

    assert sample_stream_state.state == ProcessingState.PROCESSING
    stream_repository.update.assert_awaited_once_with(sample_stream_state)
    stream_runner.streaming_analyze.assert_awaited_once_with(sample_stream_state)


@pytest.mark.anyio
async def test_perform_stream_marks_cancelled_tasks_stopped(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_runner: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    stream_repository.claim_for_processing_stream.return_value = sample_stream_state
    stream_repository.get_by_id_stream.return_value = sample_stream_state
    stream_runner.streaming_analyze.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await stream_service.perform_stream(sample_stream_state.id)

    assert sample_stream_state.state == ProcessingState.STOPPED
    assert stream_repository.update.await_count == 2


@pytest.mark.anyio
async def test_perform_stream_marks_failures_error(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_runner: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    stream_repository.claim_for_processing_stream.return_value = sample_stream_state
    stream_repository.get_by_id_stream.return_value = sample_stream_state
    stream_runner.streaming_analyze.side_effect = RuntimeError("stream crashed")

    await stream_service.perform_stream(sample_stream_state.id)

    assert sample_stream_state.state == ProcessingState.ERROR
    assert sample_stream_state.last_mae_label == "stream crashed"
    assert stream_repository.update.await_count == 2


@pytest.mark.anyio
async def test_stop_stream_returns_when_state_is_missing(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_event_producer: AsyncMock,
) -> None:
    stream_repository.get_by_id_stream.return_value = None

    await stream_service.stop_stream(uuid.uuid4())

    stream_event_producer.send_event.assert_not_awaited()
    stream_repository.update.assert_not_awaited()


@pytest.mark.anyio
async def test_stop_stream_publishes_stop_request_event(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_event_producer: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    sample_stream_state.state = ProcessingState.PROCESSING
    stream_repository.get_by_id_stream.return_value = sample_stream_state

    await stream_service.stop_stream(sample_stream_state.id)

    stream_event_producer.send_event.assert_awaited_once()
    event = stream_event_producer.send_event.await_args.args[0]
    assert event.type == "STOP_STREAM"
    assert event.camera_id == sample_stream_state.camera_id
    assert event.stream_id == sample_stream_state.id
    assert event.payload == {}
    stream_repository.update.assert_not_awaited()


@pytest.mark.anyio
async def test_stop_stream_propagates_publish_errors(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_event_producer: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    stream_repository.get_by_id_stream.return_value = sample_stream_state
    stream_event_producer.send_event.side_effect = RuntimeError("broker unavailable")

    with pytest.raises(RuntimeError, match="broker unavailable"):
        await stream_service.stop_stream(sample_stream_state.id)

    stream_repository.update.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("state"),
    [ProcessingState.STOPPED, ProcessingState.ERROR],
)
async def test_stop_stream_skips_terminal_states(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    stream_event_producer: AsyncMock,
    sample_stream_state: KlinStreamState,
    state: ProcessingState,
) -> None:
    sample_stream_state.state = state
    stream_repository.get_by_id_stream.return_value = sample_stream_state

    await stream_service.stop_stream(sample_stream_state.id)

    stream_event_producer.send_event.assert_not_awaited()
    stream_repository.update.assert_not_awaited()


@pytest.mark.anyio
async def test_get_stream_status_returns_dto(
    stream_service: StreamService,
    stream_repository: AsyncMock,
    sample_stream_state: KlinStreamState,
) -> None:
    sample_stream_state.state = ProcessingState.FINISHED
    sample_stream_state.last_mae_label = "fight"
    sample_stream_state.last_mae_confidence = 0.91
    stream_repository.get_by_id_stream.return_value = sample_stream_state

    result = await stream_service.get_stream_status(sample_stream_state.id)

    assert isinstance(result, StreamReadDto)
    assert result.id == sample_stream_state.id
    assert result.camera_id == sample_stream_state.camera_id
    assert result.last_mae_label == "fight"


@pytest.mark.anyio
async def test_get_stream_status_raises_for_unknown_stream(
    stream_service: StreamService,
    stream_repository: AsyncMock,
) -> None:
    stream_id = uuid.uuid4()
    stream_repository.get_by_id_stream.return_value = None

    with pytest.raises(ValueError, match=f"Stream not found: {stream_id}"):
        await stream_service.get_stream_status(stream_id)
