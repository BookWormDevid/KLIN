"""
Бизнес-логика для запуска и сопровождения потоковой обработки.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from app.application.dto import (
    StreamEventDto,
    StreamProcessDto,
    StreamReadDto,
    StreamUploadDto,
)
from app.application.interfaces import (
    IKlinEventProducer,
    IKlinProcessProducer,
    IKlinStream,
    IStreamStateRepository,
)
from app.application.mappers import to_stream_read_dto
from app.models import KlinStreamState, ProcessingState


logger = logging.getLogger(__name__)


class StreamService:
    """
    Сервис управления жизненным циклом видеопотоков.
    """

    def __init__(
        self,
        klin_repository: IStreamStateRepository,
        klin_stream: IKlinStream,
        klin_process_producer: IKlinProcessProducer,
        klin_event_producer: IKlinEventProducer,
    ):
        self._klin_repository = klin_repository
        self._klin_stream = klin_stream
        self._klin_process_producer = klin_process_producer
        self._klin_event_producer = klin_event_producer

    async def start_stream(self, data: StreamUploadDto) -> KlinStreamState:
        """Create a stream state record and enqueue background processing."""
        stream_state = KlinStreamState(
            camera_url=data.camera_url,
            camera_id=data.camera_id,
            state=ProcessingState.PENDING,
        )
        stream_state = await self._klin_repository.create(stream_state)

        try:
            await self._publish_stream_task(stream_state.id)
        except Exception as exc:
            await self._persist_start_failure(stream_state, exc)
            raise

        logger.info(
            "Stream started: camera_id=%s, stream_id=%s",
            data.camera_id,
            stream_state.id,
        )
        return stream_state

    async def _publish_stream_task(self, stream_id: uuid.UUID) -> None:
        max_attempts = 3
        payload = StreamProcessDto(stream_id=stream_id)

        for attempt in range(1, max_attempts + 1):
            try:
                await self._klin_process_producer.send_stream(payload)
                return
            except Exception:
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))

    async def _persist_start_failure(
        self, stream_state: KlinStreamState, error: Exception
    ) -> None:
        stream_state.state = ProcessingState.ERROR
        self._store_error_message(stream_state, error)

        try:
            await self._klin_repository.update(stream_state)
        except Exception:
            logger.exception("Failed to persist stream error state")

    @staticmethod
    def _store_error_message(stream_state: KlinStreamState, error: Exception) -> None:
        stream_state.last_mae_label = str(error)[:255]
        stream_state.last_mae_confidence = None

    async def perform_stream(self, stream_id: uuid.UUID) -> None:
        """Claim a pending stream, run analysis, and persist the final state."""
        stream_state = await self._klin_repository.claim_for_processing_stream(
            stream_id
        )

        if stream_state is None:
            logger.info("Stream skipped (already processing). id=%s", stream_id)
            return

        stream_state.state = ProcessingState.PROCESSING
        await self._klin_repository.update(stream_state)

        try:
            await self._klin_stream.streaming_analyze(stream_state)

        except asyncio.CancelledError:
            logger.info("Stream cancelled id=%s", stream_id)
            stream_state.state = ProcessingState.STOPPED

        except Exception as exc:
            logger.exception("Stream failed id=%s error=%s", stream_id, exc)
            stream_state.state = ProcessingState.ERROR
            self._store_error_message(stream_state, exc)

        finally:
            try:
                await self._klin_repository.update(stream_state)
            except Exception:
                logger.exception("Failed to update stream state")

    async def stop_stream(self, stream_id: uuid.UUID) -> None:
        """Request a graceful stop for the active stream and publish an event."""
        stream_state = await self._klin_repository.get_by_id_stream(stream_id)

        if not stream_state:
            logger.warning("Stream not found: %s", stream_id)
            return

        await self._klin_stream.stop(stream_state.camera_id)

        stopped = await self._klin_stream.wait_stopped(stream_state.camera_id)

        if stopped:
            try:
                await self._klin_event_producer.send_event(
                    StreamEventDto(
                        id=str(uuid.uuid4()),
                        type="STREAM_STOPPED",
                        camera_id=stream_state.camera_id,
                        stream_id=stream_id,
                        payload={"timestamp": datetime.now(timezone.utc).isoformat()},
                    )
                )
            except Exception:
                logger.exception("Failed to publish STREAM_STOPPED event")
        else:
            logger.warning("Processor did not stop in time. stream_id=%s", stream_id)

        # защита от перезаписи состояния
        if stream_state.state != ProcessingState.STOPPED:
            stream_state.state = ProcessingState.STOPPED

        try:
            await self._klin_repository.update(stream_state)
        except Exception:
            logger.exception("Failed to update stream state")

    async def get_stream_status(self, stream_id: uuid.UUID) -> StreamReadDto:
        """Return the latest persisted status snapshot for the requested stream."""

        stream_state = await self._klin_repository.get_by_id_stream(stream_id)

        if not stream_state:
            raise ValueError(f"Stream not found: {stream_id}")

        return to_stream_read_dto(stream_state)
