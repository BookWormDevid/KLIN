"""
Бизнес-логика для запуска и сопровождения потоковой обработки.
"""

import asyncio
import logging
import uuid

from sqlalchemy.exc import IntegrityError

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
        existing = await self._klin_repository.get_by_id_camera(data.camera_id)

        if existing and existing.state in (
            ProcessingState.PROCESSING,
            ProcessingState.PENDING,
        ):
            return existing

        if existing:
            existing.camera_url = data.camera_url
            existing.state = ProcessingState.PENDING
            await self._klin_repository.update(existing)

            try:
                await self._publish_stream_task(existing.id)
            except Exception as exc:
                await self._persist_start_failure(existing, exc)
                raise

            return existing

        stream_state = KlinStreamState(
            camera_url=data.camera_url,
            camera_id=data.camera_id,
            state=ProcessingState.PENDING,
        )

        try:
            stream_state = await self._klin_repository.create(stream_state)
        except IntegrityError as err:
            existing = await self._klin_repository.get_by_id_camera(data.camera_id)

            if existing is None:
                raise RuntimeError(
                    "Stream creation race failed: record not found"
                ) from err

            return existing

        try:
            await self._publish_stream_task(stream_state.id)
        except Exception as exc:
            await self._persist_start_failure(stream_state, exc)
            raise

        return stream_state

    async def _publish_stream_task(self, stream_id: uuid.UUID) -> None:
        payload = StreamProcessDto(stream_id=stream_id)
        await self._klin_process_producer.send_stream(payload)

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
        stream_state = await self._klin_repository.claim_for_processing_stream(
            stream_id
        )

        if stream_state is None:
            logger.info("Stream skipped (already processing). id=%s", stream_id)
            return

        stream_state.state = ProcessingState.PROCESSING
        await self._klin_repository.update(stream_state)

        final_state: ProcessingState | None = None

        try:
            await self._klin_stream.streaming_analyze(stream_state)
            final_state = ProcessingState.FINISHED

        except asyncio.CancelledError:
            logger.info("Stream cancelled id=%s", stream_id)
            final_state = ProcessingState.STOPPED

        except Exception as exc:
            logger.exception("Stream failed id=%s error=%s", stream_id, exc)
            final_state = ProcessingState.ERROR
            self._store_error_message(stream_state, exc)

        finally:
            try:
                # 🔥 КЛЮЧ: читаем актуальное состояние
                current = await self._klin_repository.get_by_id_stream(stream_id)

                if (
                    final_state is not None
                    and current
                    and current.state == ProcessingState.PROCESSING
                ):
                    stream_state.state = final_state
                    await self._klin_repository.update(stream_state)

            except Exception:
                logger.exception("Failed to update stream state")

    async def stop_stream(self, stream_id: uuid.UUID) -> None:
        stream_state = await self._klin_repository.get_by_id_stream(stream_id)

        if not stream_state:
            logger.warning("Stream not found: %s", stream_id)
            return

        if stream_state.state in (ProcessingState.STOPPED, ProcessingState.ERROR):
            logger.info(
                "Stop skipped (already stopped): camera_id=%s stream_id=%s",
                stream_state.camera_id,
                stream_id,
            )
            return

        await self._klin_event_producer.send_event(
            StreamEventDto(
                id=str(uuid.uuid4()),
                type="STOP_STREAM",
                camera_id=stream_state.camera_id,
                stream_id=stream_id,
                payload={},
            )
        )

        logger.info(
            "STOP_STREAM event sent: camera_id=%s stream_id=%s",
            stream_state.camera_id,
            stream_id,
        )

    async def get_stream_status(self, stream_id: uuid.UUID) -> StreamReadDto:
        """Return the latest persisted status snapshot for the requested stream."""

        stream_state = await self._klin_repository.get_by_id_stream(stream_id)

        if not stream_state:
            raise ValueError(f"Stream not found: {stream_id}")

        return to_stream_read_dto(stream_state)
