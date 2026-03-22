import asyncio
import logging
import uuid

from app.application.dto import (
    StreamProcessDto,
    StreamReadDto,
    StreamUploadDto,
)
from app.application.interfaces import (
    IKlinProcessProducer,
    IKlinRepository,
    IKlinStream,
)
from app.models import KlinStreamingModel, ProcessingState


logger = logging.getLogger(__name__)


class StreamService:
    _klin_stream: IKlinStream
    _klin_repository: IKlinRepository
    _klin_process_producer: IKlinProcessProducer

    async def start_stream(self, data: StreamUploadDto) -> KlinStreamingModel:
        stream = KlinStreamingModel(
            camera_url=data.camera_url,
            camera_id=data.camera_id,
            state=ProcessingState.PENDING,
        )
        stream = await self._klin_repository.create_stream(stream)

        try:
            await self._publish_stream_task(stream.id)
        except Exception as exc:
            await self._persist_start_failure(stream, exc)
            raise

        return stream

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
        self, stream: KlinStreamingModel, error: Exception
    ) -> None:
        stream.state = ProcessingState.ERROR
        stream.mae = str(error)

        try:
            await self._klin_repository.update_stream(stream)
        except Exception:
            logger.exception("Failed to persist stream error state")

    async def perform_stream(self, stream_id: uuid.UUID) -> None:
        stream: (
            KlinStreamingModel | None
        ) = await self._klin_repository.claim_for_processing_stream(stream_id)
        if stream is None:
            logger.info("Stream skipped (already processing). id=%s", stream_id)
            return
        stream.state = ProcessingState.PROCESSING
        await self._klin_repository.update_stream(stream)
        try:
            await self._klin_stream.streaming_analyze(stream)
            # ❗ ВАЖНО:
            # стрим не "завершается", но если вышли — считаем FINISHED
            stream.state = ProcessingState.FINISHED

        except asyncio.CancelledError:
            logger.info("Stream cancelled id=%s", stream_id)
            stream.state = ProcessingState.FINISHED

        except Exception as exc:
            logger.exception("Stream failed id=%s error=%s", stream_id, exc)
            stream.state = ProcessingState.ERROR
            stream.mae = str(exc)
        finally:
            try:
                await self._klin_repository.update_stream(stream)
            except Exception:
                logger.exception("Failed to update stream state")

    async def stop_stream(self, stream_id: uuid.UUID) -> None:
        stream = await self._klin_repository.get_by_id_stream(stream_id)
        if not stream:
            return

        # 🔴 ВАЖНО: сначала остановить processor
        # self._klin_stream.stop(stream.camera_id)

        # затем обновить состояние
        stream.state = ProcessingState.FINISHED
        await self._klin_repository.update_stream(stream)

    async def get_stream_status(self, stream_id: uuid.UUID) -> StreamReadDto:
        stream = await self._klin_repository.get_by_id_stream(stream_id)
        return StreamReadDto.from_streaming_model(stream)
