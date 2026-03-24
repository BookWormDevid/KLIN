"""
Бизнес-логика для запуска и сопровождения потоковой обработки.
"""

import asyncio
import logging
import uuid
from typing import cast

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
from app.models import KlinStreamState, ProcessingState


logger = logging.getLogger(__name__)


class StreamService:
    """
    Сервис управления жизненным циклом видеопотоков.
    """

    _klin_stream: IKlinStream
    _klin_repository: IKlinRepository
    _klin_process_producer: IKlinProcessProducer

    async def start_stream(self, data: StreamUploadDto) -> KlinStreamState:
        """
        Создает задачу потока и отправляет ее в очередь.
        """
        stream_state = KlinStreamState(
            camera_url=data.camera_url,
            camera_id=data.camera_id,
            state=ProcessingState.PENDING,
        )
        created = await self._klin_repository.create(stream_state)
        stream_state = cast(KlinStreamState, created)  # ← вот исправление

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
        # Сохраняем ошибку в поле error_message (если такого поля нет, нужно добавить)
        # Или используем существующее поле, например, last_mae_label для хранения ошибки
        if hasattr(stream_state, "error_message"):
            stream_state.error_message = str(error)
        else:
            # Временно используем last_mae_label для хранения ошибки
            # Рекомендуется добавить поле error_message в модель
            stream_state.last_mae_label = str(error)[:255]  # Обрезаем до 255 символов

        try:
            await self._klin_repository.update(stream_state)
        except Exception:
            logger.exception("Failed to persist stream error state")

    async def perform_stream(self, stream_id: uuid.UUID) -> None:
        """
        Запускает обработку уже созданного потока.
        """
        stream_state: (
            KlinStreamState | None
        ) = await self._klin_repository.claim_for_processing_stream(stream_id)
        if stream_state is None:
            logger.info("Stream skipped (already processing). id=%s", stream_id)
            return
        stream_state.state = ProcessingState.PROCESSING
        await self._klin_repository.update(stream_state)
        try:
            await self._klin_stream.streaming_analyze(stream_state)
            stream_state.state = ProcessingState.FINISHED

        except asyncio.CancelledError:
            logger.info("Stream cancelled id=%s", stream_id)
            stream_state.state = ProcessingState.FINISHED

        except Exception as exc:
            logger.exception("Stream failed id=%s error=%s", stream_id, exc)
            stream_state.state = ProcessingState.ERROR
            # Сохраняем ошибку в поле error_message
            if hasattr(stream_state, "error_message"):
                stream_state.error_message = str(exc)
            else:
                # Временно используем last_mae_label для хранения ошибки
                stream_state.last_mae_label = str(exc)[:255]
        finally:
            try:
                await self._klin_repository.update(stream_state)
            except Exception:
                logger.exception("Failed to update stream state")

    async def stop_stream(self, stream_id: uuid.UUID) -> None:
        stream_state = await self._klin_repository.get_by_id_stream(stream_id)
        if not stream_state:
            return

        await self._klin_stream.stop(stream_state.camera_id)

        stopped = await self._klin_stream.wait_stopped(stream_state.camera_id)

        if not stopped:
            logger.warning("Processor did not stop in time")

        stream_state.state = ProcessingState.STOPPED
        await self._klin_repository.update(stream_state)

    async def get_stream_status(self, stream_id: uuid.UUID) -> StreamReadDto:
        """Возвращает актуальное состояние потока (агрегат)."""
        stream_state = await self._klin_repository.get_by_id_stream(stream_id)
        if not stream_state:
            raise ValueError(f"Stream not found: {stream_id}")

        return StreamReadDto.from_stream_state(stream_state)
