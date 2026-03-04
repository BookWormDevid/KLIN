"""
Бизнес логика сервиса
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass

from app.application.dto import KlinProcessDto, KlinReadDto, KlinUploadDto
from app.application.exceptions import KlinEnqueueError
from app.application.interfaces import (
    IKlinCallbackSender,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
)
from app.models import KlinModel, ProcessingState


logger = logging.getLogger(__name__)


@dataclass
class KlinService:
    """
    Класс бизнес логики
    """

    _klin_repository: IKlinRepository
    _klin_inference_service: IKlinInference
    _klin_process_producer: IKlinProcessProducer
    _klin_callback_sender: IKlinCallbackSender

    async def _publish_klin_task(self, klin_id: uuid.UUID) -> None:
        """
        Публикация задачи в очередь с повторными попытками.
        """
        max_attempts = 3
        payload = KlinProcessDto(klin_id=klin_id)

        for attempt in range(1, max_attempts + 1):
            try:
                await self._klin_process_producer.send(payload)
                return
            except Exception as exc:
                if attempt == max_attempts:
                    raise KlinEnqueueError(
                        "Failed to enqueue "
                        f"klin_id={klin_id} after {max_attempts} attempts"
                    ) from exc
                await asyncio.sleep(2 ** (attempt - 1))

    async def _persist_enqueue_failure(self, klin: KlinModel, error: Exception) -> None:
        """
        Компенсирующее обновление статуса, если задача не попала в очередь.
        """
        klin.state = ProcessingState.ERROR
        klin.mae = str(error)
        try:
            await self._klin_repository.update(klin)
        except Exception as update_exc:
            logger.exception(
                "Failed to persist enqueue error state. klin_id=%s error=%s",
                klin.id,
                update_exc,
            )

        try:
            await self._klin_callback_sender.post_consumer(klin)
        except Exception as callback_exc:
            logger.exception(
                "Failed to send enqueue error callback. klin_id=%s error=%s",
                klin.id,
                callback_exc,
            )

    async def klin_image(self, data: KlinUploadDto) -> KlinModel:
        """
        Принимает ссылку для отправки результатов, путь видео.
        Создаёт подключение к бд, отправляет запрос к брокеру.
        """
        klin = KlinModel(
            response_url=data.response_url,
            video_path=data.video_path,
            state=ProcessingState.PENDING,
        )
        klin = await self._klin_repository.create(klin)

        try:
            await self._publish_klin_task(klin.id)
        except KlinEnqueueError as exc:
            await self._persist_enqueue_failure(klin, exc)
            logger.exception("Failed to enqueue klin_id=%s", klin.id)
            raise

        return klin

    async def perform_klin(self, klin_id: uuid.UUID) -> None:
        """
        Запускает процессор по id задачи.
        В конце удаляет файл который обработался.
        """
        klin: KlinModel | None = await self._klin_repository.claim_for_processing(
            klin_id
        )
        if klin is None:
            logger.info(
                "Skip klin processing due to idempotency guard. klin_id=%s",
                klin_id,
            )
            return

        try:
            process = await self._klin_inference_service.analyze(klin)
            klin.mae = process.mae
            klin.yolo = process.yolo
            klin.objects = process.objects if process.objects is not None else []
            klin.all_classes = (
                process.all_classes if process.all_classes is not None else []
            )
            klin.state = ProcessingState.FINISHED
            await self._klin_callback_sender.post_consumer(klin)
            logger.info("Klin processing succeeded. klin_id=%s", klin_id)

        except Exception as exc:
            klin.mae = str(exc)
            klin.state = ProcessingState.ERROR
            await self._klin_callback_sender.post_consumer(klin)
            logger.exception(
                "Klin processing failed. klin_id=%s error=%s",
                klin_id,
                exc,
            )

        finally:
            try:
                await self._klin_repository.update(klin)
            except Exception as exc:
                logger.exception(
                    "Failed to persist klin state. klin_id=%s error=%s",
                    klin_id,
                    exc,
                )

            if klin.video_path and os.path.exists(klin.video_path):
                try:
                    os.remove(klin.video_path)
                except OSError as exc:
                    logger.warning(
                        "Failed to delete temp file. path=%s error=%s",
                        klin.video_path,
                        exc,
                    )

    async def get_inference_status(self, klin_id: uuid.UUID) -> KlinReadDto:
        """
        Получение вывода по id
        """
        klin = await self._klin_repository.get_by_id(klin_id)
        return KlinReadDto.from_model(klin)

    async def get_n_imferences(self, count: int) -> list[KlinModel]:
        """
        Получение вывода последних n-количества строк в бд.
        """
        imfer_list = await self._klin_repository.get_first_n(count)
        return imfer_list
