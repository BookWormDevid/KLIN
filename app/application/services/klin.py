"""
Бизнес-логика жизненного цикла задач Klin.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.application.dto import KlinProcessDto, KlinReadDto, KlinUploadDto
from app.application.exceptions import KlinEnqueueError
from app.application.interfaces import (
    IKlinCallbackSender,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRuntimeSettings,
    IKlinTaskRepository,
    IKlinVideoStorage,
)
from app.application.mappers import to_klin_read_dto
from app.models import KlinModel, ProcessingState


logger = logging.getLogger(__name__)


@dataclass
class KlinService:
    """
    Координирует постановку в очередь, обработку, callback и очистку артефактов.
    """

    _klin_repository: IKlinTaskRepository
    _klin_inference_service: IKlinInference
    _klin_process_producer: IKlinProcessProducer
    _klin_callback_sender: IKlinCallbackSender
    _klin_video_storage: IKlinVideoStorage
    _runtime_settings: IKlinRuntimeSettings

    async def _publish_klin_task(self, klin_id: uuid.UUID) -> None:
        """
        Публикует задачу на обработку с повторными попытками.
        """

        max_attempts = self._runtime_settings.max_retry_attempts
        payload = KlinProcessDto(klin_id=klin_id)

        for attempt in range(1, max_attempts + 1):
            try:
                await self._klin_process_producer.send(payload)
                return
            except Exception as exc:  # pylint: disable=broad-except
                if attempt >= max_attempts:
                    raise KlinEnqueueError(
                        "Failed to enqueue "
                        f"klin_id={klin_id} after {max_attempts} attempts"
                    ) from exc
                await asyncio.sleep(2 ** (attempt - 1))

    async def _persist_enqueue_failure(self, klin: KlinModel, error: Exception) -> None:
        """
        Сохраняет финальное состояние ошибки,
        если задачу не удалось поставить в очередь.
        """

        klin.state = ProcessingState.ERROR
        klin.mae = str(error)
        try:
            await self._klin_repository.update(klin)
        except Exception as update_exc:  # pylint: disable=broad-except
            logger.exception(
                "Failed to persist enqueue error state. klin_id=%s error=%s",
                klin.id,
                update_exc,
            )

        try:
            await self._klin_callback_sender.post_consumer(klin)
        except Exception as callback_exc:  # pylint: disable=broad-except
            logger.exception(
                "Failed to send enqueue error callback. klin_id=%s error=%s",
                klin.id,
                callback_exc,
            )

    @staticmethod
    def _is_s3_uri(video_path: str) -> bool:
        return video_path.startswith("s3://")

    @staticmethod
    def _build_temp_suffix(video_path: str) -> str:
        parsed_path = (
            urlparse(video_path).path if video_path.startswith("s3://") else video_path
        )
        suffix = Path(parsed_path).suffix
        return suffix or ".mp4"

    async def _prepare_processing_video(
        self,
        video_path: str,
    ) -> tuple[str, str | None]:
        if not self._is_s3_uri(video_path):
            return video_path, None

        fd, local_path = tempfile.mkstemp(suffix=self._build_temp_suffix(video_path))
        os.close(fd)

        try:
            await self._klin_video_storage.download_to_path(
                source_uri=video_path,
                destination_path=local_path,
            )
        except Exception:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

        return local_path, local_path

    async def _cleanup_video_artifacts(
        self,
        *,
        source_video_path: str,
        local_video_path: str | None,
    ) -> None:
        if self._is_s3_uri(source_video_path):
            if not self._runtime_settings.keep_s3_source_objects:
                try:
                    await self._klin_video_storage.delete(source_video_path)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to delete S3 object. uri=%s error=%s",
                        source_video_path,
                        exc,
                    )

            if local_video_path and os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                except OSError as exc:
                    logger.warning(
                        "Failed to delete temp file. path=%s error=%s",
                        local_video_path,
                        exc,
                    )
            return

        if source_video_path and os.path.exists(source_video_path):
            try:
                os.remove(source_video_path)
            except OSError as exc:
                logger.warning(
                    "Failed to delete temp file. path=%s error=%s",
                    source_video_path,
                    exc,
                )

    async def klin_image(self, data: KlinUploadDto) -> KlinModel:
        """
        Создает задачу в БД и ставит ее в очередь фоновой обработки.
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
        Захватывает задачу, выполняет инференс, сохраняет статус и очищает хранилище.
        """

        klin: KlinModel | None = await self._klin_repository.claim_for_processing(
            klin_id
        )
        if klin is None:
            logger.info(
                "Klin processig skipped because claim was not aquired. klin_id=%s",
                klin_id,
            )
            return

        source_video_path = klin.video_path
        local_video_path: str | None = None
        processing_succeeded = False

        try:
            (
                processing_video_path,
                local_video_path,
            ) = await self._prepare_processing_video(source_video_path)
            logger.info(
                "Prepared video for processing. klin_id=%s"
                " source_video_path=%s processing_video_path=%s",
                klin_id,
                source_video_path,
                processing_video_path,
            )
            klin.video_path = processing_video_path

            process = await self._klin_inference_service.analyze(klin)
            klin.video_path = source_video_path
            klin.x3d = process.x3d
            klin.mae = process.mae
            klin.yolo = process.yolo
            klin.objects = process.objects if process.objects is not None else []
            klin.all_classes = (
                process.all_classes if process.all_classes is not None else []
            )
            klin.state = ProcessingState.FINISHED
            processing_succeeded = True

        except Exception as exc:  # pylint: disable=broad-except
            klin.video_path = source_video_path
            klin.x3d = "Execution error. View mae column."
            klin.mae = str(exc)
            klin.state = ProcessingState.ERROR
            logger.exception(
                "Klin processing failed. klin_id=%s error=%s",
                klin_id,
                exc,
            )

        finally:
            klin.video_path = source_video_path
            try:
                await self._klin_repository.update(klin)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    "Failed to persist klin state. klin_id=%s error=%s",
                    klin_id,
                    exc,
                )

            try:
                await self._klin_callback_sender.post_consumer(klin)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    "Failed to send callback. klin_id=%s error=%s",
                    klin_id,
                    exc,
                )

            await self._cleanup_video_artifacts(
                source_video_path=source_video_path,
                local_video_path=local_video_path,
            )

        if processing_succeeded:
            logger.info("Klin processing succeeded. klin_id=%s", klin_id)

    async def get_inference_status(self, klin_id: uuid.UUID) -> KlinReadDto:
        """
        Возвращает статус задачи по идентификатору.
        """

        klin = await self._klin_repository.get_by_id(klin_id)
        return to_klin_read_dto(klin)

    async def get_n_imferences(self, count: int) -> list[KlinModel]:
        """
        Возвращает последние N задач.
        """

        return await self._klin_repository.get_first_n(count)
