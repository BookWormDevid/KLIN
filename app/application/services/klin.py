"""
Бизнес логика сервиса
"""

# pylint: disable= broad-exception-caught
import os
import uuid
from dataclasses import dataclass

from app.application.dto import KlinProcessDto, KlinReadDto, KlinUploadDto
from app.application.interfaces import (IKlinCallbackSender, IKlinInference,
                                        IKlinProcessProducer, IKlinRepository)
from app.models import KlinModel, ProcessingState


@dataclass
class KlinService:
    """
    Класс бизнес логики
    """

    _klin_repository: IKlinRepository
    _klin_inference_service: IKlinInference
    _klin_process_producer: IKlinProcessProducer
    _klin_callback_sender: IKlinCallbackSender

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

        await self._klin_process_producer.send(KlinProcessDto(klin_id=klin.id))

        return klin

    async def perform_klin(self, klin_id: uuid.UUID) -> None:
        """
        Запускает процессор по id задачи.
        В конце удаляет файл который обработался.
        """
        klin: KlinModel = await self._klin_repository.get_by_id(klin_id)

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
            print(
                f"✅ Успех : {klin.mae}, {klin.yolo}, {klin.objects}, {klin.all_classes}"
            )

        except Exception as e:
            klin.mae = str(e)
            klin.state = ProcessingState.ERROR
            await self._klin_callback_sender.post_consumer(klin)
            print(f"❌ Ошибка : {klin.mae}")

        finally:
            try:
                await self._klin_repository.update(klin)
                if klin.video_path and os.path.exists(klin.video_path):
                    os.remove(klin.video_path)
            except Exception as e:
                print(f"Не удалось удалить временный файл. {e}")

    async def get_inference_status(self, klin_id: uuid.UUID) -> KlinReadDto:
        """
        Получение вывода по id
        """
        klin = await self._klin_repository.get_by_id(klin_id)
        if not klin:
            raise ValueError(f"MAE {klin_id} not found")
        return KlinReadDto.from_model(klin)

    async def get_n_imferences(self, count: int) -> list[KlinModel]:
        """
        Получение вывода последних n-количества строк в бд.
        """
        imfer_list = await self._klin_repository.get_first_n(count)
        return imfer_list
