import uuid
from dataclasses import dataclass

from app.application.dto import YoloProcessDto, YoloReadDto, YoloUploadDto
from app.application.interfaces import (
    IYoloInference,
    IYoloProcessProducer,
    IYoloRepository,
)
from app.models import ProcessingState, YoloModel


@dataclass
class YoloService:
    _yolo_repository: IYoloRepository
    _yolo_inference_service: IYoloInference
    _yolo_process_producer: IYoloProcessProducer

    async def yolo_image(self, data: YoloUploadDto) -> YoloModel:
        yolo = YoloModel(
            target_url=data.target_url,
            response_url=data.response_url,
            state=ProcessingState.PENDING,
        )
        yolo = await self._yolo_repository.create(yolo)

        await self._yolo_process_producer.send(YoloProcessDto(yolo_id=yolo.id))

        return yolo

    async def perform_yolo(self, yolo_id: uuid.UUID) -> None:
        # пиши сервисную часть

        yolo: YoloModel = await self._yolo_repository.get_by_id(yolo_id)

        try:
            result_yolo = await self._yolo_inference_service.analyze(yolo)
            yolo.result = result_yolo
            yolo.state = ProcessingState.FINISHED
            # callback
            print(f"✅ Успех : {yolo.result}")

        except Exception as e:
            yolo.result = str(e)
            yolo.state = ProcessingState.ERROR
            # callback
            print(f"❌ Ошибка : {yolo.result}")

        finally:
            await self._yolo_repository.update(yolo)

    async def get_inference_status(self, yolo_id: uuid.UUID) -> YoloReadDto:
        yolo = await self._yolo_repository.get_by_id(yolo_id)
        if not yolo:
            raise ValueError(f"yolo {yolo_id} not found")
        return YoloReadDto.from_model(yolo)
