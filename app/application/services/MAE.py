import uuid
from dataclasses import dataclass

from app.application.dto import MAEProcessDto, MAEReadDto, MAEUploadDto
from app.application.interfaces import (
    IMAEInference,
    IMAEProcessProducer,
    IMAERepository,
)
from app.models import MAEModel, ProcessingState


@dataclass
class MAEService:
    _MAE_repository: IMAERepository
    _MAE_inference_service: IMAEInference
    _MAE_process_producer: IMAEProcessProducer

    async def MAE_image(self, data: MAEUploadDto) -> MAEModel:
        MAE = MAEModel(
            response_url=data.response_url,
            video_path=data.video_path,
            state=ProcessingState.PENDING,
        )
        MAE = await self._MAE_repository.create(MAE)

        await self._MAE_process_producer.send(MAEProcessDto(MAE_id=MAE.id))

        return MAE

    async def perform_MAE(self, MAE_id: uuid.UUID) -> None:
        # пиши сервисную часть

        MAE: MAEModel = await self._MAE_repository.get_by_id(MAE_id)

        try:
            result_MAE = await self._MAE_inference_service.analyze(MAE)
            MAE.result = result_MAE
            MAE.state = ProcessingState.FINISHED
            # callback
            print(f"✅ Успех : {MAE.result}")

        except Exception as e:
            MAE.result = str(e)
            MAE.state = ProcessingState.ERROR
            # callback
            print(f"❌ Ошибка : {MAE.result}")

        finally:
            await self._MAE_repository.update(MAE)

    async def get_inference_status(self, mae_id: uuid.UUID) -> MAEReadDto:
        mae = await self._MAE_repository.get_by_id(mae_id)
        if not mae:
            raise ValueError(f"MAE {mae_id} not found")
        return MAEReadDto.from_model(mae)
