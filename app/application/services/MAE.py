import os
import uuid
from dataclasses import dataclass

from app.application.dto import MAEProcessDto, MAEReadDto, MAEUploadDto
from app.application.interfaces import (
    IMAECallbackSender,
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
    _MAE_callback_sender: IMAECallbackSender

    async def MAE_image(self, data: MAEUploadDto) -> MAEModel:
        mae = MAEModel(
            response_url=data.response_url,
            video_path=data.video_path,
            state=ProcessingState.PENDING,
        )
        mae = await self._MAE_repository.create(mae)

        await self._MAE_process_producer.send(MAEProcessDto(MAE_id=mae.id))

        return mae

    async def perform_MAE(self, MAE_id: uuid.UUID) -> None:
        mae: MAEModel = await self._MAE_repository.get_by_id(MAE_id)

        try:
            process = await self._MAE_inference_service.analyze(mae)
            mae.result = process.result
            mae.state = ProcessingState.FINISHED
            await self._MAE_callback_sender.post_consumer(mae)
            print(f"✅ Успех : {mae.result}")

        except Exception as e:
            mae.result = str(e)
            mae.state = ProcessingState.ERROR
            await self._MAE_callback_sender.post_consumer(mae)
            print(f"❌ Ошибка : {mae.result}")

        finally:
            try:
                if mae.video_path and os.path.exists(mae.video_path):
                    os.remove(mae.video_path)
            except Exception as e:
                print(f"Не удалось удалить временный файл. {e}")

    async def get_inference_status(self, mae_id: uuid.UUID) -> MAEReadDto:
        mae = await self._MAE_repository.get_by_id(mae_id)
        if not mae:
            raise ValueError(f"MAE {mae_id} not found")
        return MAEReadDto.from_model(mae)
