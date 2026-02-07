import uuid
from dataclasses import dataclass

from app.application.dto import KlinProcessDto, KlinReadDto, KlinUploadDto
from app.application.interfaces import (
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
)
from app.models import KlinModel, ProcessingState


@dataclass
class KlinService:
    _klin_repository: IKlinRepository
    _klin_inference_service: IKlinInference
    _klin_process_producer: IKlinProcessProducer

    async def klin_image(self, data: KlinUploadDto) -> KlinModel:
        klin = KlinModel(
            target_url=data.target_url,
            response_url=data.response_url,
            state=ProcessingState.PENDING,
        )
        klin = await self._klin_repository.create(klin)

        await self._klin_process_producer.send(KlinProcessDto(klin_id=klin.id))

        return klin

    async def perform_klin(self, klin_id: uuid.UUID) -> None:
        # пиши сервисную часть

        klin: KlinModel = await self._klin_repository.get_by_id(klin_id)

        try:
            result_klin = await self._klin_inference_service.analyze(klin)
            klin.result = result_klin
            klin.state = ProcessingState.FINISHED
            # callback
            print(f"✅ Успех : {klin.result}")

        except Exception as e:
            klin.result = str(e)
            klin.state = ProcessingState.ERROR
            # callback
            print(f"❌ Ошибка : {klin.result}")

        finally:
            await self._klin_repository.update(klin)

    async def get_inference_status(self, klin_id: uuid.UUID) -> KlinReadDto:
        klin = await self._klin_repository.get_by_id(klin_id)
        if not klin:
            raise ValueError(f"klin {klin_id} not found")
        return KlinReadDto.from_model(klin)
