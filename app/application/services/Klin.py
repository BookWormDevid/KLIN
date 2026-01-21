import uuid
from dataclasses import dataclass

from app.application.dto import KlinProcessDto, KlinUploadDto
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
            state=ProcessingState.PENDING,
        )
        klin = await self._klin_repository.create(klin)

        await self._klin_process_producer.send(KlinProcessDto(klin_id=klin.id))

        return klin

    async def perform_klin(self, klin_id: uuid.UUID) -> None:
        klin = await self._klin_repository.get_by_id(klin_id)
        await self._klin_repository.update(klin)

        result_klin = await self._klin_inference_service.analyze(klin)
        klin.result = result_klin
        klin.state = ProcessingState.FINISHED

        await self._klin_repository.update(klin)

    async def get_klin_status(self, klin_id: uuid.UUID) -> KlinModel:
        klin = await self._klin_repository.get_by_id(klin_id)
        return klin
