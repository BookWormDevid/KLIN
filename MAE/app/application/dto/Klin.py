import uuid

import msgspec

from app.models import KlinModel, ProcessingState


class KlinUploadDto(msgspec.Struct, frozen=True):
    target_url: str
    response_url: str


class KlinResultDto(msgspec.Struct, frozen=True):
    result: str


class KlinReadDto(msgspec.Struct, frozen=True):
    id: uuid.UUID
    result: str | None
    state: ProcessingState

    @classmethod
    def from_model(cls, model: KlinModel) -> "KlinReadDto":
        return KlinReadDto(id=model.id, result=model.result, state=model.state)


class KlinProcessDto(msgspec.Struct, frozen=True):
    klin_id: uuid.UUID
