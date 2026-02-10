import uuid

import msgspec

from app.models import MAEModel, ProcessingState


class MAEUploadDto(msgspec.Struct, frozen=True):
    response_url: str
    video_path: str

class MAEResultDto(msgspec.Struct, frozen=True):
    result: str

class MAEReadDto(msgspec.Struct, frozen=True):
    id: uuid.UUID
    result: str | None
    state: ProcessingState

    @classmethod
    def from_model(cls, model: MAEModel) -> "MAEReadDto":
        return MAEReadDto(id=model.id, result=model.result, state=model.state)


class MAEProcessDto(msgspec.Struct, frozen=True):
    MAE_id: uuid.UUID
