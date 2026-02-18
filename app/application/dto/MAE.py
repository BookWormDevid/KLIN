import uuid
import msgspec

from app.models import MAEModel, ProcessingState


class MAEUploadDto(msgspec.Struct, frozen=True):
    response_url: str
    video_path: str


class MAEResultDto(msgspec.Struct, frozen=True):
    event: str | None
    confidence: float | None
    objects: list[str]

class MAEReadDto(msgspec.Struct, frozen=True):
    id: uuid.UUID
    event: str | None
    confidence: float | None
    objects: list[str]
    state: ProcessingState

    @classmethod
    def from_model(cls, model: MAEModel) -> "MAEReadDto":
        return MAEReadDto(
            id=model.id,
            event=model.event,
            confidence=model.confidence,
            objects=model.objects,
            state=model.state,
        )


class TimeReadDto(msgspec.Struct, frozen=True):
    datetime: str
    data: MAEReadDto

class MAEProcessDto(msgspec.Struct, frozen=True):
    MAE_id: uuid.UUID
