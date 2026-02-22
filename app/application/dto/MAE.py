import uuid

import msgspec

from app.models import MAEModel, ProcessingState


class MAEUploadDto(msgspec.Struct, frozen=True):
    response_url: str
    video_path: str


class MAEResultDto(msgspec.Struct, frozen=True):
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None


class MAEReadDto(msgspec.Struct, frozen=True):
    id: uuid.UUID
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None
    state: ProcessingState

    @classmethod
    def from_model(cls, model: MAEModel) -> "MAEReadDto":
        return MAEReadDto(
            id=model.id,
            mae=model.mae,
            yolo=model.yolo,
            objects=model.objects,
            all_classes=model.all_classes,
            state=model.state,
        )


class MAEProcessDto(msgspec.Struct, frozen=True):
    MAE_id: uuid.UUID
