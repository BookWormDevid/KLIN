import uuid

import msgspec

from app.models import ProcessingState, YoloModel


class YoloUploadDto(msgspec.Struct, frozen=True):
    target_url: str
    response_url: str


class YoloResultDto(msgspec.Struct, frozen=True):
    result: str


class YoloReadDto(msgspec.Struct, frozen=True):
    id: uuid.UUID
    result: str | None
    state: ProcessingState

    @classmethod
    def from_model(cls, model: YoloModel) -> "YoloReadDto":
        return YoloReadDto(id=model.id, result=model.result, state=model.state)


class YoloProcessDto(msgspec.Struct, frozen=True):
    yolo_id: uuid.UUID
