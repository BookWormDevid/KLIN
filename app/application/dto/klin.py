# pylint: disable=too-few-public-methods
"""
DTO для работы с проектом.
"""

import uuid
from dataclasses import dataclass

import msgspec

from app.models import KlinModel, KlinStreamingModel, ProcessingState


class KlinUploadDto(msgspec.Struct, frozen=True):
    """
    DTO для загрузки видео на обработку.
    Содержит URL для ответа и путь к видеофайлу.
    """

    video_path: str
    response_url: str | None = None


class KlinResultDto(msgspec.Struct, frozen=True):
    """
    DTO для результатов обработки MAE.
    Включает MAE-результат, YOLO-анализ,
    найденные координаты bounding box
    и все классы найденные MAE.
    """

    x3d: str | None
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None


class KlinReadDto(msgspec.Struct, frozen=True):
    """
    DTO для чтения результата из базы данных.
    Включает идентификатор, результаты обработки и текущее состояние.
    """

    id: uuid.UUID
    x3d: str | None
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None
    state: ProcessingState

    @classmethod
    def from_model(cls, model: KlinModel) -> "KlinReadDto":
        """
        Создает MAEReadDto из модели базы данных MAEModel.
        """
        return KlinReadDto(
            id=model.id,
            x3d=model.x3d,
            mae=model.mae,
            yolo=model.yolo,
            objects=model.objects,
            all_classes=model.all_classes,
            state=model.state,
        )


class KlinProcessDto(msgspec.Struct, frozen=True):
    """
    DTO для передачи идентификатора задачи в очередь обработки.
    """

    klin_id: uuid.UUID


class StreamProcessDto(msgspec.Struct, frozen=True):
    """
    DTO для передачи идентификатора задачи в очередь обработки.
    """

    stream_id: uuid.UUID


class StreamUploadDto(msgspec.Struct, frozen=True):
    camera_url: str
    camera_id: str


class StreamResultDto(msgspec.Struct, frozen=True):
    x3d: str | None
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None


class StreamReadDto(msgspec.Struct, frozen=True):
    id: uuid.UUID
    x3d: str | None
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None
    state: ProcessingState

    @classmethod
    def from_streaming_model(cls, model: KlinStreamingModel) -> "StreamReadDto":
        return StreamReadDto(
            id=model.id,
            x3d=model.x3d,
            mae=model.mae,
            yolo=model.yolo,
            objects=model.objects,
            all_classes=model.all_classes,
            state=model.state,
        )


@dataclass
class StreamEventDto:
    id: str
    stream_id: uuid.UUID
    camera_id: str
    type: str
    payload: dict
