# pylint: disable=too-few-public-methods
"""
DTO для работы с проектом.
"""

import uuid

import msgspec

from app.models import KlinModel, ProcessingState


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
