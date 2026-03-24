# pylint: disable=too-few-public-methods
"""
DTO для работы с проектом.
"""

import uuid
from dataclasses import dataclass

import msgspec

from app.models import KlinModel, KlinStreamState, ProcessingState


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
        Создает KlinReadDto из модели базы данных.
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
    klin_id: uuid.UUID


class StreamProcessDto(msgspec.Struct, frozen=True):
    stream_id: uuid.UUID


class StreamUploadDto(msgspec.Struct, frozen=True):
    camera_url: str
    camera_id: str


# ===================== НОВЫЕ DTO ДЛЯ СТРИМИНГА =====================


class StreamResultDto(msgspec.Struct, frozen=True):
    """Результаты потоковой обработки (можно использовать для API ответов)."""

    x3d: str | None = None
    x3d_confidence: float | None = None
    mae: str | None = None
    mae_confidence: float | None = None
    objects: list[str] | None = None
    all_classes: list[str] | None = None


class StreamReadDto(msgspec.Struct, frozen=True):
    """
    DTO для чтения текущего состояния потоковой обработки.
    Отражает реальную структуру KlinStreamState.
    """

    id: uuid.UUID
    camera_id: str
    camera_url: str | None
    state: ProcessingState

    # Последние известные результаты
    last_x3d_label: str | None = None
    last_x3d_confidence: float | None = None

    last_mae_label: str | None = None
    last_mae_confidence: float | None = None

    objects: list[str] | None = None  # последние обнаруженные объекты (YOLO)
    all_classes: list[str] | None = None  # если нужно хранить все классы

    @classmethod
    def from_stream_state(cls, model: KlinStreamState) -> "StreamReadDto":
        """
        Создаёт StreamReadDto из KlinStreamState.
        """
        return StreamReadDto(
            id=model.id,
            camera_id=model.camera_id,
            camera_url=model.camera_url,
            state=model.state,
            last_x3d_label=model.last_x3d_label,
            last_x3d_confidence=model.last_x3d_confidence,
            last_mae_label=model.last_mae_label,
            last_mae_confidence=model.last_mae_confidence,
            objects=model.objects,
            all_classes=model.all_classes,
        )


@dataclass
class StreamEventDto:
    """
    Внутреннее событие, которое процессор отправляет в consumer.
    """

    id: str
    stream_id: uuid.UUID
    camera_id: str
    type: str  # "YOLO", "MAE", "X3D_VIOLENCE"
    payload: dict  # содержимое зависит от типа события
