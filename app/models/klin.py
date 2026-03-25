"""
Описание состояний строк и свойств колонок в бд
"""

# pylint: disable= too-few-public-methods
import enum
from uuid import UUID

from sqlalchemy import ARRAY, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class ProcessingState(str, enum.Enum):
    """
    Описание состояния строки для процессора
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class KlinModel(BaseModel):
    """
    response_url - ссылка для отправки вывода
    video_path - ссылка на созданное видео
    state - состояние строки
    x3d - вывод x3d
    mae - вывод videomae
    yolo - вывод yolo
    all_classes - вывод всех классов, что нашёл videomae
    objects - вывод yolo bounding boxes на кадрах
    """

    __tablename__ = "klin"

    response_url: Mapped[str | None] = mapped_column(nullable=True)
    video_path: Mapped[str] = mapped_column(String(), nullable=False)
    state: Mapped[ProcessingState] = mapped_column(String())
    x3d: Mapped[str | None] = mapped_column(String(), nullable=True)
    mae: Mapped[str | None] = mapped_column(String(), nullable=True)
    yolo: Mapped[str | None] = mapped_column(String(), nullable=True)
    all_classes: Mapped[list[str] | None] = mapped_column(
        ARRAY(String()),
        nullable=True,
    )
    objects: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)


class KlinStreamState(BaseModel):
    __tablename__ = "klin_stream_state"

    id: Mapped[UUID]  # PK из BaseModel

    camera_id: Mapped[str] = mapped_column(String(), unique=True, index=True)
    camera_url: Mapped[str | None] = mapped_column(nullable=True)

    state: Mapped[ProcessingState] = mapped_column(String(), index=True)

    # last known values (для UI)
    last_x3d_label: Mapped[str | None] = mapped_column(String(), nullable=True)
    last_x3d_confidence: Mapped[float | None] = mapped_column(nullable=True)

    last_mae_label: Mapped[str | None] = mapped_column(String(), nullable=True)
    last_mae_confidence: Mapped[float | None] = mapped_column(nullable=True)

    # агрегаты (ОПЦИОНАЛЬНО)
    objects: Mapped[list[str] | None] = mapped_column(ARRAY(String()), nullable=True)
    all_classes: Mapped[list[str] | None] = mapped_column(
        ARRAY(String()), nullable=True
    )


class KlinX3DResult(BaseModel):
    __tablename__ = "klin_x3d_result"

    id: Mapped[UUID]

    stream_id: Mapped[UUID] = mapped_column(
        ForeignKey("klin_stream_state.id"), index=True
    )

    camera_id: Mapped[str] = mapped_column(index=True)

    event_id: Mapped[str] = mapped_column(String(), unique=True, index=True)

    label: Mapped[str] = mapped_column(String())
    confidence: Mapped[float] = mapped_column()

    ts: Mapped[float] = mapped_column()


class KlinMaeResult(BaseModel):
    __tablename__ = "klin_mae_result"

    id: Mapped[UUID]

    stream_id: Mapped[UUID] = mapped_column(
        ForeignKey("klin_stream_state.id"), index=True
    )

    camera_id: Mapped[str] = mapped_column(index=True)

    event_id: Mapped[str] = mapped_column(String(), unique=True, index=True)

    label: Mapped[str] = mapped_column(String())
    confidence: Mapped[float] = mapped_column()

    start_ts: Mapped[float] = mapped_column()
    end_ts: Mapped[float] = mapped_column()

    probs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class KlinYoloResult(BaseModel):
    __tablename__ = "klin_yolo_result"

    id: Mapped[UUID]

    stream_id: Mapped[UUID] = mapped_column(
        ForeignKey("klin_stream_state.id"), index=True
    )

    camera_id: Mapped[str] = mapped_column(index=True)

    event_id: Mapped[str] = mapped_column(String(), unique=True, index=True)

    frame_idx: Mapped[int | None] = mapped_column(nullable=True)
    ts: Mapped[float] = mapped_column()

    detections: Mapped[list[dict]] = mapped_column(JSONB)
