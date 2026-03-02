"""
Описание состояний строк и свойств колонок в бд
"""

# pylint: disable= too-few-public-methods
import enum

from sqlalchemy import ARRAY, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class ProcessingState(str, enum.Enum):
    """
    Описание состояния строки для процессора
    """

    PENDING = "PENDING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class KlinModel(BaseModel):
    """
    response_url - ссылка для отправки вывода
    video_path - ссылка на созданное видео
    state - состояние строки
    mae - вывод videomae
    yolo - вывод yolo
    all_classes - вывод всех классов, что нашёл videomae
    objects - вывод yolo bounding boxes на кадрах
    """

    __tablename__ = "MAE"

    response_url: Mapped[str | None] = mapped_column(nullable=True)
    video_path: Mapped[str] = mapped_column(String(), nullable=False)
    state: Mapped[ProcessingState] = mapped_column(String())
    mae: Mapped[str | None] = mapped_column(String(), nullable=True)
    yolo: Mapped[str | None] = mapped_column(String(), nullable=True)
    all_classes: Mapped[list[str]] = mapped_column(ARRAY(String()), nullable=True)
    objects: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True)
