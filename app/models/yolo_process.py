"""
Async YOLO process model.
"""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.process_base import BaseAsyncProcessModel


class YoloProcessModel(BaseAsyncProcessModel):
    """
    Состояние асинхронной стадии YOLO.
    """

    __tablename__ = "yolo_process"

    detected_objects: Mapped[str | None] = mapped_column(String(), nullable=True)
    detections_ref: Mapped[str | None] = mapped_column(String(), nullable=True)
