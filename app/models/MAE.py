import enum

from sqlalchemy import String, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class ProcessingState(str, enum.Enum):
    PENDING = "PENDING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class MAEModel(BaseModel):
    __tablename__ = "MAE"

    response_url: Mapped[str | None] = mapped_column(nullable=True)
    video_path: Mapped[str] = mapped_column(String(), nullable=False)
    state: Mapped[ProcessingState] = mapped_column(String())
    mae: Mapped[str | None] = mapped_column(String(), nullable=True)
    yolo: Mapped[str | None] = mapped_column(String(), nullable=True)
    all_classes: Mapped[list[str]] = mapped_column(ARRAY(String()), nullable=True)
    objects: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True)

