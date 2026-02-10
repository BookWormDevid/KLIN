import enum

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class ProcessingState(str, enum.Enum):
    PENDING = "PENDING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class MAEModel(BaseModel):
    __tablename__ = "MAE"

    response_url: Mapped[str] = mapped_column(nullable=True)
    video_path: Mapped[str] = mapped_column(String(), nullable=False)
    state: Mapped[ProcessingState] = mapped_column(String())
    result: Mapped[str | None] = mapped_column(Text, default=None)
