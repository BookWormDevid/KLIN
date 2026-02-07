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

    target_url: Mapped[str] = mapped_column(Text)
    response_url: Mapped[str] = mapped_column(nullable=True)
    state: Mapped[ProcessingState] = mapped_column(String())
    result: Mapped[str | None] = mapped_column(Text, default=None)
