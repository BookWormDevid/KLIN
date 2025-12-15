import enum

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import BaseModel


class ProcessingState(str, enum.Enum):
    PENDING = "PENDING"
    FINISHED = "FINISHED"


class KlinModel(BaseModel):
    __tablename__ = "Klin"

    target_url: Mapped[str]
    state: Mapped[ProcessingState] = mapped_column(String())
    result: Mapped[str | None] = mapped_column(default=None)