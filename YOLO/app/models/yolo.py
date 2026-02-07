import enum

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from YOLO.app.models.base import BaseModel


class ProcessingState(str, enum.Enum):
    PENDING = "PENDING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class YoloModel(BaseModel):
    __tablename__ = "yolo"

    camera_data: Mapped[dict] = mapped_column(JSONB(), default={})

    yolo_result: Mapped[dict] = mapped_column(JSONB(), default={})

    state: Mapped[ProcessingState] = mapped_column(String(), default="ERROR")

    result: Mapped[str] = mapped_column(default="Дождитесь завершения проверки.")
