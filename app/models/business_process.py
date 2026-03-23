"""
Async business orchestration process model.
"""

import uuid

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.process_base import BaseAsyncProcessModel


class BusinessProcessModel(BaseAsyncProcessModel):
    """
    Состояние асинхронной бизнес-оркестрации пайплайна.
    """

    __tablename__ = "business_process"

    decision: Mapped[str | None] = mapped_column(String(), nullable=True)
    x3d_task_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    mae_task_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    yolo_task_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
