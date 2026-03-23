"""
Shared ORM base for per-stage async process tracking.
"""

import uuid

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel
from app.models.klin import ProcessingState


class BaseAsyncProcessModel(BaseModel):
    """Common fields for all async model-stage processes."""

    __abstract__ = True

    task_id: Mapped[uuid.UUID] = mapped_column(
        unique=True,
        default=uuid.uuid4,
        index=True,
    )
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    klin_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    stream_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    state: Mapped[ProcessingState] = mapped_column(
        String(),
        default=ProcessingState.PENDING,
    )
    input_ref: Mapped[str | None] = mapped_column(String(), nullable=True)
    output_ref: Mapped[str | None] = mapped_column(String(), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(), nullable=True)
