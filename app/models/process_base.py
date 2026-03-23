"""
Shared ORM base for per-stage async process tracking.
"""

import uuid

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel
from app.models.klin import ProcessingState


class BaseAsyncProcessModel(BaseModel):
    """
    Common fields for all async model-stage processes.
    """

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

    def mark_finished(self, output_ref: str | None = None) -> None:
        """Mark the process as finished and optionally store an output reference."""

        self.state = ProcessingState.FINISHED
        self.error_message = None
        if output_ref is not None:
            self.output_ref = output_ref

    def mark_failed(self, error_message: str) -> None:
        """Mark the process as failed and store the error details."""

        self.state = ProcessingState.ERROR
        self.error_message = error_message
