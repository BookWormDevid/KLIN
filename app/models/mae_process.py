"""
Async VideoMAE process model.
"""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.process_base import BaseAsyncProcessModel


class MaeProcessModel(BaseAsyncProcessModel):
    __tablename__ = "mae_process"

    predicted_class: Mapped[str | None] = mapped_column(String(), nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
