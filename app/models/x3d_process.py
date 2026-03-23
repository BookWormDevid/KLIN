"""
Async X3D process model.
"""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.process_base import BaseAsyncProcessModel


class X3DProcessModel(BaseAsyncProcessModel):
    __tablename__ = "x3d_process"

    prediction: Mapped[str | None] = mapped_column(String(), nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
