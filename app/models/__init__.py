"""
Бочка для передачи методов
"""

from .base import BaseModel, Model
from .klin import KlinModel, KlinStreamingModel, ProcessingState


__all__ = (
    "KlinModel",
    "KlinStreamingModel",
    "ProcessingState",
    "BaseModel",
    "Model",
)
