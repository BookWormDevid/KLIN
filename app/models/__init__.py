"""
Бочка для передачи методов
"""

from .base import BaseModel, Model
from .klin import KlinModel, ProcessingState

__all__ = (
    "KlinModel",
    "ProcessingState",
    "BaseModel",
    "Model",
)
