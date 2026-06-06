"""
Бочка для передачи методов
"""

from .auth import AuthController
from .klin import KlinController
from .stream import KlinStreamController


__all__ = (
    "AuthController",
    "KlinController",
    "KlinStreamController",
)
