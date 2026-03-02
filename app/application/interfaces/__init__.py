"""
Бочка для передачи методов
"""

from .klin import (
    IKlinCallbackSender,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
)


__all__ = (
    "IKlinRepository",
    "IKlinInference",
    "IKlinProcessProducer",
    "IKlinCallbackSender",
)
