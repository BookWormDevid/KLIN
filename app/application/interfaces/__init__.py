"""
Бочка для передачи методов
"""

from .klin import (
    IKlinCallbackSender,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
    IKlinStream,
)


__all__ = (
    "IKlinRepository",
    "IKlinInference",
    "IKlinProcessProducer",
    "IKlinCallbackSender",
    "IKlinStream",
)
