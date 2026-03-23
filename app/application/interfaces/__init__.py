"""
Бочка для передачи методов
"""

from .klin import (
    IKlinCallbackSender,
    IKlinEventProducer,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
    IKlinStream,
    IKlinVideoStorage,
)


__all__ = (
    "IKlinRepository",
    "IKlinInference",
    "IKlinProcessProducer",
    "IKlinCallbackSender",
    "IKlinStream",
    "IKlinEventProducer",
    "IKlinVideoStorage",
)
