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
    IKlinStreamEventConsumer,
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
    "IKlinStreamEventConsumer",
)
