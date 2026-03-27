"""
Бочка для передачи методов
"""

from .klin import (
    IKlinCallbackSender,
    IKlinEventProducer,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
    IKlinRuntimeSettings,
    IKlinStream,
    IKlinStreamEventConsumer,
    IKlinTaskRepository,
    IKlinVideoStorage,
    IStreamEventRepository,
    IStreamStateRepository,
)


__all__ = (
    "IKlinRepository",
    "IKlinTaskRepository",
    "IStreamStateRepository",
    "IStreamEventRepository",
    "IKlinInference",
    "IKlinProcessProducer",
    "IKlinCallbackSender",
    "IKlinStream",
    "IKlinEventProducer",
    "IKlinVideoStorage",
    "IKlinStreamEventConsumer",
    "IKlinRuntimeSettings",
)
