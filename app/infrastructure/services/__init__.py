"""
Бочка для передачи методов
"""

from .target import (
    InferenceProcessor,
    KlinCallbackSender,
    ProcessorConfig,
    StreamProcessor,
)


# pylint: disable=duplicate-code
__all__ = [
    "InferenceProcessor",
    "KlinCallbackSender",
    "ProcessorConfig",
    "StreamProcessor",
]
