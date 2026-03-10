"""
Бочка для передачи методов
"""

from .processor import (
    InferenceProcessor,
    KlinCallbackSender,
    ProcessorConfig,
)


__all__ = [
    "InferenceProcessor",
    "KlinCallbackSender",
    "ProcessorConfig",
]
