"""
Бочка для передачи методов
"""

from .target import (
    InferenceProcessor,
    KlinCallbackSender,
    ProcessorConfig,
    VideoMAEConfig,
    YoloConfig,
)


__all__ = [
    "InferenceProcessor",
    "KlinCallbackSender",
    "VideoMAEConfig",
    "YoloConfig",
    "ProcessorConfig",
]
