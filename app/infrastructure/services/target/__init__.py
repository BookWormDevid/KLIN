"""
Бочка для передачи методов
"""

from .processor import (
    InferenceProcessor,
    KlinCallbackSender,
    ProcessorConfig,
    VideoMAEConfig,
    YoloConfig,
)


__all__ = [
    "InferenceProcessor",
    "KlinCallbackSender",
    "ProcessorConfig",
    "YoloConfig",
    "VideoMAEConfig",
]
