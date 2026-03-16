"""
Бочка для передачи методов
"""

from .stream_processor import StreamProcessor
from .video_processor import (
    InferenceProcessor,
    KlinCallbackSender,
    ProcessorConfig,
)


__all__ = [
    "InferenceProcessor",
    "KlinCallbackSender",
    "ProcessorConfig",
    "StreamProcessor",
]
