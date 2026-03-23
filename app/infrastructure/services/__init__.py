"""
Бочка для передачи методов
"""

from .callback_sender import KlinCallbackSender
from .s3_storage import S3ObjectStorage
from .target import (
    InferenceProcessor,
    ProcessorConfig,
    StreamProcessor,
)


__all__ = [
    "InferenceProcessor",
    "KlinCallbackSender",
    "ProcessorConfig",
    "S3ObjectStorage",
    "StreamProcessor",
]
