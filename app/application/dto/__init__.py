"""
Бочка для передачи методов
"""

from .auth import JWTLoginDto, JWTTokenDto
from .klin import (
    KlinProcessDto,
    KlinReadDto,
    KlinResultDto,
    KlinUploadDto,
    StreamEventDto,
    StreamProcessDto,
    StreamReadDto,
    StreamUploadDto,
)


__all__ = (
    "JWTLoginDto",
    "JWTTokenDto",
    "KlinUploadDto",
    "KlinResultDto",
    "KlinReadDto",
    "KlinProcessDto",
    "StreamUploadDto",
    "StreamReadDto",
    "StreamProcessDto",
    "StreamEventDto",
)
