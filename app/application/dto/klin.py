# pylint: disable=too-few-public-methods
"""DTO contracts used by the application layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import msgspec

from app.models import ProcessingState


class KlinUploadDto(msgspec.Struct, frozen=True):
    """Input DTO for uploaded offline videos."""

    video_path: str
    response_url: str | None = None


class KlinResultDto(msgspec.Struct, frozen=True):
    """Inference output returned by the offline processor."""

    x3d: str | None
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None


class KlinReadDto(msgspec.Struct, frozen=True):
    """Read DTO returned for one offline klin task."""

    id: uuid.UUID
    x3d: str | None
    mae: str | None
    yolo: str | None
    objects: list[str] | None
    all_classes: list[str] | None
    state: ProcessingState


class KlinProcessDto(msgspec.Struct, frozen=True):
    klin_id: uuid.UUID


class StreamProcessDto(msgspec.Struct, frozen=True):
    stream_id: uuid.UUID


class StreamUploadDto(msgspec.Struct, frozen=True):
    camera_url: str
    camera_id: str


class StreamReadDto(msgspec.Struct, frozen=True):
    """Read DTO returned for one stream state."""

    id: uuid.UUID
    camera_id: str
    camera_url: str | None
    state: ProcessingState
    last_x3d_label: str | None = None
    last_x3d_confidence: float | None = None
    last_mae_label: str | None = None
    last_mae_confidence: float | None = None
    objects: list[str] | None = None
    all_classes: list[str] | None = None


@dataclass
class StreamEventDto:
    """Internal stream event emitted by the processor."""

    id: str
    stream_id: uuid.UUID
    camera_id: str
    type: str
    payload: dict
