"""Mapping helpers between persistence models and read DTOs."""

from __future__ import annotations

from app.application.dto import KlinReadDto, StreamReadDto
from app.models import KlinModel, KlinStreamState


def to_klin_read_dto(model: KlinModel) -> KlinReadDto:
    """Build a read DTO from an offline klin model."""

    return KlinReadDto(
        id=model.id,
        x3d=model.x3d,
        mae=model.mae,
        yolo=model.yolo,
        objects=model.objects,
        all_classes=model.all_classes,
        state=model.state,
    )


def to_stream_read_dto(model: KlinStreamState) -> StreamReadDto:
    """Build a read DTO from a persisted stream state."""

    return StreamReadDto(
        id=model.id,
        camera_id=model.camera_id,
        camera_url=model.camera_url,
        state=model.state,
        last_x3d_label=model.last_x3d_label,
        last_x3d_confidence=model.last_x3d_confidence,
        last_mae_label=model.last_mae_label,
        last_mae_confidence=model.last_mae_confidence,
        objects=model.objects,
        all_classes=model.all_classes,
    )
