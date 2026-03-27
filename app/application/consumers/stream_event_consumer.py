"""
Literally all the consumers for application
"""

import asyncio
import logging

from app.application.dto import StreamEventDto
from app.application.interfaces import IStreamEventRepository
from app.models import KlinMaeResult, KlinX3DResult, KlinYoloResult


logger = logging.getLogger(__name__)


class StreamEventConsumer:
    """
    Persists stream events in the repository layer.
    """

    def __init__(self, repository: IStreamEventRepository) -> None:
        self.repository = repository

    async def handle(self, event: StreamEventDto) -> None:
        """
        Persists one stream event according to its stage type.
        """

        handlers = {
            "YOLO": self._handle_yolo,
            "MAE": self._handle_mae,
            "X3D_VIOLENCE": self._handle_x3d,
        }
        handler = handlers.get(event.type)
        if handler is None:
            raise ValueError(f"Unsupported event type: {event.type}")

        try:
            retries = 3
            for attempt in range(retries):
                try:
                    await handler(event)
                    return
                except Exception as e:
                    logger.exception(
                        "Retry %d failed: event_id=%s stream_id=%s error=%s",
                        attempt + 1,
                        event.id,
                        event.stream_id,
                        str(e),
                    )
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))

        except Exception:
            logger.exception("Failed to save %s event", event.type)
            raise

    async def _handle_yolo(self, event: StreamEventDto) -> None:
        """Convert StreamEventDto to KlinYoloResult and save."""
        data = event.payload

        yolo_result = KlinYoloResult(
            event_id=event.id,
            stream_id=event.stream_id,
            camera_id=event.camera_id,
            frame_idx=data.get("frame_idx"),
            ts=data.get("timestamp"),
            detections=data.get("detections"),
        )
        await self.repository.save_yolo(yolo_result)

    async def _handle_mae(self, event: StreamEventDto) -> None:
        """Convert StreamEventDto to KlinMaeResult and save."""
        data = event.payload

        mae_result = KlinMaeResult(
            event_id=event.id,
            stream_id=event.stream_id,
            camera_id=event.camera_id,
            label=data["label"],
            confidence=data["confidence"],
            probs=data.get("probs"),
            start_ts=data["start_ts"],
            end_ts=data["end_ts"],
        )
        await self.repository.save_mae(mae_result)

    async def _handle_x3d(self, event: StreamEventDto) -> None:
        """Convert StreamEventDto to KlinX3DResult and save."""
        data = event.payload

        # Защита от разных форматов timestamp
        ts_raw = data.get("timestamp")
        if ts_raw is None:
            logger.error("Event %s missing timestamp payload", event.id)
            return
        ts = float(ts_raw)

        x3d_result = KlinX3DResult(
            event_id=event.id,
            stream_id=event.stream_id,
            camera_id=event.camera_id,
            label=data.get("label", "violence"),
            confidence=data.get("prob"),
            ts=ts,
        )
        await self.repository.save_x3d(x3d_result)

    async def handle_many(self, events: list[StreamEventDto]) -> None:
        """Persists a sequence of stream events one by one."""

        for event in events:
            await self.handle(event)
