import asyncio
import logging

from app.application.dto import StreamEventDto
from app.application.interfaces import (
    IKlinStream,
    IStreamEventRepository,
    IStreamStateRepository,
)
from app.models import KlinMaeResult, KlinX3DResult, KlinYoloResult, ProcessingState


logger = logging.getLogger(__name__)


class StreamEventService:
    def __init__(
        self,
        repository_event: IStreamEventRepository,
        repository_id: IStreamStateRepository,
        stream_processor: IKlinStream,
    ) -> None:
        self.repository_event = repository_event
        self.repository_id = repository_id
        self.stream_processor = stream_processor

    async def process(self, event: StreamEventDto) -> None:
        if event.type == "YOLO":
            await self._process_yolo(event)

        elif event.type == "MAE":
            await self._process_mae(event)

        elif event.type == "X3D_VIOLENCE":
            await self._process_x3d(event)

        elif event.type == "STOP_STREAM":
            await self._handle_stop(event)

        elif event.type == "STREAM_STOPPED":
            await self.handle_stream_stopped(event)

        else:
            raise ValueError(f"Unsupported event type: {event.type}")

    async def _process_yolo(self, event: StreamEventDto) -> None:
        data = event.payload or {}

        result = KlinYoloResult(
            event_id=event.id,
            stream_id=event.stream_id,
            camera_id=event.camera_id,
            frame_idx=data.get("frame_idx"),
            ts=data.get("timestamp"),
            detections=data.get("detections") or [],
        )

        await self.repository_event.save_yolo(result)

    async def _process_mae(self, event: StreamEventDto) -> None:
        data = event.payload or {}

        result = KlinMaeResult(
            event_id=event.id,
            stream_id=event.stream_id,
            camera_id=event.camera_id,
            label=data["label"],
            confidence=data["confidence"],
            probs=data.get("probs"),
            start_ts=data["start_ts"],
            end_ts=data["end_ts"],
        )

        await self.repository_event.save_mae(result)

    async def _process_x3d(self, event: StreamEventDto) -> None:
        data = event.payload or {}

        ts_raw = data.get("timestamp")
        if ts_raw is None:
            raise ValueError(f"Event {event.id} missing timestamp")

        result = KlinX3DResult(
            event_id=event.id,
            stream_id=event.stream_id,
            camera_id=event.camera_id,
            label=data.get("label", "violence"),
            confidence=data.get("prob"),
            ts=float(ts_raw),
        )

        await self.repository_event.save_x3d(result)

    async def _handle_stop(self, event: StreamEventDto) -> None:
        try:
            await self.stream_processor.stop(event.camera_id)
            await asyncio.wait_for(
                self.stream_processor.wait_stopped(event.camera_id),
                timeout=10,
            )

        except Exception:
            logger.exception("Failed to stop stream")
            raise

    async def handle_stream_stopped(self, event: StreamEventDto) -> None:
        stream = await self.repository_id.get_by_id_stream(event.stream_id)

        if not stream:
            logger.error("Stream not found for STOPPED event: %s", event.stream_id)
            return

        stream.state = ProcessingState.STOPPED
        await self.repository_id.update(stream)
