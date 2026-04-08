# pylint: disable=too-few-public-methods
"""Application ports used by services, consumers, and adapters."""

from __future__ import annotations

import uuid
from typing import BinaryIO, Protocol, overload

from app.application.dto import (
    KlinProcessDto,
    KlinResultDto,
    StreamEventDto,
    StreamProcessDto,
)
from app.models import (
    KlinMaeResult,
    KlinModel,
    KlinStreamState,
    KlinX3DResult,
    KlinYoloResult,
)


class IKlinEventProducer(Protocol):
    """Publishes streaming events to the broker."""

    async def send_event(self, event: StreamEventDto) -> None:
        """Send one stream event."""


class IKlinStream(Protocol):
    """Runs and stops streaming inference."""

    async def streaming_analyze(self, stream: KlinStreamState) -> None:
        """Start the streaming inference loop for one stream."""

    async def stop(self, camera_id: str) -> None:
        """Request a graceful stop for the active camera."""

    async def wait_stopped(self, camera_id: str, timeout: float = 5) -> bool:
        """Wait until the stream has fully stopped."""


class IKlinStreamEventConsumer(Protocol):
    """Consumes stream events emitted by the processor."""

    async def handle(self, event: StreamEventDto) -> None:
        """Handle one event."""


class IKlinStreamEventService(Protocol):
    async def process(self, event: StreamEventDto) -> None:
        """Process one stream event."""


class IKlinInference(Protocol):
    """Offline inference port."""

    async def analyze(self, model: KlinModel) -> KlinResultDto:
        """Analyze one offline task."""


class IKlinVideoStorage(Protocol):
    """Object storage used for uploaded videos."""

    async def upload_fileobj(
        self,
        *,
        fileobj: BinaryIO,
        object_key: str,
        content_type: str | None = None,
        max_size_bytes: int | None = None,
    ) -> str:
        """Upload content and return its storage URI."""

    async def download_to_path(self, *, source_uri: str, destination_path: str) -> None:
        """Download an object to a local path."""

    async def delete(self, source_uri: str) -> None:
        """Delete an object by URI."""

    async def list_objects(self, prefix: str) -> list[str]: ...


class IKlinRuntimeSettings(Protocol):
    """Small settings surface used by application services."""

    @property
    def max_retry_attempts(self) -> int:
        """Number of enqueue retries for offline processing."""


class IKlinTaskRepository(Protocol):
    """Persistence port for offline klin tasks."""

    async def get_by_id(self, klin_id: uuid.UUID) -> KlinModel:
        """Load one offline task by id."""

    async def claim_for_processing(self, klin_id: uuid.UUID) -> KlinModel | None:
        """Atomically move a pending task into processing state."""

    async def create(self, model: KlinModel) -> KlinModel:
        """Persist a new offline task."""

    async def update(self, model: KlinModel) -> None:
        """Persist an updated offline task."""

    async def get_first_n(self, count: int) -> list[KlinModel]:
        """Return the latest offline tasks."""


class IStreamStateRepository(Protocol):
    """Persistence port for stream lifecycle state."""

    async def get_by_id_camera(self, camera_id: str) -> KlinStreamState | None:
        """Load one stream by camera id."""

    async def get_by_id_stream(self, stream_id: uuid.UUID) -> KlinStreamState | None:
        """Load one stream state by id, if it exists."""

    async def claim_for_processing_stream(
        self, stream_id: uuid.UUID
    ) -> KlinStreamState | None:
        """Atomically move a pending stream into processing state."""

    async def create(self, model: KlinStreamState) -> KlinStreamState:
        """Persist a new stream state."""

    async def update(self, model: KlinStreamState) -> None:
        """Persist an updated stream state."""


class IStreamEventRepository(Protocol):
    """Persistence port for streaming stage results."""

    async def save_yolo(self, event: KlinYoloResult) -> None: ...

    async def save_mae(self, event: KlinMaeResult) -> None: ...

    async def save_x3d(self, event: KlinX3DResult) -> None: ...


class IKlinRepository(Protocol):
    """Backward-compatible aggregate repository port."""

    async def save_yolo(self, event: KlinYoloResult) -> None: ...

    async def save_mae(self, event: KlinMaeResult) -> None: ...

    async def save_x3d(self, event: KlinX3DResult) -> None: ...

    async def get_by_id(self, klin_id: uuid.UUID) -> KlinModel: ...

    async def get_by_id_stream(
        self, stream_id: uuid.UUID
    ) -> KlinStreamState | None: ...

    async def claim_for_processing(self, klin_id: uuid.UUID) -> KlinModel | None: ...

    async def claim_for_processing_stream(
        self, stream_id: uuid.UUID
    ) -> KlinStreamState | None: ...

    @overload
    async def create(self, model: KlinModel) -> KlinModel: ...

    @overload
    async def create(self, model: KlinStreamState) -> KlinStreamState: ...

    async def create(
        self, model: KlinModel | KlinStreamState
    ) -> KlinModel | KlinStreamState: ...

    @overload
    async def update(self, model: KlinModel) -> None: ...

    @overload
    async def update(self, model: KlinStreamState) -> None: ...

    async def update(self, model: KlinModel | KlinStreamState) -> None: ...

    async def get_first_n(self, count: int) -> list[KlinModel]: ...


class IKlinProcessProducer(Protocol):
    """Publishes offline and streaming jobs to the broker."""

    async def send(self, data: KlinProcessDto) -> None:
        """Publish one offline processing job."""

    async def send_stream(self, data: StreamProcessDto) -> None:
        """Publish one stream processing job."""


class IKlinCallbackSender(Protocol):
    """Sends final offline processing callbacks."""

    async def post_consumer(self, model: KlinModel) -> None:
        """Send one callback payload."""
