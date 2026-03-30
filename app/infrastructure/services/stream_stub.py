"""Lightweight API-side streaming stub."""

from app.application.interfaces import IKlinStream
from app.models import KlinStreamState


class ApiStreamStub(IKlinStream):
    """Protect the API process from starting local stream processing."""

    def unavailable_reason(self, stream: KlinStreamState | None = None) -> str:
        """Return a clear message about where stream processing should run."""

        stream_suffix = "" if stream is None else f" for stream_id={stream.id}"
        return (
            "Stream processor is not available in the API container. "
            f"Use the queue worker to process streams{stream_suffix}."
        )

    async def streaming_analyze(self, stream: KlinStreamState) -> None:
        """Fail fast if the API process is asked to run streaming inference."""

        raise RuntimeError(self.unavailable_reason(stream))

    async def stop(self, camera_id: str) -> None:
        """Fail fast if stop is routed to the API container by mistake."""

        raise RuntimeError(self.unavailable_reason())

    async def wait_stopped(self, camera_id: str, timeout: float = 5) -> bool:
        """Fail fast if wait is routed to the API container by mistake."""

        raise RuntimeError(self.unavailable_reason())
