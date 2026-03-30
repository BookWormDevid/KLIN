"""Lightweight exports for infrastructure services.

The API process imports submodules from this package during startup.
Keep this module free from eager worker-only imports such as OpenCV/Triton
adapters, otherwise the API container starts requiring worker dependencies.
"""

from typing import TYPE_CHECKING, Any

from ._lazy_loader import load_lazy_export
from .callback_sender import KlinCallbackSender
from .s3_storage import S3ObjectStorage


if TYPE_CHECKING:
    from app.infrastructure.services.target.stream_processor import StreamProcessor
    from app.infrastructure.services.target.video_processor import (
        InferenceProcessor,
        ProcessorConfig,
    )


_LAZY_EXPORTS = {
    "InferenceProcessor": "app.infrastructure.services.target.video_processor",
    "ProcessorConfig": "app.infrastructure.services.target.video_processor",
    "StreamProcessor": "app.infrastructure.services.target.stream_processor",
}

__all__ = [
    "InferenceProcessor",
    "KlinCallbackSender",
    "ProcessorConfig",
    "S3ObjectStorage",
    "StreamProcessor",
]


def __getattr__(name: str) -> Any:
    """Load worker-only adapters on demand."""

    return load_lazy_export(
        package_name=__name__,
        export_name=name,
        exports=_LAZY_EXPORTS,
    )
