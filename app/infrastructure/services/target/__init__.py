"""Lazy exports for worker-side processor modules."""

from typing import TYPE_CHECKING, Any

from app.infrastructure.services._lazy_loader import load_lazy_export


if TYPE_CHECKING:
    from .business_processor import BusinessProcessor
    from .mae_processor import MaeProcessor
    from .stream_processor import StreamProcessor
    from .video_processor import InferenceProcessor, ProcessorConfig
    from .x3d_processor import X3DProcessor
    from .yolo_processor import YoloProcessor


_LAZY_EXPORTS = {
    "BusinessProcessor": "app.infrastructure.services.target.business_processor",
    "InferenceProcessor": "app.infrastructure.services.target.video_processor",
    "MaeProcessor": "app.infrastructure.services.target.mae_processor",
    "ProcessorConfig": "app.infrastructure.services.target.video_processor",
    "StreamProcessor": "app.infrastructure.services.target.stream_processor",
    "X3DProcessor": "app.infrastructure.services.target.x3d_processor",
    "YoloProcessor": "app.infrastructure.services.target.yolo_processor",
}

__all__ = [
    "BusinessProcessor",
    "InferenceProcessor",
    "MaeProcessor",
    "ProcessorConfig",
    "StreamProcessor",
    "X3DProcessor",
    "YoloProcessor",
]


def __getattr__(name: str) -> Any:
    """Load one processor module only when the symbol is requested."""

    return load_lazy_export(
        package_name=__name__,
        export_name=name,
        exports=_LAZY_EXPORTS,
    )
