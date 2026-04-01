"""
Shared configuration and runtime assembly for target processors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]

from app.config import app_settings
from app.infrastructure.helpers import (
    MaeConfig,
    PrepareForTriton,
    StreamConfig,
    YoloConfig,
)

from .business_processor import BusinessProcessor
from .mae_processor import MaeProcessor
from .x3d_processor import X3DProcessor
from .yolo_processor import YoloProcessor


@dataclass
class ProcessorConfig:
    """Shared configuration for offline inference processors."""

    stream: StreamConfig = field(default_factory=StreamConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    mae: MaeConfig = field(default_factory=MaeConfig)

    @property
    def chunk_size(self) -> int:
        return self.stream.chunk_size

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.stream.frame_size

    @property
    def yolo_stride(self) -> int:
        return self.yolo.yolo_stride

    @property
    def yolo_batch_size(self) -> int:
        return self.yolo.yolo_batch_size

    @property
    def yolo_conf(self) -> float:
        return self.yolo.yolo_conf

    @property
    def yolo_classes(self) -> dict[int, str]:
        return self.yolo.yolo_classes

    @property
    def allowed_classes(self) -> set[int]:
        return self.yolo.allowed_classes

    @property
    def mae_classes(self) -> dict[int, str]:
        return self.mae.mae_classes


@dataclass
class StreamProcessorConfig(ProcessorConfig):
    """Configuration for streaming inference."""

    x3d_conf: float = 0.68


@dataclass
class ProcessorRuntime:
    """
    Initialized Triton client and all stage processors.
    """

    prepare: PrepareForTriton
    triton: grpcclient.InferenceServerClient
    x3d_processor: X3DProcessor
    mae_processor: MaeProcessor
    yolo_processor: YoloProcessor
    business_processor: BusinessProcessor


def build_processor_runtime(config: ProcessorConfig) -> ProcessorRuntime:
    """
    Build a fully initialized processor runtime for the given config.
    """
    prepare = PrepareForTriton()
    triton = grpcclient.InferenceServerClient(url=app_settings.triton_url)
    return ProcessorRuntime(
        prepare=prepare,
        triton=triton,
        x3d_processor=X3DProcessor(triton, prepare),
        mae_processor=MaeProcessor(triton, prepare),
        yolo_processor=YoloProcessor(triton),
        business_processor=BusinessProcessor(
            mae_classes=config.mae_classes,
            yolo_classes=config.yolo_classes,
            allowed_classes=config.allowed_classes,
            yolo_conf=config.yolo_conf,
        ),
    )
