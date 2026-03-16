"""
Класс с конфигами для процессора
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VideoStreamState:
    """
    Изменяемое состояние потоковой обработки видео.
    """

    mae_results: list[dict[str, Any]] = field(default_factory=list)
    bbox_by_time: dict[float, list[list[float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    detected_class_ids: set[int] = field(default_factory=set)
    chunk_buffer: list[np.ndarray] = field(default_factory=list)
    yolo_buffer: list[tuple[int, np.ndarray]] = field(default_factory=list)
    chunk_start_frame: int = 0
    frame_idx: int = 0


@dataclass
class StreamConfig:
    """Настройки потоковой обработки."""

    chunk_size: int = 16
    frame_size: tuple[int, int] = (224, 224)


@dataclass
class YoloConfig:
    """Настройки инференса YOLO."""

    yolo_stride: int = 2
    yolo_batch_size: int = 32
    yolo_conf: float = 0.6
    yolo_classes: dict[int, str] = field(default_factory=lambda: {0: "person"})
    allowed_classes: set[int] = field(default_factory=lambda: {0})


@dataclass
class MaeConfig:
    """Настройки классификации VideoMAE."""

    mae_classes: dict[int, str] = field(
        default_factory=lambda: {
            0: "Abuse",
            1: "Arrest",
            2: "Arson",
            3: "Assault",
            4: "Burglary",
            5: "Explosion",
            6: "Fighting",
            7: "Normal",
            8: "RoadAccident",
            9: "Robbery",
            10: "Shooting",
            11: "Shoplifting",
            12: "Stealing",
            13: "Vandalism",
        }
    )


@dataclass
class StreamProcessingContext:
    """Временный контекст обработки одного видео"""

    frame_queue: asyncio.Queue
    yolo_task: asyncio.Task
    mae_task: asyncio.Task
    state: VideoStreamState
    total_frames: int
    fps: float
    duration: float
    frame_idx: int = 0
