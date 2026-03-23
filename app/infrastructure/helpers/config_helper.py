"""
Класс с конфигами для процессора
"""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    """
    Настройки потоковой обработки.
    """

    chunk_size: int = 16
    frame_size: tuple[int, int] = (224, 224)


@dataclass
class YoloConfig:
    """
    Настройки инференса YOLO.
    """

    yolo_stride: int = 2
    yolo_batch_size: int = 32
    yolo_conf: float = 0.6
    yolo_classes: dict[int, str] = field(default_factory=lambda: {0: "person"})
    allowed_classes: set[int] = field(default_factory=lambda: {0})


@dataclass
class MaeConfig:
    """
    Настройки классификации VideoMAE.
    """

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
    """
    Временный контекст обработки одного видео
    """

    pipeline: "PipelineQueues"
    state: VideoStreamState
    stats: "VideoProcessingStats"


@dataclass
class PipelineQueues:
    """
    Очереди и фоновые задачи пайплайна обработки одного видео.
    """

    yolo_queue: asyncio.Queue
    mae_queue: asyncio.Queue
    yolo_task: asyncio.Task
    mae_task: asyncio.Task


@dataclass
class VideoProcessingStats:
    """Счетчики и временные метаданные текущей обработки видео."""

    total_frames: int
    fps: float
    duration: float
    frame_idx: int = 0


@dataclass
class Queue:
    """
    Очереди и примитивы конкурентности для потокового инференса.
    """

    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=4)
    )
    infer_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(4)
    )
    yolo_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=80))
    mae_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=80))
    source_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=80)
    )
    x3d_window: list[np.ndarray] = field(default_factory=list)


@dataclass
class HeavyLogic:
    """
    Состояние активации тяжелых стадий потокового анализа.
    """

    heavy_active: asyncio.Event = field(default_factory=asyncio.Event)
    last_trigger_time: float = 0.0
    heavy_cooldown: float = 10.0

    def activate(self, now: float) -> None:
        """
        Включает тяжелый режим и сохраняет время триггера.
        """
        self.heavy_active.set()
        self.last_trigger_time = now

    def should_disable(self, now: float) -> bool:
        """
        Проверяет, истек ли cooldown для тяжелого режима.
        """
        return (
            self.heavy_active.is_set()
            and now - self.last_trigger_time > self.heavy_cooldown
        )
