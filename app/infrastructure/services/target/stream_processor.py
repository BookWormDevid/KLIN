"""
Процессор для взаимодействия с приложением litestar (РЕАЛ-ТАЙМ СТРИМИНГ БЕЗ CALLBACK)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from app.application.dto import StreamEventDto
from app.application.interfaces import IKlinEventProducer, IKlinRepository, IKlinStream
from app.infrastructure.helpers import (
    HeavyLogic,
    Queue,
)
from app.models.klin import KlinStreamingModel

from .runtime_processor import StreamProcessorConfig, build_processor_runtime


logger = logging.getLogger(__name__)


@dataclass
class StreamContext:
    """
    Runtime context for one active camera stream.
    """

    queue: Queue
    heavy: HeavyLogic
    stop_event: asyncio.Event


class StreamProcessor(IKlinStream):
    """
    Runs the streaming inference pipeline and emits stage events.
    """

    def __init__(self, event_producer: IKlinEventProducer) -> None:
        self.processing = StreamProcessorConfig()
        self.runtime = build_processor_runtime(self.processing)
        self.contexts: dict[str, StreamContext] = {}
        self.event_producer = event_producer

    def _create_context(self) -> StreamContext:
        return StreamContext(
            queue=Queue(), heavy=HeavyLogic(), stop_event=asyncio.Event()
        )

    async def _quick_x3d_check_for_streaming(
        self, frames: list[np.ndarray], context: StreamContext
    ) -> dict[str, float] | None:
        logits = await self._triton_infer_with_retry(
            self.runtime.x3d_processor.infer_clip, frames, context=context
        )
        if logits is None:
            return None

        return self.runtime.business_processor.classify_x3d_logits(logits)

    async def _triton_infer_with_retry(
        self,
        infer_fn: Callable[..., Any],
        *args: Any,
        context: StreamContext,
        timeout: float = 10,
        retries: int = 4,
    ) -> Any:
        loop = asyncio.get_running_loop()

        async with context.queue.infer_semaphore:
            for attempt in range(retries + 1):
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            context.queue.executor, lambda: infer_fn(*args)
                        ),
                        timeout=timeout,
                    )

                except asyncio.TimeoutError:
                    logger.warning(
                        "Triton timeout (attempt %d/%d)", attempt + 1, retries + 1
                    )

                except Exception:
                    logger.exception(
                        "Triton error (attempt %d/%d)", attempt + 1, retries + 1
                    )

                if attempt < retries:
                    await asyncio.sleep(0.1 * (attempt + 1))

        logger.error("Triton failed after retries")
        return None

    async def _frame_reader(
        self,
        camera_url: str,
        frame_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            logger.error("Не удалось открыть поток камеры: %s", camera_url)
            return

        # определяем: файл или поток
        is_file = camera_url.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

        # FPS для "реалистичного" проигрывания файла
        fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        frame_delay = 1.0 / fps if fps and fps > 0 else 0.0

        frame_idx = 0
        reconnect_attempts = 0
        reconnect_delay = 1.0

        while not stop_event.is_set():
            ret, frame = cap.read()

            # ===================== КОНЕЦ ФАЙЛА =====================
            if not ret:
                if is_file:
                    logger.info(
                        "Достигнут конец видео, перемотка в начало: %s", camera_url
                    )
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # ===================== ПОТЕРЯ КАМЕРЫ =====================
                reconnect_attempts += 1
                logger.warning(
                    "Потеря соединения с камерой %s, переподключение #%d...",
                    camera_url,
                    reconnect_attempts,
                )

                cap.release()

                await asyncio.sleep(min(reconnect_delay, 10.0))
                reconnect_delay *= 2

                cap = cv2.VideoCapture(camera_url)

                if cap.isOpened():
                    logger.info("Переподключение к камере %s успешно", camera_url)
                    reconnect_attempts = 0
                    reconnect_delay = 1.0

                continue

            # ===================== УСПЕШНЫЙ КАДР =====================
            reconnect_attempts = 0
            reconnect_delay = 1.0

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, self.processing.frame_size)

            try:
                frame_queue.put_nowait((resized, frame_idx, time.time()))
            except asyncio.QueueFull:
                logger.debug("source_queue full — кадр пропущен")

            frame_idx += 1

            # ===================== REAL-TIME ДЛЯ ФАЙЛА =====================
            if is_file and frame_delay > 0:
                await asyncio.sleep(frame_delay)

        cap.release()

    def _should_flush_yolo_buffer(
        self,
        buffer: list[tuple[int, np.ndarray, float]],
        timestamp: float,
        last_send: float,
    ) -> bool:
        return len(buffer) >= self.processing.yolo_batch_size or (
            bool(buffer) and timestamp - last_send > 0.8
        )

    def _build_stream_yolo_batch(
        self, buffer: list[tuple[int, np.ndarray, float]]
    ) -> np.ndarray:
        return np.stack(
            [
                self.runtime.prepare.prepare_yolo_frame_for_triton(frame)[0]
                for _, frame, _ in buffer
            ],
            axis=0,
        )

    def _build_frame_detections(
        self,
        preds: NDArray[np.float32],
        timestamp: float,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for pred in preds:
            detection = self.runtime.business_processor.parse_yolo_detection(pred)
            if detection is None:
                continue
            class_id, bbox = detection
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.processing.yolo_classes.get(class_id, "unknown"),
                    "bbox": bbox,
                    "confidence": float(pred[4 + class_id]),
                    "timestamp": timestamp,
                }
            )
        return detections

    def _collect_yolo_detections(
        self,
        buffer: list[tuple[int, np.ndarray, float]],
        preds_list: list[NDArray[np.float32]],
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for (_, _, timestamp), preds in zip(buffer, preds_list, strict=True):
            detections.extend(self._build_frame_detections(preds, timestamp))
        return detections

    async def _emit_event(
        self,
        *,
        event_type: str,
        camera_id: str,
        stream_id: uuid.UUID,
        payload: dict[str, Any],
    ) -> None:
        try:
            await self.event_producer.send_event(
                StreamEventDto(
                    id=str(uuid.uuid4()),
                    type=event_type,
                    camera_id=camera_id,
                    stream_id=stream_id,
                    payload=payload,
                )
            )
        except Exception:
            logger.exception("Failed to send event")

    async def _yolo_stream_pipeline(
        self,
        context: StreamContext,
        camera_id: str,
        stream_id: uuid.UUID,
    ) -> None:
        buffer: list[tuple[int, np.ndarray, float]] = []
        last_send = -999.0
        frame_queue = context.queue.yolo_queue
        while not context.stop_event.is_set():
            try:
                frame, idx, ts = await asyncio.wait_for(frame_queue.get(), timeout=0.08)
            except asyncio.TimeoutError:
                continue

            # Ключевое условие: тяжёлый анализ включён?
            if not context.heavy.heavy_active.is_set():
                frame_queue.task_done()
                continue

            if idx % self.processing.yolo_stride == 0:
                buffer.append((idx, frame, ts))

            if self._should_flush_yolo_buffer(buffer, ts, last_send):
                batch_imgs = self._build_stream_yolo_batch(buffer)
                preds_list = await self._triton_infer_with_retry(
                    self.runtime.yolo_processor.infer_batch,
                    batch_imgs,
                    context=context,
                )
                if preds_list is None:
                    logger.warning("YOLO inference failed — dropping batch")
                    buffer.clear()
                    last_send = ts
                    await asyncio.sleep(0.05)
                    continue

                detections = self._collect_yolo_detections(buffer, preds_list)

                if detections:
                    await self._emit_event(
                        event_type="YOLO",
                        camera_id=camera_id,
                        stream_id=stream_id,
                        payload={
                            "frame_idx": idx,
                            "timestamp": ts,
                            "detections": detections,
                        },
                    )
                buffer.clear()
                last_send = ts

            frame_queue.task_done()

    async def _mae_stream_pipeline(
        self,
        context: StreamContext,
        camera_id: str,
        stream_id: uuid.UUID,
    ) -> None:
        window: list[np.ndarray] = []
        window_ts: list[float] = []
        last_inference = -999.0
        frame_queue = context.queue.mae_queue
        x3d_window = context.queue.x3d_window
        while not context.stop_event.is_set():
            try:
                frame, _, ts = await asyncio.wait_for(frame_queue.get(), 0.08)
            except asyncio.TimeoutError:
                continue

            x3d_window.append(frame)
            if len(x3d_window) > 64:
                x3d_window.pop(0)

            # Ключевое условие: тяжёлый анализ включён?
            if not context.heavy.heavy_active.is_set():
                frame_queue.task_done()
                continue

            window.append(frame)
            window_ts.append(ts)

            if len(window) > self.processing.chunk_size:
                window.pop(0)
                window_ts.pop(0)

            if len(window) == self.processing.chunk_size and ts - last_inference > 2.0:
                probs = await self._triton_infer_with_retry(
                    self.runtime.mae_processor.infer_probs,
                    window[:],
                    context=context,
                )
                if probs is None:
                    continue
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                label = self.processing.mae_classes.get(pred_idx, str(pred_idx))
                await self._emit_event(
                    event_type="MAE",
                    camera_id=camera_id,
                    stream_id=stream_id,
                    payload={
                        "label": label,
                        "confidence": confidence,
                        "start_ts": window_ts[0],
                        "end_ts": window_ts[-1],
                    },
                )
                last_inference = ts

            frame_queue.task_done()

    async def _periodic_x3d_checker(
        self,
        stream_id: uuid.UUID,
        context: StreamContext,
        camera_id: str,
        interval: float = 3.0,
    ) -> None:
        """Лёгкий постоянный чекер X3D. При триггере включает тяжёлый анализ."""

        x3d_window = context.queue.x3d_window
        while not context.stop_event.is_set():
            await asyncio.sleep(interval)
            if len(x3d_window) < 16:
                continue

            clip = x3d_window[-16:]
            result = await self._quick_x3d_check_for_streaming(clip, context=context)
            now = time.time()
            violence_prob = result.get("True", 0.0) if result is not None else 0.0

            if violence_prob > self.processing.x3d_conf:
                await self._emit_event(
                    event_type="X3D_VIOLENCE",
                    camera_id=camera_id,
                    stream_id=stream_id,
                    payload={
                        "prob": violence_prob,
                        "timestamp": now,
                    },
                )
                context.heavy.activate(now)

            # Авто-выключение тяжёлого режима
            elif context.heavy.should_disable(now):
                logger.info("X3D stable normal → heavy OFF")
                context.heavy.heavy_active.clear()

    async def broadcast_frames(self, context: StreamContext) -> None:
        """
        Задача-бродкастер: раздаёт каждый кадр во все нужные очереди
        """
        while not context.stop_event.is_set():
            try:
                frame_tuple = await asyncio.wait_for(
                    context.queue.source_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            for q in (context.queue.yolo_queue, context.queue.mae_queue):
                try:
                    q.put_nowait(frame_tuple)
                except asyncio.QueueFull:
                    logger.debug("Очередь переполнена — кадр пропущен")

            context.queue.source_queue.task_done()

    def stop(self, camera_id: str) -> None:
        """Stops an active stream by camera id."""

        context = self.contexts.get(camera_id)

        if not context:
            logger.warning("Stop requested but context not found: %s", camera_id)
            return

        logger.info("Stopping stream processor for camera_id=%s", camera_id)

        context.stop_event.set()

    async def streaming_analyze(self, model: KlinStreamingModel) -> None:
        """Launches the background tasks for one camera stream."""

        assert model.camera_url is not None
        assert model.camera_id is not None

        context = self._create_context()
        self.contexts[model.camera_id] = context
        tasks = [
            asyncio.create_task(
                self._frame_reader(
                    model.camera_url, context.queue.source_queue, context.stop_event
                )
            ),
            asyncio.create_task(self.broadcast_frames(context)),
            asyncio.create_task(
                self._periodic_x3d_checker(
                    context=context,
                    camera_id=model.camera_id,
                    stream_id=model.id,
                )
            ),
            asyncio.create_task(
                self._yolo_stream_pipeline(
                    context=context,
                    camera_id=model.camera_id,
                    stream_id=model.id,
                )
            ),
            asyncio.create_task(
                self._mae_stream_pipeline(
                    context=context,
                    camera_id=model.camera_id,
                    stream_id=model.id,
                )
            ),
        ]

        logger.info(
            "🚀 СТРИМИНГ ЗАПУЩЕН (id=%s, url=%s)", model.camera_id, model.camera_url
        )

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("⛔ Стриминг отменён (id=%s)", model.camera_id)
            context.stop_event.set()
            # ждём завершения
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            logger.exception("Критическая ошибка в стриминге (id=%s)", model.camera_id)
            context.stop_event.set()
            raise
        finally:
            logger.info("Стриминг завершён (id=%s)", model.camera_id)
            self.contexts.pop(model.camera_id, None)


class StreamEventConsumer:
    """
    Persists stream events in the repository layer.
    """

    def __init__(self, repository: IKlinRepository) -> None:
        self.repository = repository

    async def handle(self, event: StreamEventDto) -> None:
        """
        Persists one stream event according to its stage type.
        """

        handlers = {
            "YOLO": self.repository.save_yolo,
            "MAE": self.repository.save_mae,
            "X3D_VIOLENCE": self.repository.save_x3d,
        }
        handler = handlers.get(event.type)
        if handler is None:
            logger.warning("Unknown stream event type: %s", event.type)
            return

        try:
            await handler(event)
        except Exception:
            logger.exception("Failed to save %s", event.type)
            raise

    async def handle_many(self, events: list[StreamEventDto]) -> None:
        """Persists a sequence of stream events one by one."""

        for event in events:
            await self.handle(event)
