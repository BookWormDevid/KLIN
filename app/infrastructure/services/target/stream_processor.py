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
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.application.dto import StreamEventDto
from app.application.interfaces import IKlinEventProducer, IKlinStream
from app.infrastructure.helpers import (
    HeavyLogic,
    Queue,
)
from app.models.klin import (
    KlinStreamState,
)

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
    stopped_event: asyncio.Event


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
            queue=Queue(),
            heavy=HeavyLogic(),
            stop_event=asyncio.Event(),
            stopped_event=asyncio.Event(),
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
        for attempt in range(3):
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
                return
            except Exception:
                logger.exception("Emit retry %d", attempt + 1)
                await asyncio.sleep(0.2 * (attempt + 1))

        logger.error("Event dropped after retries")

    async def _infer_yolo_batch_with_retry(
        self,
        context: StreamContext,
        batch_imgs: np.ndarray,
    ) -> list[NDArray[np.float32]] | None:
        """Run YOLO inference with a short retry loop for transient failures."""

        max_batch_retries = 2

        for attempt in range(max_batch_retries + 1):
            preds_list = cast(
                list[NDArray[np.float32]] | None,
                await self._triton_infer_with_retry(
                    self.runtime.yolo_processor.infer_batch,
                    batch_imgs,
                    context=context,
                ),
            )
            if preds_list is not None:
                return preds_list

            logger.warning("YOLO batch retry %d/%d", attempt + 1, max_batch_retries + 1)
            await asyncio.sleep(0.1 * (attempt + 1))

        return None

    async def _emit_yolo_batch(
        self,
        batch: list[tuple[int, np.ndarray, float]],
        preds_list: list[NDArray[np.float32]],
        camera_id: str,
        stream_id: uuid.UUID,
    ) -> None:
        """Publish a YOLO event for the processed batch when detections exist."""

        detections = self._collect_yolo_detections(batch, preds_list)
        if not detections:
            return

        await self._emit_event(
            event_type="YOLO",
            camera_id=camera_id,
            stream_id=stream_id,
            payload={
                "frame_idx": batch[-1][0],
                "timestamp": batch[-1][2],
                "detections": detections,
            },
        )

    async def _yolo_stream_pipeline(
        self, context: StreamContext, camera_id: str, stream_id: uuid.UUID
    ) -> None:
        """Buffer frames for YOLO and flush them in bounded batches."""

        buffer: list[tuple[int, np.ndarray, float]] = []
        last_send = -999.0

        while not context.stop_event.is_set():
            try:
                frame, idx, ts = await asyncio.wait_for(
                    context.queue.yolo_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            if not context.heavy.heavy_active.is_set():
                context.queue.yolo_queue.task_done()
                continue

            try:
                if idx % self.processing.yolo_stride == 0:
                    buffer.append((idx, frame, ts))

                if len(buffer) > self.processing.yolo_batch_size * 3:
                    logger.warning("YOLO buffer overflow → dropping oldest frames")
                    buffer = buffer[-self.processing.yolo_batch_size :]

                if self._should_flush_yolo_buffer(buffer, ts, last_send):
                    batch_size = min(len(buffer), self.processing.yolo_batch_size)
                    batch = buffer[:batch_size]
                    buffer = buffer[batch_size:]
                    batch_imgs = self._build_stream_yolo_batch(batch)

                    preds_list = await self._infer_yolo_batch_with_retry(
                        context, batch_imgs
                    )
                    if preds_list is None:
                        logger.error("YOLO batch dropped after retries")
                        last_send = ts
                    else:
                        await self._emit_yolo_batch(
                            batch, preds_list, camera_id, stream_id
                        )
                        last_send = ts
            finally:
                context.queue.yolo_queue.task_done()

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
                    payload={"prob": violence_prob, "timestamp": now},
                )
                context.heavy.activate(now)  # X3D True → heavy ON

            elif context.heavy.should_disable(now):
                logger.info("X3D stable normal → heavy OFF")
                context.heavy.heavy_active.clear()

                yolo_q = context.queue.yolo_queue
                cleared = 0
                while not yolo_q.empty():
                    try:
                        yolo_q.get_nowait()
                        yolo_q.task_done()
                        cleared += 1
                    except asyncio.QueueEmpty:
                        break
                if cleared:
                    logger.debug(
                        "Cleared %d frames from yolo_queue on heavy disable", cleared
                    )

    async def broadcast_frames(self, context: StreamContext) -> None:
        """Задача-бродкастер: раздаёт каждый кадр во все нужные очереди"""
        while not context.stop_event.is_set():
            try:
                frame_tuple = await asyncio.wait_for(
                    context.queue.source_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            # MAE всегда получает кадры (нужно для x3d_window)
            try:
                context.queue.mae_queue.put_nowait(frame_tuple)
            except asyncio.QueueFull:
                logger.debug("mae_queue full — кадр пропущен")

            # YOLO получает кадры ТОЛЬКО когда heavy-режим активен
            if context.heavy.heavy_active.is_set():
                try:
                    context.queue.yolo_queue.put_nowait(frame_tuple)
                except asyncio.QueueFull:
                    logger.debug("yolo_queue full — кадр пропущен")

            context.queue.source_queue.task_done()

    async def stop(self, camera_id: str) -> None:
        """Stops an active stream by camera id."""
        context = self.contexts.get(camera_id)
        if not context:
            logger.warning("Stop requested but context not found: %s", camera_id)
            return

        logger.info("Stopping stream processor for camera_id=%s", camera_id)
        context.stop_event.set()

    async def wait_stopped(self, camera_id: str, timeout: float = 5) -> bool:
        """Wait until the stream context reports that shutdown has completed."""

        context = self.contexts.get(camera_id)
        if not context:
            return True  # уже нет → считаем остановленным

        try:
            await asyncio.wait_for(context.stopped_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _watchdog(self, tasks, context):
        """Stop the stream if any worker task exits with an unexpected error."""

        while not context.stop_event.is_set():
            for t in tasks:
                if t is asyncio.current_task():
                    continue

                if t.done():
                    if t.cancelled():
                        continue

                    exc = t.exception()
                    if exc is not None:
                        logger.error("Task crashed: %s", t, exc_info=exc)
                        context.stop_event.set()
                        return

            await asyncio.sleep(1)

    async def streaming_analyze(self, stream: KlinStreamState) -> None:
        """Launches the background tasks for one camera stream.
        Теперь принимает KlinStreamState вместо KlinStreamingModel.
        """
        assert stream.camera_url is not None
        assert stream.camera_id is not None

        context = self._create_context()
        self.contexts[stream.camera_id] = context

        tasks = [
            asyncio.create_task(
                self._frame_reader(
                    stream.camera_url,
                    context.queue.source_queue,
                    context.stop_event,
                )
            ),
            asyncio.create_task(self.broadcast_frames(context)),
            asyncio.create_task(
                self._periodic_x3d_checker(
                    context=context,
                    camera_id=stream.camera_id,
                    stream_id=stream.id,
                )
            ),
            asyncio.create_task(
                self._yolo_stream_pipeline(
                    context=context,
                    camera_id=stream.camera_id,
                    stream_id=stream.id,
                )
            ),
            asyncio.create_task(
                self._mae_stream_pipeline(
                    context=context,
                    camera_id=stream.camera_id,
                    stream_id=stream.id,
                )
            ),
        ]

        watchdog_task = asyncio.create_task(self._watchdog(tasks, context))
        tasks.append(watchdog_task)

        logger.info(
            "🚀 СТРИМИНГ ЗАПУЩЕН (camera_id=%s, stream_id=%s, url=%s)",
            stream.camera_id,
            stream.id,
            stream.camera_url,
        )

        try:
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()

                await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("⛔ Стриминг отменён (stream_id=%s)", stream.id)
            context.stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            logger.exception("Критическая ошибка в стриминге (stream_id=%s)", stream.id)
            context.stop_event.set()
            raise
        finally:
            logger.info("Стриминг завершён (stream_id=%s)", stream.id)
            context.stopped_event.set()
            self.contexts.pop(stream.camera_id, None)
