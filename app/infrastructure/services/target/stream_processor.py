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
from datetime import datetime, timezone
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
    heartbeats: dict[str, float]
    tasks: list[asyncio.Task] | None = None


class StreamProcessor(IKlinStream):
    """
    Runs the streaming inference pipeline and emits stage events.
    """

    def __init__(self, event_producer: IKlinEventProducer) -> None:
        self.processing = StreamProcessorConfig()
        self.runtime = build_processor_runtime(self.processing)
        self.contexts: dict[str, StreamContext] = {}
        self.event_producer = event_producer

    def _build_task_factories(self, context: StreamContext, stream: KlinStreamState):
        return [
            lambda: asyncio.create_task(
                self._frame_reader(
                    cast(str, stream.camera_url),
                    context.queue.source_queue,
                    context.stop_event,
                    context,
                ),
                name="frame_reader",
            ),
            lambda: asyncio.create_task(
                self.broadcast_frames(context),
                name="broadcast",
            ),
            lambda: asyncio.create_task(
                self._periodic_x3d_checker(
                    context=context,
                    camera_id=stream.camera_id,
                    stream_id=stream.id,
                ),
                name="x3d_checker",
            ),
            lambda: asyncio.create_task(
                self._yolo_stream_pipeline(
                    context=context,
                    camera_id=stream.camera_id,
                    stream_id=stream.id,
                ),
                name="yolo",
            ),
            lambda: asyncio.create_task(
                self._mae_stream_pipeline(
                    context=context,
                    camera_id=stream.camera_id,
                    stream_id=stream.id,
                ),
                name="mae",
            ),
        ]

    def _beat(self, context: StreamContext, name: str) -> None:
        context.heartbeats[name] = time.time()

    def _create_context(self) -> StreamContext:
        return StreamContext(
            queue=Queue(),
            heavy=HeavyLogic(),
            stop_event=asyncio.Event(),
            stopped_event=asyncio.Event(),
            heartbeats={},
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
        last_exc = None
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

                except Exception as e:
                    last_exc = e
                    logger.exception(
                        "Triton error (attempt %d/%d)", attempt + 1, retries + 1
                    )

                if attempt < retries:
                    await asyncio.sleep(0.1 * (attempt + 1))

        logger.error("Triton failed after retries", exc_info=last_exc)
        return None

    @staticmethod
    def _open_capture(camera_url: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Failed to open camera stream: {camera_url}")
        return cap

    # The reader loop intentionally keeps reconnect and playback state local.
    # pylint: disable=too-many-locals
    async def _frame_reader(
        self,
        camera_url: str | None,
        frame_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        context: StreamContext,
    ) -> None:
        if camera_url is None:
            logger.warning("Отсутствует URL камеры")
            return

        cap = self._open_capture(camera_url)

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
            try:
                self._beat(context, "frame_reader")
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

            except Exception:
                logger.exception("Frame reader unexpected error")
                await asyncio.sleep(1)
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
                    "label": self.processing.yolo_classes.get(class_id, "unknown"),
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
        event = StreamEventDto(
            id=str(uuid.uuid4()),
            type=event_type,
            camera_id=camera_id,
            stream_id=stream_id,
            payload=payload,
        )

        try:
            await self.event_producer.send_event(event)

        except Exception:
            logger.exception(
                "Failed to emit event (type=%s, event_id=%s, stream_id=%s)",
                event_type,
                event.id,
                stream_id,
            )
            raise

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

    def _build_mae_top_probs(self, probs: NDArray[np.float32]) -> list[dict[str, Any]]:
        top_probs: list[dict[str, Any]] = []
        for class_idx in np.argsort(probs)[::-1][:3]:
            if class_idx not in self.processing.mae_classes:
                continue
            top_probs.append(
                {
                    "class_name": self.processing.mae_classes[class_idx],
                    "probability": float(probs[class_idx]),
                }
            )
        return top_probs

    async def _emit_mae_event(
        self,
        probs: NDArray[np.float32],
        window_ts: list[float],
        camera_id: str,
        stream_id: uuid.UUID,
    ) -> None:
        pred_idx = int(np.argmax(probs))
        await self._emit_event(
            event_type="MAE",
            camera_id=camera_id,
            stream_id=stream_id,
            payload={
                "label": self.processing.mae_classes.get(pred_idx, str(pred_idx)),
                "confidence": float(probs[pred_idx]),
                "start_ts": window_ts[0],
                "end_ts": window_ts[-1],
                "probs": self._build_mae_top_probs(probs),
            },
        )

    async def _yolo_stream_pipeline(
        self, context: StreamContext, camera_id: str, stream_id: uuid.UUID
    ) -> None:
        """Buffer frames for YOLO and flush them in bounded batches."""

        buffer: list[tuple[int, np.ndarray, float]] = []
        last_send = -999.0

        while not context.stop_event.is_set():
            self._beat(context, "YOLO")

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
                        logger.error(
                            "YOLO batch dropped after retries (size=%d, last_frame=%d)",
                            len(batch),
                            batch[-1][0] if batch else -1,
                        )
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

        while not context.stop_event.is_set():
            self._beat(context, "MAE")

            try:
                frame, _, ts = await asyncio.wait_for(
                    context.queue.mae_queue.get(), 0.08
                )
            except asyncio.TimeoutError:
                continue

            # 🔥 Если heavy выключен — просто игнорируем
            if not context.heavy.heavy_active.is_set():
                context.queue.mae_queue.task_done()
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
                    logger.error("MAE inference failed after retries")
                    context.queue.mae_queue.task_done()
                    continue

                await self._emit_mae_event(probs, window_ts, camera_id, stream_id)
                last_inference = ts

            context.queue.mae_queue.task_done()

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
            self._beat(context, "X3D_VIOLENCE")

            await asyncio.sleep(interval)
            if len(x3d_window) < 16:
                continue

            clip = x3d_window[-16:]
            result = await self._quick_x3d_check_for_streaming(clip, context=context)

            if result is None:
                logger.error("X3D inference failed")
                continue

            now = time.time()

            violence_prob = result.get("True", 0.0)

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

                mae_q = context.queue.mae_queue

                cleared_mae = 0

                while not mae_q.empty():
                    try:
                        mae_q.get_nowait()

                        mae_q.task_done()

                        cleared_mae += 1

                    except asyncio.QueueEmpty:
                        break

                if cleared_mae:
                    logger.debug(
                        "Cleared %d frames from mae_queue on heavy disable", cleared_mae
                    )

    async def broadcast_frames(self, context: StreamContext) -> None:
        """Задача-бродкастер: раздаёт каждый кадр во все нужные очереди"""
        while not context.stop_event.is_set():
            self._beat(context, "broadcast")

            try:
                frame_tuple = await asyncio.wait_for(
                    context.queue.source_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            frame, _, _ = frame_tuple

            context.queue.x3d_window.append(frame)
            if len(context.queue.x3d_window) > 64:
                context.queue.x3d_window.pop(0)

            if context.heavy.heavy_active.is_set():
                try:
                    context.queue.mae_queue.put_nowait(frame_tuple)
                except asyncio.QueueFull:
                    logger.warning("mae_queue full — кадр пропущен")

            if context.heavy.heavy_active.is_set():
                try:
                    context.queue.yolo_queue.put_nowait(frame_tuple)
                except asyncio.QueueFull:
                    logger.warning("yolo_queue full — кадр пропущен")

            context.queue.source_queue.task_done()

    async def stop(self, camera_id: str) -> None:
        context = self.contexts.get(camera_id)
        if not context:
            logger.warning("Stop requested but context not found: %s", camera_id)
            return

        logger.info("Stopping stream processor for camera_id=%s", camera_id)

        context.stop_event.set()

        if context.tasks:
            for task in context.tasks:
                task.cancel()

            await asyncio.gather(*context.tasks, return_exceptions=True)

        await self.wait_stopped(camera_id)

    async def wait_stopped(self, camera_id: str, timeout: float = 5) -> bool:
        """Wait until the stream context reports that shutdown has completed."""

        context = self.contexts.get(camera_id)
        if not context:
            return True

        try:
            await asyncio.wait_for(context.stopped_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _watchdog(
        self, task_factories, tasks, context: StreamContext, model: KlinStreamState
    ):
        restart_limits = {
            "frame_reader": 10,
            "broadcast": 10,
            "x3d_checker": 10,
            "yolo": 10,
            "mae": 10,
        }

        task_timeouts = {
            "frame_reader": 10,
            "broadcast": 10,
            "x3d_checker": 10,
            "yolo": 10,
            "mae": 10,
        }

        restart_counts = {name: 0 for name in restart_limits}

        while not context.stop_event.is_set():
            now = time.time()

            for i, task in enumerate(tasks):
                if task is asyncio.current_task():
                    continue

                name = task.get_name()

                if task.done():
                    if task.cancelled():
                        continue

                    exc = task.exception()
                    if exc is None:
                        continue

                    logger.error(
                        "Task crashed: %s (camera_id=%s)",
                        name,
                        model.camera_id,
                        exc_info=exc,
                    )

                    restart_counts[name] += 1

                    if restart_counts[name] > restart_limits[name]:
                        logger.critical("Task %s exceeded restart limit", name)
                        context.stop_event.set()
                        raise RuntimeError(f"Critical task failed: {name}")

                    await asyncio.sleep(min(2 ** restart_counts[name], 10))
                    tasks[i] = task_factories[i]()
                    continue

                last = context.heartbeats.get(name)

                if last is None:
                    continue

                timeout = task_timeouts[name]

                if now - last > timeout:
                    logger.error(
                        "Task %s is STUCK (no heartbeat for %.1fs)",
                        name,
                        now - last,
                    )

                    restart_counts[name] += 1

                    if restart_counts[name] > restart_limits[name]:
                        logger.critical("Task %s exceeded restart limit (stuck)", name)
                        context.stop_event.set()
                        raise RuntimeError(f"Task stuck: {name}")

                    task.cancel()
                    try:
                        await task
                    except Exception:
                        pass

                    await asyncio.sleep(1)

                    logger.info("Restarting stuck task: %s", name)
                    tasks[i] = task_factories[i]()

            await asyncio.sleep(1)

    async def streaming_analyze_once(self, stream: KlinStreamState) -> None:
        """
        Запуск тасок, обработка ошибок
        """
        assert stream.camera_url is not None
        assert stream.camera_id is not None

        context = self._create_context()
        self.contexts[stream.camera_id] = context

        task_factories = self._build_task_factories(context, stream)
        tasks = [factory() for factory in task_factories]
        context.tasks = tasks

        watchdog_task = asyncio.create_task(
            self._watchdog(task_factories, tasks, context, stream),
            name="watchdog",
        )

        tasks.append(watchdog_task)

        logger.info(
            "🚀 СТРИМИНГ ЗАПУЩЕН (camera_id=%s, stream_id=%s, url=%s)",
            stream.camera_id,
            stream.id,
            stream.camera_url,
        )

        try:
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            logger.info("⛔ Stream stopped gracefully (stream_id=%s)", stream.id)
            context.stop_event.set()
            raise

        except Exception:
            logger.exception("Критическая ошибка в стриминге (stream_id=%s)", stream.id)
            context.stop_event.set()
            raise

        finally:
            for task in tasks:
                task.cancel()

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(tasks, results, strict=True):
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logger.error(
                        "Task failed during shutdown: %s",
                        task.get_name(),
                        exc_info=result,
                    )

            logger.info("Стриминг завершён (stream_id=%s)", stream.id)

            context.stopped_event.set()

            try:
                await self.event_producer.send_event(
                    StreamEventDto(
                        id=str(uuid.uuid4()),
                        type="STREAM_STOPPED",
                        camera_id=stream.camera_id,
                        stream_id=stream.id,
                        payload={"timestamp": datetime.now(timezone.utc).isoformat()},
                    )
                )

            except Exception:
                logger.exception("Failed to send STREAM_STOPPED event")

            self.contexts.pop(stream.camera_id, None)

    async def run_stream_with_restarts(
        self,
        stream: KlinStreamState,
        max_restarts: int = 5,
    ) -> None:
        """Restart the whole streaming pipeline with backoff after fatal failures."""
        restart_count = 0

        while restart_count <= max_restarts:
            context = self.contexts.get(stream.camera_id)

            if context and context.stop_event.is_set():
                logger.info("Stream stopped manually → exit restart loop")
                return
            try:
                await self.streaming_analyze_once(stream)
                return

            except asyncio.CancelledError:
                logger.info("Stream cancelled → no restart")
                raise

            except Exception:
                restart_count += 1

                logger.exception(
                    "Stream crashed (attempt %d/%d)",
                    restart_count,
                    max_restarts,
                )

                context = self.contexts.get(stream.camera_id)

                if context and context.stop_event.is_set():
                    logger.info("Stream stopped manually → no restart")
                    return

                if restart_count > max_restarts:
                    logger.critical("Stream exceeded max restarts → giving up")
                    return

                delay = min(2**restart_count, 30)
                await asyncio.sleep(delay)

    async def streaming_analyze(self, stream: KlinStreamState) -> None:
        await self.run_stream_with_restarts(stream)
