"""
Процессор для взаимодействия с приложением litestar (РЕАЛ-ТАЙМ СТРИМИНГ БЕЗ CALLBACK)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import cast

import cv2
import msgspec
import numpy as np
import torch
import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]
from numpy.typing import NDArray

from app.application.interfaces import IKlinStream
from app.infrastructure.helpers import (
    HeavyLogic,
    LoggingHelper,
    MaeConfig,
    PrepareForTriton,
    Queue,
    StreamConfig,
    YoloConfig,
)
from app.models.klin import KlinStreamingModel


logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Конфигурация процессора."""

    stream: StreamConfig = field(default_factory=StreamConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    mae: MaeConfig = field(default_factory=MaeConfig)
    x3d_conf: float = 0.68

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


class StreamProcessor(IKlinStream):
    def __init__(self) -> None:
        self.processing = ProcessorConfig()
        self.prepare = PrepareForTriton()
        self.logging = LoggingHelper()
        self.queue = Queue()
        self.heavy = HeavyLogic()
        self.triton = grpcclient.InferenceServerClient(url="127.0.0.1:8001")

    def infer_fn(self, video_arr):
        inputs = grpcclient.InferInput("video", video_arr.shape, "FP32")
        inputs.set_data_from_numpy(video_arr)
        outputs = grpcclient.InferRequestedOutput("logits")
        result = self.triton.infer(
            model_name="x3d_violence", inputs=[inputs], outputs=[outputs]
        )
        return result.as_numpy("logits")[0]

    async def _quick_x3d_check_for_streaming(
        self, frames: list[np.ndarray]
    ) -> dict[str, float] | None:
        video = self.prepare.prepare_x3d_for_triton(frames)

        logits = await self._triton_infer_with_retry(self.infer_fn, video)
        if logits is None:
            return None

        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        return {str(bool(pred)): confidence}

    async def _triton_infer_with_retry(
        self,
        infer_fn,
        *args,
        timeout: float = 10,
        retries: int = 4,
    ):
        loop = asyncio.get_running_loop()

        async with self.queue.infer_semaphore:
            for attempt in range(retries + 1):
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(self.queue.executor, infer_fn, *args),
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

    def _infer_yolo_batch(self, batch_imgs: np.ndarray) -> list[NDArray[np.float32]]:
        inputs = grpcclient.InferInput("images", batch_imgs.shape, "FP32")
        inputs.set_data_from_numpy(batch_imgs)
        outputs = grpcclient.InferRequestedOutput("output0")
        result = self.triton.infer(
            model_name="yolo_person", inputs=[inputs], outputs=[outputs]
        )
        raw_output: NDArray[np.float32] = result.as_numpy("output0")
        return [raw_output[b].transpose(1, 0) for b in range(raw_output.shape[0])]

    def _infer_mae_probs(self, chunk_frames: list[np.ndarray]) -> NDArray[np.float32]:
        img = self.prepare.prepare_mae_chunk_for_triton(chunk_frames)
        inputs = grpcclient.InferInput("pixel_values", img.shape, "FP32")
        inputs.set_data_from_numpy(img)
        outputs = grpcclient.InferRequestedOutput("logits")

        result = self.triton.infer(
            model_name="videomae_crime", inputs=[inputs], outputs=[outputs]
        )

        logits = np.asarray(result.as_numpy("logits")[0], dtype=np.float32)
        exp = np.exp(logits - np.max(logits))

        probs = exp / np.sum(exp)
        return cast(NDArray[np.float32], probs.astype(np.float32, copy=False))

    def _parse_yolo_detection(self, pred: np.ndarray) -> tuple[int, list[float]] | None:
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        conf = scores[class_id]
        if (
            conf < self.processing.yolo_conf
            or class_id not in self.processing.allowed_classes
        ):
            return None
        x, y, w, h = pred[:4]
        bbox = [
            float(x - w / 2),
            float(y - h / 2),
            float(x + w / 2),
            float(y + h / 2),
        ]
        return class_id, bbox

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
                await frame_queue.put((resized, frame_idx, time.time()))
            except asyncio.QueueFull:
                logger.debug("Очередь переполнена — кадр пропущен")

            frame_idx += 1

            # ===================== REAL-TIME ДЛЯ ФАЙЛА =====================
            if is_file and frame_delay > 0:
                await asyncio.sleep(frame_delay)

        cap.release()

    async def _yolo_stream_pipeline(
        self,
        frame_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        camera_id: str,
    ) -> None:
        buffer: list[tuple[int, np.ndarray, float]] = []
        last_send = -999.0

        while not stop_event.is_set():
            try:
                frame, idx, ts = await asyncio.wait_for(frame_queue.get(), timeout=0.08)
            except asyncio.TimeoutError:
                continue

            # Ключевое условие: тяжёлый анализ включён?
            if not self.heavy.heavy_active.is_set():
                frame_queue.task_done()
                continue

            if idx % self.processing.yolo_stride == 0:
                buffer.append((idx, frame, ts))

            if len(buffer) >= self.processing.yolo_batch_size or (
                buffer and ts - last_send > 0.8
            ):
                batch_imgs = np.stack(
                    [
                        self.prepare.prepare_yolo_frame_for_triton(f)[0]
                        for _, f, _ in buffer
                    ],
                    axis=0,
                )
                preds_list = await self._triton_infer_with_retry(
                    self._infer_yolo_batch,
                    batch_imgs,
                )
                if preds_list is None:
                    continue

                detections = []
                for (_idx, _, ts), preds in zip(buffer, preds_list, strict=True):
                    for pred in preds:
                        det = self._parse_yolo_detection(pred)
                        if det:
                            class_id, bbox = det
                            detections.append(
                                {
                                    "class_id": class_id,
                                    "class_name": self.processing.yolo_classes.get(
                                        class_id, "unknown"
                                    ),
                                    "bbox": bbox,
                                    "confidence": float(pred[4 + class_id]),
                                    "timestamp": ts,
                                }
                            )

                if detections:
                    event = {
                        "type": "YOLO",
                        "camera_id": camera_id,
                        "frame_idx": idx,
                        "timestamp": ts,
                        "detections": detections,
                    }
                    logger.info(
                        "STREAM_EVENT: %s", msgspec.json.encode(event).decode("utf-8")
                    )

                buffer.clear()
                last_send = ts

            frame_queue.task_done()

    async def _mae_stream_pipeline(
        self,
        frame_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        x3d_window: list[np.ndarray],
        camera_id: str,
    ) -> None:
        window: list[np.ndarray] = []
        window_ts: list[float] = []
        last_inference = -999.0

        while not stop_event.is_set():
            try:
                frame, _, ts = await asyncio.wait_for(frame_queue.get(), 0.08)
            except asyncio.TimeoutError:
                continue

            x3d_window.append(frame)
            if len(x3d_window) > 64:
                x3d_window.pop(0)

            # Ключевое условие: тяжёлый анализ включён?
            if not self.heavy.heavy_active.is_set():
                frame_queue.task_done()
                continue

            window.append(frame)
            window_ts.append(ts)

            if len(window) > self.processing.chunk_size:
                window.pop(0)
                window_ts.pop(0)

            if len(window) == self.processing.chunk_size and ts - last_inference > 2.0:
                probs = await self._triton_infer_with_retry(
                    self._infer_mae_probs,
                    window[:],
                )
                if probs is None:
                    continue
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                label = self.processing.mae_classes.get(pred_idx, str(pred_idx))

                event = {
                    "type": "MAE",
                    "camera_id": camera_id,
                    "label": label,
                    "confidence": confidence,
                    "probs": probs.tolist(),
                    "start_ts": window_ts[0],
                    "end_ts": window_ts[-1],
                }
                logger.info(
                    "STREAM_EVENT: %s", msgspec.json.encode(event).decode("utf-8")
                )
                last_inference = ts

            frame_queue.task_done()

    async def _periodic_x3d_checker(
        self,
        x3d_window: list[np.ndarray],
        stop_event: asyncio.Event,
        camera_id: str,
        interval: float = 3.0,
    ) -> None:
        """Лёгкий постоянный чекер X3D. При триггере включает тяжёлый анализ."""
        while not stop_event.is_set():
            await asyncio.sleep(interval)
            if len(x3d_window) < 16:
                continue

            clip = x3d_window[-16:]
            result = await self._quick_x3d_check_for_streaming(clip)
            now = time.time()
            violence_prob = result.get("True", 0.0) if result is not None else 0.0

            if violence_prob > self.processing.x3d_conf:
                logger.info("X3D TRIGGER → heavy ON (prob=%.3f)", violence_prob)
                self.heavy.heavy_active.set()
                self.heavy.last_trigger_time = now

                event = {
                    "type": "X3D_VIOLENCE",
                    "camera_id": camera_id,
                    "prob": violence_prob,
                    "timestamp": now,
                }
                logger.info(
                    "STREAM_EVENT: %s", msgspec.json.encode(event).decode("utf-8")
                )

            # Авто-выключение тяжёлого режима
            elif (
                self.heavy.heavy_active.is_set()
                and now - self.heavy.last_trigger_time > self.heavy.HEAVY_COOLDOWN
            ):
                logger.info("X3D stable normal → heavy OFF")
                self.heavy.heavy_active.clear()

    # Задача-бродкастер: раздаёт каждый кадр во все нужные очереди
    async def broadcast_frames(self):
        """
        Задача-бродкастер: раздаёт каждый кадр во все нужные очереди
        """
        while not self.queue.stop_event.is_set():
            frame_tuple = await self.queue.source_queue.get()

            for q in (self.queue.yolo_queue, self.queue.mae_queue):
                try:
                    q.put_nowait(frame_tuple)
                except asyncio.QueueFull:
                    logger.debug("Очередь переполнена — кадр пропущен")

            self.queue.source_queue.task_done()

    async def streaming_analyze(self, model: KlinStreamingModel) -> None:
        assert model.camera_url is not None
        assert model.camera_id is not None

        self.queue.x3d_window = []

        tasks = [
            asyncio.create_task(
                self._frame_reader(
                    model.camera_url, self.queue.source_queue, self.queue.stop_event
                )
            ),
            asyncio.create_task(self.broadcast_frames()),
            asyncio.create_task(
                self._periodic_x3d_checker(
                    self.queue.x3d_window,
                    self.queue.stop_event,
                    model.camera_id,
                    interval=3.0,
                )
            ),
            asyncio.create_task(
                self._yolo_stream_pipeline(
                    self.queue.yolo_queue, self.queue.stop_event, model.camera_id
                )
            ),
            asyncio.create_task(
                self._mae_stream_pipeline(
                    self.queue.mae_queue,
                    self.queue.stop_event,
                    self.queue.x3d_window,
                    model.camera_id,
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
            self.queue.stop_event.set()
            # ждём завершения
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            logger.exception("Критическая ошибка в стриминге (id=%s)", model.camera_id)
            self.queue.stop_event.set()
            raise
        finally:
            logger.info("Стриминг завершён (id=%s)", model.camera_id)
