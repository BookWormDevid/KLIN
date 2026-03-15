"""
Процессор для взаимодействия с приложением litestar
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

import aiohttp
import async_timeout
import cv2
import msgspec
import numpy as np
import torch
import tritonclient.http as httpclient  # type: ignore[import-untyped]
from numpy.typing import NDArray

from app.application.dto import KlinResultDto
from app.application.interfaces import IKlinCallbackSender, IKlinInference
from app.infrastructure.helpers import (
    LoggingHelper,
    MaeConfig,
    PrepareForTriton,
    StreamConfig,
    TimeRangeHelper,
    YoloConfig,
)
from app.models.klin import KlinModel


logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Конфигурация процессора."""

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


class InferenceProcessor(IKlinInference):  # pylint: disable=too-few-public-methods
    """
    Процессор. Содержит методы подключения моделей,
    обработки видео с помощью videomae, yolo, x3d.
    """

    def __init__(self) -> None:
        self.processing = ProcessorConfig()
        self.prepare = PrepareForTriton()
        self.timerange = TimeRangeHelper()
        self.logging = LoggingHelper()
        self.triton = httpclient.InferenceServerClient(url="localhost:8000")

    def _quick_x3d_check(self, video_path: str) -> dict[str, float]:
        """
        Просматриваем кадры,
        подготовленные данных отправляем в triton,
        вычисляем предсказанный класс и вероятность.
        """
        check_answer: dict[str, float] = {}

        cap = cv2.VideoCapture(video_path)
        frames_list: list[np.ndarray] = []

        while len(frames_list) < 16:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)

        cap.release()

        if len(frames_list) < 16:
            check_answer["False"] = 0.0
            return check_answer

        video = self.prepare.prepare_x3d_for_triton(frames_list)

        inputs = httpclient.InferInput("video", video.shape, "FP32")
        inputs.set_data_from_numpy(video)
        outputs = httpclient.InferRequestedOutput("logits")

        result = self.triton.infer(
            model_name="x3d_violence",
            inputs=[inputs],
            outputs=[outputs],
        )

        logits = result.as_numpy("logits")[0]
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()

        pred = int(np.argmax(probs))
        confidence = float(probs[pred])

        check_answer[str(bool(pred))] = confidence
        return check_answer

    def _infer_yolo_batch(self, batch_imgs: np.ndarray) -> list[NDArray[np.float32]]:
        """
        Один вызов Triton на весь батч
        """
        inputs = httpclient.InferInput("images", batch_imgs.shape, "FP32")
        inputs.set_data_from_numpy(batch_imgs)

        outputs = httpclient.InferRequestedOutput("output0")

        result = self.triton.infer(
            model_name="yolo_person",
            inputs=[inputs],
            outputs=[outputs],
        )

        raw_output: NDArray[np.float32] = result.as_numpy("output0")

        batch_preds: list[NDArray[np.float32]] = []
        for b in range(raw_output.shape[0]):
            per_image = raw_output[b]
            preds = per_image.transpose(1, 0)
            batch_preds.append(preds)

        return batch_preds

    def _parse_yolo_detection(self, pred: np.ndarray) -> tuple[int, list[float]] | None:
        """
        Вычисляем класс id и bbox
        """
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        conf = scores[class_id]

        if conf < self.processing.yolo_conf:
            return None
        if class_id not in self.processing.allowed_classes:
            return None

        x, y, w, h = pred[:4]
        bbox = [
            float(x - w / 2),
            float(y - h / 2),
            float(x + w / 2),
            float(y + h / 2),
        ]

        return class_id, bbox

    def _build_yolo_batch(
        self, yolo_buffer: list[tuple[int, np.ndarray]]
    ) -> tuple[np.ndarray, list[int]]:
        """
        Подготовленные данные оборачиваем в батч.
        """
        frame_indices = [frame_idx for frame_idx, _ in yolo_buffer]
        batch_imgs = [
            self.prepare.prepare_yolo_frame_for_triton(frame)[0]
            for _, frame in yolo_buffer
        ]
        return np.stack(batch_imgs, axis=0), frame_indices

    def _store_yolo_detections(
        self,
        state: VideoStreamState,
        preds: np.ndarray,
        timesteps: float,
    ) -> None:
        """
        Заворачиваем вывод в нужные типы данных
        """
        for pred in preds:
            detection = self._parse_yolo_detection(pred)
            if detection is None:
                continue
            class_id, bbox = detection
            state.bbox_by_time[timesteps].append(bbox)
            state.detected_class_ids.add(class_id)

    def _process_yolo_batch(self, state: VideoStreamState, fps: float) -> None:
        """
        Обрабатывает накопленный батч кадров YOLO
        """
        if not state.yolo_buffer:
            return

        batch_array, frame_indices = self._build_yolo_batch(state.yolo_buffer)
        batch_preds_list = self._infer_yolo_batch(batch_array)

        for frame_idx, preds in zip(frame_indices, batch_preds_list, strict=False):
            timesteps = frame_idx / fps if fps > 0 else 0.0
            self._store_yolo_detections(state, preds, timesteps)

    def _update_yolo_stream_state(
        self, frame_resized: np.ndarray, state: VideoStreamState, fps: float
    ) -> None:
        """
        Собирает кадры в батч (каждый yolo_stride кадр).
        Когда батч заполнен — сразу обрабатывает.
        """
        if state.frame_idx % self.processing.yolo_stride != 0:
            return

        state.yolo_buffer.append((state.frame_idx, frame_resized))

        if len(state.yolo_buffer) >= self.processing.yolo_batch_size:
            self._process_yolo_batch(state, fps)
            state.yolo_buffer.clear()

    def _infer_mae_probs(self, chunk_frames: list[np.ndarray]) -> NDArray[np.float32]:
        """
        Подготовленные данные отправляем в triton
        """
        img = self.prepare.prepare_mae_chunk_for_triton(chunk_frames)

        inputs = httpclient.InferInput("pixel_values", img.shape, "FP32")
        inputs.set_data_from_numpy(img)
        outputs = httpclient.InferRequestedOutput("logits")

        result = self.triton.infer(
            model_name="videomae_crime",
            inputs=[inputs],
            outputs=[outputs],
        )

        logits = np.asarray(result.as_numpy("logits")[0], dtype=np.float32)
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        return cast(NDArray[np.float32], probs.astype(np.float32, copy=False))

    def _predict_mae_chunk(
        self,
        *,
        chunk_frames: list[np.ndarray],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> dict[str, Any]:
        """
        Вычисляем из полученных данных время, класс, уверенность
        """
        probs = self._infer_mae_probs(chunk_frames)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        answer = self.processing.mae_classes.get(pred_idx, str(pred_idx))

        return {
            "time": self.timerange.build_time_range(start_frame, end_frame, fps),
            "answer": answer,
            "confident": confidence,
        }

    def _update_mae_stream_state(
        self, frame_resized: np.ndarray, state: VideoStreamState, fps: float
    ) -> None:
        """
        Обновляем полученные данные
        """
        state.chunk_buffer.append(frame_resized)
        if len(state.chunk_buffer) != self.processing.chunk_size:
            return

        state.mae_results.append(
            self._predict_mae_chunk(
                chunk_frames=state.chunk_buffer,
                start_frame=state.chunk_start_frame,
                end_frame=state.frame_idx,
                fps=fps,
            )
        )
        state.chunk_buffer = []
        state.chunk_start_frame = state.frame_idx + 1

    def _process_stream_frame(
        self, frame: np.ndarray, state: VideoStreamState, fps: float
    ) -> None:
        """
        Собираем полученные данные вместе
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.processing.frame_size)

        self._update_yolo_stream_state(frame_resized, state, fps)
        self._update_mae_stream_state(frame_resized, state, fps)
        state.frame_idx += 1

    def _flush_partial_chunk(self, state: VideoStreamState, fps: float) -> None:
        """
        Дообрабатываем остаток чанка mae в конце видео
        """
        if not state.chunk_buffer:
            return

        pad_count = self.processing.chunk_size - len(state.chunk_buffer)
        if pad_count > 0:
            state.chunk_buffer.extend([state.chunk_buffer[-1]] * pad_count)

        state.mae_results.append(
            self._predict_mae_chunk(
                chunk_frames=state.chunk_buffer,
                start_frame=state.chunk_start_frame,
                end_frame=state.frame_idx - 1,
                fps=fps,
            )
        )

    def _flush_yolo_batch(self, state: VideoStreamState, fps: float) -> None:
        """
        Добрабатываем остаток батча YOLO в конце видео"""
        if state.yolo_buffer:
            self._process_yolo_batch(state, fps)
            state.yolo_buffer.clear()

    def _collect_stream_results(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        video_path: str,
    ) -> VideoStreamState:
        """
        Все данные вместе уже с обработанными чанками и батчами
        """
        state = VideoStreamState()
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self._process_stream_frame(frame, state, fps)

        if state.frame_idx == 0:
            raise ValueError(f"Кадры не прочитаны: {video_path}")

        self._flush_partial_chunk(state, fps)
        self._flush_yolo_batch(state, fps)
        return state

    def _resolve_detected_objects(self, detected_class_ids: set[int]) -> list[str]:
        """
        Вычисляем классы yolo
        """
        return [
            self.processing.yolo_classes[c]
            for c in detected_class_ids
            if c in self.processing.yolo_classes
        ]

    async def _process_video_stream(
        self, video_path: str
    ) -> tuple[
        list[dict[str, Any]],
        dict[float, list[list[float]]],
        list[str],
        dict[str, Any],
    ]:
        """
        Процессор обработки
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        duration = (total_frames / fps) if fps > 0 else 0.0

        try:
            state = self._collect_stream_results(cap, fps, video_path)
        finally:
            cap.release()

        objects = self._resolve_detected_objects(state.detected_class_ids)
        video_info = self.logging.build_video_info(
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            frames_read=state.frame_idx,
        )
        return state.mae_results, dict(state.bbox_by_time), objects, video_info

    async def analyze(self, model: KlinModel) -> KlinResultDto:
        """
        Конечный метод для вызова процессора
        Вначале x3d проверяет есть ли аномалия на видео.
        Если есть включаем mae и yolo.
        """
        start_ts = time.time()
        video_name = os.path.basename(model.video_path)

        result_dict = self._quick_x3d_check(model.video_path)
        is_fight = list(result_dict.keys())[0]

        if is_fight == "False":
            return KlinResultDto(
                x3d=msgspec.json.encode(result_dict).decode("utf-8"),
                mae="[]",
                yolo="{}",
                all_classes=[],
                objects=[],
            )

        try:
            (
                mae_results,
                yolo_bbox,
                detected_objects,
                video_info,
            ) = await self._process_video_stream(model.video_path)

            all_classes = list({result["answer"] for result in mae_results})

            processing_time = time.time() - start_ts
            self.logging.log_processing(video_name, video_info, processing_time)

            return KlinResultDto(
                x3d=msgspec.json.encode(result_dict).decode("utf-8"),
                mae=msgspec.json.encode(mae_results).decode("utf-8"),
                yolo=msgspec.json.encode(yolo_bbox).decode("utf-8"),
                all_classes=all_classes,
                objects=detected_objects,
            )
        except Exception as exc:
            logger.exception("Ошибка обработки видео %s: %s", video_name, exc)
            raise


class KlinCallbackSender(IKlinCallbackSender):  # pylint: disable=too-few-public-methods
    """
    Класс для отправки вывода результата
    """

    async def post_consumer(self, model: KlinModel) -> None:
        """
        Отправляет вывод результата в виде json.
        Если попыток больше чем 3 выдаёт ошибку и выдаёт ошибку.
        """

        if not model.response_url:
            return

        payload: dict[str, Any] = {
            "klin_id": str(model.id),
            "x3d": model.x3d,
            "mae": model.mae,
            "yolo": model.yolo,
            "objects": model.objects,
            "all_classes": model.all_classes,
            "state": model.state,
        }

        data = msgspec.json.encode(payload)

        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                async with async_timeout.timeout(30):  # 30 секунд таймаут
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            model.response_url,
                            data=data,
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            if 200 <= resp.status < 300:
                                return

                            body = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status}, body={body}")

            except Exception as exc:
                if attempt == max_attempts:
                    logger.error(
                        "Callback failed after %d attempts. "
                        "model.id=%s response_url=%s error=%s",
                        max_attempts,
                        model.id,
                        model.response_url,
                        exc,
                    )
                    return
