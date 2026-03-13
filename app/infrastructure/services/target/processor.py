"""
Процессор для взаимодействия с приложением litestar
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from app.config import app_settings
from app.models.klin import KlinModel


logger = logging.getLogger(__name__)

BASE_DIR_MAE = Path(__file__).parent.parent.parent.parent.parent
MAE_DIR = BASE_DIR_MAE / app_settings.videomae_path
YOLO_DIR = BASE_DIR_MAE / app_settings.yolo_path
X3D_DIR = BASE_DIR_MAE / app_settings.x3d_path


# pylint: disable=too-many-instance-attributes
@dataclass
class ProcessorConfig:
    """
    Переменные для процессора
    """

    chunk_size: int = 16
    frame_size: tuple[int, int] = (224, 224)
    yolo_stride: int = 2
    yolo_batch_size: int = 32
    yolo_conf: float = 0.6
    yolo_classes: dict[int, str] = field(default_factory=lambda: {0: "person"})
    allowed_classes: set[int] = field(default_factory=lambda: {0})
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


# pylint: disable=too-few-public-methods
class InferenceProcessor(IKlinInference):
    """
    Процессор. Содержит методы подключения моделей,
    обработки видео с помощью videomae и yolo.
    """

    def __init__(self) -> None:
        self.processing = ProcessorConfig()
        self.triton = httpclient.InferenceServerClient(url="localhost:8000")

    def _prepare_x3d_for_triton(self, frames: list[np.ndarray]) -> np.ndarray:
        frames_np = np.array(frames).astype(np.float32) / 255.0
        frames_np = np.transpose(frames_np, (3, 0, 1, 2))
        frames_np = np.expand_dims(frames_np, axis=0)
        return frames_np

    def _quick_x3d_check(self, video_path: str) -> dict[str, float]:
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

        video = self._prepare_x3d_for_triton(frames_list)

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

    def _prepare_yolo_frame_for_triton(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _infer_yolo_batch(self, batch_imgs: np.ndarray) -> list[NDArray[np.float32]]:
        """Один вызов Triton на весь батч"""
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

    def _parse_yolo_detection(
        self, pred: np.ndarray, timesteps: float
    ) -> dict[str, Any] | None:
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

        return {
            "class_id": class_id,
            "timesteps": timesteps,
            "bbox": bbox,
        }

    # pylint: disable=too-many-locals
    def _process_yolo_batch(self, state: VideoStreamState, fps: float) -> None:
        """Обрабатывает накопленный батч кадров YOLO"""
        if not state.yolo_buffer:
            return

        batch_imgs_list = []
        frame_indices = []

        for f_idx, frame in state.yolo_buffer:
            img = self._prepare_yolo_frame_for_triton(frame)
            batch_imgs_list.append(img[0])  # убираем размерность batch=1
            frame_indices.append(f_idx)

        batch_array = np.stack(batch_imgs_list, axis=0)  # (B, 3, 640, 640)
        batch_preds_list = self._infer_yolo_batch(batch_array)

        for i, preds in enumerate(batch_preds_list):
            f_idx = frame_indices[i]
            timesteps = f_idx / fps if fps > 0 else 0.0

            for pred in preds:
                det = self._parse_yolo_detection(pred, timesteps)
                if det:
                    timestep = float(det["timesteps"])
                    state.bbox_by_time[timestep].append(det["bbox"])
                    state.detected_class_ids.add(int(det["class_id"]))

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

    def _prepare_mae_chunk_for_triton(self, frames: list[np.ndarray]) -> np.ndarray:
        frames_np = np.array(frames).astype(np.float32) / 255.0
        frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # T C H W
        frames_np = np.expand_dims(frames_np, axis=0)  # B T C H W
        return frames_np

    # pylint: disable=too-many-locals
    def _predict_mae_chunk(
        self,
        *,
        chunk_frames: list[np.ndarray],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> dict[str, Any]:
        img = self._prepare_mae_chunk_for_triton(chunk_frames)

        inputs = httpclient.InferInput("pixel_values", img.shape, "FP32")
        inputs.set_data_from_numpy(img)
        outputs = httpclient.InferRequestedOutput("logits")

        result = self.triton.infer(
            model_name="videomae_crime",
            inputs=[inputs],
            outputs=[outputs],
        )

        logits = result.as_numpy("logits")[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        answer = self.processing.mae_classes.get(pred_idx, str(pred_idx))

        start_time = start_frame / fps if fps > 0 else 0.0
        end_time = (end_frame + 1) / fps if fps > 0 else 0.0

        return {
            "time": [start_time, end_time],
            "answer": answer,
            "confident": conf,
        }

    def _update_mae_stream_state(
        self, frame_resized: np.ndarray, state: VideoStreamState, fps: float
    ) -> None:
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.processing.frame_size)

        self._update_yolo_stream_state(frame_resized, state, fps)
        self._update_mae_stream_state(frame_resized, state, fps)
        state.frame_idx += 1

    def _flush_partial_chunk(self, state: VideoStreamState, fps: float) -> None:
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
        """Добрабатываем остаток батча YOLO в конце видео"""
        if state.yolo_buffer:
            self._process_yolo_batch(state, fps)
            state.yolo_buffer.clear()

    def _collect_stream_results(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        video_path: str,
    ) -> VideoStreamState:
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

    def _build_video_info(
        self,
        *,
        total_frames: int,
        fps: float,
        duration: float,
        frames_read: int,
    ) -> dict[str, Any]:
        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "frames_read": frames_read,
        }

    def _resolve_detected_objects(self, detected_class_ids: set[int]) -> list[str]:
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
        video_info = self._build_video_info(
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            frames_read=state.frame_idx,
        )
        return state.mae_results, dict(state.bbox_by_time), objects, video_info

    def _log_processing(
        self, video_name: str, video_info: dict[str, Any], processing_time: float
    ) -> None:
        logger.info(
            "РЕЗУЛЬТАТЫ АНАЛИЗА ВИДЕО: video=%s "
            "duration=%.1fs processing=%.2fs frames=%d/%d",
            video_name,
            float(video_info["duration"]),
            processing_time,
            int(video_info["frames_read"]),
            int(video_info["total_frames"]),
        )

    async def analyze(self, model: KlinModel) -> KlinResultDto:
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
            self._log_processing(video_name, video_info, processing_time)

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
