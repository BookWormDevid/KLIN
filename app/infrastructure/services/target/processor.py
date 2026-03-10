"""
Процессор для взаимодействия с приложением litestar
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import aiohttp
import async_timeout
import cv2
import msgspec
import numpy as np
import torch
import tritonclient.http as httpclient
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from ultralytics import YOLO

from app.application.dto import KlinResultDto
from app.application.interfaces import IKlinCallbackSender, IKlinInference
from app.config import app_settings
from app.infrastructure.services.target.x3d_net.x3d_net import generate_model
from app.models.klin import KlinModel


logger = logging.getLogger(__name__)

BASE_DIR_MAE = Path(__file__).parent.parent.parent.parent.parent
MAE_DIR = BASE_DIR_MAE / app_settings.videomae_path
YOLO_DIR = BASE_DIR_MAE / app_settings.yolo_path
X3D_DIR = BASE_DIR_MAE / app_settings.x3d_path


@dataclass
class VideoMAEConfig:
    """
    Переменные для videomae
    """

    model: VideoMAEForVideoClassification | None = None
    processor: VideoMAEImageProcessor | None = None
    mae_model: str | None = None


@dataclass
class YoloConfig:
    """
    Переменные для yolo
    """

    yolo: YOLO | None = None
    yolo_path: str | None = None


@dataclass
class X3DConfig:
    """
    Переменные для x3d
    """

    model: torch.nn.Module | None = None
    model_path: str | None = None


@dataclass
class ProcessorConfig:
    """
    Переменные для процессора
    """

    chunk_size: int = 16
    frame_size: tuple[int, int] = (224, 224)
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    yolo_stride: int = 2
    yolo_conf: float = 0.6
    yolo_iou: float = 0.45
    yolo_classes: list[int] = field(default_factory=lambda: [0])


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
    chunk_start_frame: int = 0
    frame_idx: int = 0


class InferenceProcessor(IKlinInference):
    """
    Процессор. Содержит методы подключения моделей,
    обработки видео с помощью videomae и yolo.
    """

    def __init__(self) -> None:
        """
        Содержит переменные для методов.
        self.model - модель videomae
        self.processor - процессор videomae
        self.mae_model - путь к модели videomae
        self.yolo_path - путь к модели yolo
        self.yolo - модель yolo
        self.chunk_size - разделение кадров на чанки
        self.frame_size - разрешение для чтение кадров
        self.device - cuda
        """
        self.mae = VideoMAEConfig()
        self.yolo = YoloConfig()
        self.processing = ProcessorConfig()
        self.x3d = X3DConfig()
        self.triton = httpclient.InferenceServerClient(url="triton:8000")

    def _ensure_models_loaded(self) -> None:
        """
        Проверка, что обе модели загружены.
        """
        if self.mae.model is None:
            self.ensure_mae_model_loaded()
        if self.yolo.yolo is None:
            self.ensure_yolo_loaded()

        assert self.mae.model is not None
        assert self.mae.processor is not None
        assert self.yolo.yolo is not None

    def ensure_mae_model_loaded(self) -> None:
        """
        Проверка, что модель videomae загружена.
        Загружает код для модели и процессор из библиотеки
        Проверяет cuda к модели
        """
        if self.mae.model is not None:
            return

        self.mae.mae_model = self.find_mae_path()
        self.mae.processor = VideoMAEImageProcessor.from_pretrained(
            self.mae.mae_model, local_files_only=True
        )

        model = cast(
            VideoMAEForVideoClassification,
            VideoMAEForVideoClassification.from_pretrained(
                self.mae.mae_model, local_files_only=True
            ),
        )
        cast(torch.nn.Module, model).to(self.processing.device)
        model.eval()
        self.mae.model = model

    def find_mae_path(self) -> str:
        """
        Автоматически найти путь к модели videomae
        Ищет по заданному пути
        """
        self.mae.mae_model = str(MAE_DIR)
        return self.mae.mae_model

    def find_yolo_path(self) -> str:
        """
        Автоматически найти путь к модели yolo
        Ищет по заданному пути
        """
        self.yolo.yolo_path = str(YOLO_DIR)
        return self.yolo.yolo_path

    def find_x3d_path(self) -> str:
        self.x3d.model_path = str(X3D_DIR)
        return self.x3d.model_path

    def ensure_yolo_loaded(self) -> None:
        """
        Загружает модель и проверяет cuda к модели
        """
        if self.yolo.yolo is not None:
            return

        weights_path = self.find_yolo_path()
        self.yolo.yolo = YOLO(weights_path)

        self.yolo.yolo.to(self.processing.device)

    def ensure_x3d_loaded(self) -> None:
        """
        Загружает модель x3d
        """
        if self.x3d.model is not None:
            return

        model = generate_model(
            x3d_version="M",  # S, M, XL
            n_classes=2,
            n_input_channels=3,
            dropout=0,
            base_bn_splits=1,
        )

        model = torch.nn.DataParallel(model)

        # Загружаем веса
        weights = torch.load(X3D_DIR, map_location=self.processing.device)
        model.load_state_dict(weights)

        model.to(self.processing.device)
        model.eval()

        self.x3d.model = model

    def _quick_x3d_check(self, video_path: str) -> dict[int, float]:
        """
        Возвращает True если обнаружена драка.
        """
        check_answer = {}
        self.ensure_x3d_loaded()

        if self.x3d.model is None:
            raise RuntimeError("Модель X3D не загружена")

        cap = cv2.VideoCapture(video_path)
        frames_list: list = []

        while len(frames_list) < 16:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)

        cap.release()

        if len(frames_list) < 16:
            check_answer[0] = 0.0
            return check_answer

        frames_np = np.array(frames_list)

        frames_tensor = (
            torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        )
        frames_tensor = frames_tensor.to(self.processing.device)

        with torch.no_grad():
            output = self.x3d.model(frames_tensor)

        probs = torch.nn.functional.softmax(output.squeeze(), dim=0)
        confidence = probs[1].item()  # уверенность для класса 'fight'
        pred = torch.argmax(probs).item()
        check_answer[int(pred)] = confidence

        return check_answer  # если 1 = fight

    def _prepare_yolo_frame_for_triton(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def _run_yolo_on_frame(
        self, frame: np.ndarray, frame_idx: int, fps: float
    ) -> list[dict[str, Any]]:
        timesteps = frame_idx / fps if fps > 0 else 0.0

        img = self._prepare_yolo_frame_for_triton(frame)

        inputs = httpclient.InferInput(
            "images",
            img.shape,
            "FP32",
        )

        inputs.set_data_from_numpy(img)

        outputs = httpclient.InferRequestedOutput("output0")

        result = self.triton.infer(
            model_name="yolo",
            inputs=[inputs],
            outputs=[outputs],
        )

        preds = result.as_numpy("output0")

        detections: list[dict[str, Any]] = []

        for pred in preds[0]:
            conf = pred[4]

            if conf < self.processing.yolo_conf:
                continue

            class_id = int(np.argmax(pred[5:]))

            if class_id not in self.processing.yolo_classes:
                continue

            x, y, w, h = pred[:4]

            bbox = [
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2,
            ]

            detections.append(
                {
                    "class_id": class_id,
                    "timesteps": float(timesteps),
                    "bbox": bbox,
                }
            )

        return detections

    def _predict_mae_chunk(
        self,
        *,
        chunk_frames: list[np.ndarray],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> dict[str, Any]:
        """
        Классификация одного чанка кадров через VideoMAE.
        """
        assert self.mae.model is not None
        assert self.mae.processor is not None
        id2label: dict[int, str] = self.mae.model.config.id2label or {}

        inputs = self.mae.processor(chunk_frames, return_tensors="pt").to(
            self.processing.device
        )
        outputs = self.mae.model(**inputs)
        logits = outputs.logits[0]
        probs = torch.nn.functional.softmax(logits, dim=0)

        pred_idx = int(logits.argmax().item())
        conf = float(probs[pred_idx].item())
        answer = id2label.get(pred_idx, str(pred_idx))

        start_time = start_frame / fps if fps > 0 else 0.0
        end_time = (end_frame + 1) / fps if fps > 0 else 0.0

        return {
            "time": [start_time, end_time],
            "answer": answer,
            "confident": conf,
        }

    def _update_yolo_stream_state(
        self, frame_resized: np.ndarray, state: VideoStreamState, fps: float
    ) -> None:
        """
        Запускает YOLO на кадре по stride и обновляет состояние.
        """
        if state.frame_idx % self.processing.yolo_stride != 0:
            return

        detections = self._run_yolo_on_frame(frame_resized, state.frame_idx, fps)
        for detection in detections:
            timestep = float(detection["timesteps"])
            state.bbox_by_time[timestep].append(detection["bbox"])
            state.detected_class_ids.add(int(detection["class_id"]))

    def _update_mae_stream_state(
        self, frame_resized: np.ndarray, state: VideoStreamState, fps: float
    ) -> None:
        """
        Копит кадры в чанк и при заполнении запускает MAE.
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
        Обрабатывает один кадр для YOLO и MAE.
        """

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.processing.frame_size)

        self._update_yolo_stream_state(frame_resized, state, fps)
        self._update_mae_stream_state(frame_resized, state, fps)
        state.frame_idx += 1

    def _flush_partial_chunk(self, state: VideoStreamState, fps: float) -> None:
        """
        Дополняет неполный чанк последним кадром и запускает MAE.
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

    def _collect_stream_results(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        video_path: str,
    ) -> VideoStreamState:
        """
        Собирает MAE и YOLO результаты из видеопотока.
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
        return state

    def _build_video_info(
        self,
        *,
        total_frames: int,
        fps: float,
        duration: float,
        frames_read: int,
    ) -> dict[str, Any]:
        """
        Формирует метаданные обработки видео.
        """
        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "frames_read": frames_read,
        }

    def _resolve_detected_objects(self, detected_class_ids: set[int]) -> list[str]:
        """
        Преобразует ID детектированных классов в названия.
        """
        assert self.yolo.yolo is not None
        names = self.yolo.yolo.names
        return [names[class_id] for class_id in detected_class_ids]

    async def _process_video_stream(
        self, video_path: str
    ) -> tuple[
        list[dict[str, Any]],
        dict[float, list[list[float]]],
        list[str],
        dict[str, Any],
    ]:
        """
        Потоковая обработка видео без загрузки всех кадров в память.
        Одним проходом собирает MAE и YOLO результаты.
        """
        assert self.mae.model is not None
        assert self.yolo.yolo is not None

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

    # не используется
    async def _chunk_frames(self, frames: np.ndarray) -> np.ndarray:
        """Разделение кадров на чанки"""
        t = len(frames)
        if t == 0:
            raise ValueError("Нет кадров для процессирования: входящий массив пустой")

        padding_needed = (-t) % self.processing.chunk_size

        if padding_needed > 0:
            last_frame = frames[-1]
            padding = np.tile(last_frame, (padding_needed, 1, 1, 1))
            frames = np.vstack((frames, padding))

        num_chunks = len(frames) // self.processing.chunk_size
        if num_chunks == 0:
            raise ValueError(
                "Чанки не создались: видео очень короткое даже после паддинга"
            )

        frame_h, frame_w = self.processing.frame_size
        return frames.reshape(
            num_chunks, self.processing.chunk_size, frame_h, frame_w, 3
        )

    def _log_processing(
        self, video_name: str, video_info: dict[str, Any], processing_time: float
    ) -> None:
        """
        Финальный лог обработки.
        """
        logger.info(
            "РЕЗУЛЬТАТЫ АНАЛИЗА ВИДЕО: video=%s"
            "duration=%.1fs processing=%.2fs frames=%d/%d",
            video_name,
            float(video_info["duration"]),
            processing_time,
            int(video_info["frames_read"]),
            int(video_info["total_frames"]),
        )

    async def analyze(self, model: KlinModel) -> KlinResultDto:
        """
        Сам процессор
        Сначала выполняется быстрый X3D check.
        Если драка НЕ обнаружена (False) — сразу возвращаем x3d="False",
        все остальные поля пустые (mae, yolo и т.д.).
        Если драка обнаружена (True) — запускаем полную обработку YOLO + VideoMAE
        и возвращаем полный результат с x3d="True".
        """
        self.ensure_x3d_loaded()  # X3D загружается всегда (нужен для быстрой проверки)

        start_ts = time.time()
        video_name = os.path.basename(model.video_path)

        # === Быстрая проверка X3D ===
        result_dict = self._quick_x3d_check(model.video_path)
        is_fight = list(result_dict.keys())[0]

        if not is_fight:
            return KlinResultDto(
                x3d=msgspec.json.encode(result_dict).decode("utf-8"),
                mae="[]",
                yolo="{}",
                all_classes=[],
                objects=[],
            )

        # === Если драка обнаружена — полная обработка ===

        self._ensure_models_loaded()  # теперь загружаем MAE и YOLO (X3D уже загружен)

        try:
            (
                mae_results,
                yolo_bbox,
                detected_objects,
                video_info,
            ) = await self._process_video_stream(
                model.video_path,
            )
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

                await asyncio.sleep(2**attempt)
