# pylint: disable=no-member
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
from typing import Any

import aiohttp
import async_timeout
import cv2
import msgspec
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from ultralytics import YOLO

from app.application.dto import KlinResultDto
from app.application.interfaces import IKlinCallbackSender, IKlinInference
from app.config import app_settings
from app.models.klin import KlinModel


logger = logging.getLogger(__name__)

BASE_DIR_MAE = Path(__file__).parent.parent.parent.parent.parent
MAE_DIR = BASE_DIR_MAE / app_settings.videomae_path


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

        model = VideoMAEForVideoClassification.from_pretrained(
            self.mae.mae_model, local_files_only=True
        )

        self.mae.model = model.to(self.processing.device)  # type: ignore
        self.mae.model.eval()

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
        base_yolo_path = Path(__file__).parent.parent.parent.parent.parent
        self.yolo.yolo_path = str(base_yolo_path / "models" / "yolov8x.pt")
        return self.yolo.yolo_path

    def ensure_yolo_loaded(self) -> None:
        """
        Загружает модель и проверяет cuda к модели
        """
        if self.yolo.yolo is not None:
            return

        weights_path = self.find_yolo_path()
        self.yolo.yolo = YOLO(weights_path)

        self.yolo.yolo.to(self.processing.device)

    def _run_yolo_on_frame(
        self, frame: np.ndarray, frame_idx: int, fps: float
    ) -> list[dict[str, Any]]:
        """
        Запуск YOLO на одном кадре.
        """
        assert self.yolo.yolo is not None

        timesteps = frame_idx / fps if fps > 0 else 0.0
        use_half = self.processing.device.type != "cpu"

        preds = self.yolo.yolo(
            frame,
            conf=self.processing.yolo_conf,
            iou=self.processing.yolo_iou,
            classes=self.processing.yolo_classes,
            half=use_half,
            show=False,
            save=False,
        )

        detections: list[dict[str, Any]] = []
        for result in preds:
            if result.boxes is None:
                continue

            for box in result.boxes:
                detections.append(
                    {
                        "class_id": int(box.cls.item()),
                        "timesteps": float(timesteps),
                        "bbox": box.xyxyn[0].tolist(),
                    }
                )

        return detections

    async def run_yolo(self, frames: np.ndarray, fps: float) -> list[dict[str, Any]]:
        """
        Запуск yolo
        Возвращает список с
        class_id - что нашёл
        timestaps - нашёл на какой временном промежутке
        bbox - boundingbox(квадрат, где обнаружил объект)
        """
        assert self.yolo.yolo is not None

        results: list[dict[str, Any]] = []
        step = self.processing.yolo_stride

        for i in range(0, len(frames), step):
            frame = frames[i]
            detections = self._run_yolo_on_frame(frame, i, fps)
            results.extend(detections)

        return results

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

        mae_results: list[dict[str, Any]] = []
        bbox_by_time: dict[float, list[list[float]]] = defaultdict(list)
        detected_class_ids: set[int] = set()

        chunk_buffer: list[np.ndarray] = []
        chunk_start_frame = 0
        frame_idx = 0
        try:
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, self.processing.frame_size)

                    if frame_idx % self.processing.yolo_stride == 0:
                        detections = self._run_yolo_on_frame(
                            frame_resized,
                            frame_idx,
                            fps,
                        )
                        for detection in detections:
                            timesteps = float(detection["timesteps"])
                            bbox_by_time[timesteps].append(detection["bbox"])
                            detected_class_ids.add(int(detection["class_id"]))

                    chunk_buffer.append(frame_resized)
                    if len(chunk_buffer) == self.processing.chunk_size:
                        mae_results.append(
                            self._predict_mae_chunk(
                                chunk_frames=chunk_buffer,
                                start_frame=chunk_start_frame,
                                end_frame=frame_idx,
                                fps=fps,
                            )
                        )
                        chunk_buffer = []
                        chunk_start_frame = frame_idx + 1

                    frame_idx += 1

                if frame_idx == 0:
                    raise ValueError(f"Кадры не прочитаны: {video_path}")

                if chunk_buffer:
                    pad_count = self.processing.chunk_size - len(chunk_buffer)
                    if pad_count > 0:
                        chunk_buffer.extend([chunk_buffer[-1]] * pad_count)

                    mae_results.append(
                        self._predict_mae_chunk(
                            chunk_frames=chunk_buffer,
                            start_frame=chunk_start_frame,
                            end_frame=frame_idx - 1,
                            fps=fps,
                        )
                    )
        finally:
            cap.release()

        names = self.yolo.yolo.names
        objects = [names[class_id] for class_id in detected_class_ids]

        video_info = {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "frames_read": frame_idx,
        }
        return mae_results, dict(bbox_by_time), objects, video_info

    async def _read_video_frames(
        self, video_path: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Чтение видео и возврат кадров + информация о видео"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")

        frames: list[np.ndarray] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        duration = (total_frames / fps) if fps > 0 else 0.0

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.processing.frame_size)
            frames.append(frame_resized)

        cap.release()

        if not frames:
            raise ValueError(f"Кадры не прочитаны: {video_path}")

        video_info = {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "frames_read": len(frames),
        }

        return np.array(frames, dtype=np.uint8), video_info

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

    async def _process_mae(
        self, *, chunks: np.ndarray, fps: float, actual_frames: int
    ) -> list[dict[str, Any]]:
        """
        Классификация чанков через VideoMAE.
        """
        assert self.mae.model is not None
        assert self.mae.processor is not None

        results: list[dict[str, Any]] = []
        id2label: dict[int, str] = self.mae.model.config.id2label or {}

        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                start_frame = i * self.processing.chunk_size
                if start_frame >= actual_frames:
                    break

                end_frame = min((i + 1) * self.processing.chunk_size, actual_frames) - 1
                start_time = start_frame / fps if fps > 0 else 0.0
                end_time = (end_frame + 1) / fps if fps > 0 else 0.0

                inputs = self.mae.processor(list(chunk), return_tensors="pt").to(
                    self.processing.device
                )
                outputs = self.mae.model(**inputs)
                logits = outputs.logits[0]
                probs = torch.nn.functional.softmax(logits, dim=0)

                pred_idx = int(logits.argmax().item())
                conf = float(probs[pred_idx].item())
                answer = id2label.get(pred_idx, str(pred_idx))

                results.append(
                    {
                        "time": [start_time, end_time],
                        "answer": answer,
                        "confident": conf,
                    }
                )

        return results

    async def _process_yolo(
        self, *, frames: np.ndarray, fps: float
    ) -> tuple[dict[float, list[list[float]]], list[str]]:
        """
        Детекция объектов через YOLO + агрегация bbox по таймстемпам.
        """
        yolo_results = await self.run_yolo(frames, fps)

        bbox_by_time: dict[float, list[list[float]]] = defaultdict(list)
        detected_class_ids: set[int] = set()

        for detection in yolo_results:
            timesteps = float(detection["timesteps"])
            bbox_by_time[timesteps].append(detection["bbox"])
            detected_class_ids.add(int(detection["class_id"]))

        assert self.yolo.yolo is not None
        names = self.yolo.yolo.names
        objects = [names[class_id] for class_id in detected_class_ids]
        return dict(bbox_by_time), objects

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
        Проверяет, что все модели загружены
        Читает кадры с видео
        Делит видео на чанки для обработки
        Обработка videomae + yolo
        Возвращает:
        список mae [{time: [start_time_chunk, end_time_chunk], answer, confident}]
        start_time_chunk - старт времени, когда найден класс
        end_time_chunk - конец времени, когда найден класс
        answer - класс
        confident - уверенность в классе
        словарь yolo
        timestaps - время когда обнаружены классы
        bbox - boundingbox
        all_classes - список все классы что выдало videomae
        objects - список все классы что выдало yolo
        """
        self._ensure_models_loaded()

        start_ts = time.time()
        video_name = os.path.basename(model.video_path)

        try:
            logger.info("[API] Начало обработки видео: %s", video_name)

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
