# pylint: disable=no-member
"""
Процессор для взаимодействия с приложением litestar
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
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
from app.models import KlinModel

logger = logging.getLogger(__name__)

BASE_DIR_MAE = Path(__file__).parent.parent.parent.parent.parent
mae_dir = BASE_DIR_MAE / "videomae_results" / "videomae-ufc-crime"


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

    chunk_size = 16
    frame_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceProcessor(IKlinInference):
    """
    Процессор. Содержит методы подключения моделей, обработки видео с помощью videomae и yolo.
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

        base_dir_mae = Path(__file__).parent.parent.parent.parent.parent
        print(base_dir_mae)
        self.mae.mae_model = str(
            base_dir_mae / "videomae_results" / "videomae-ufc-crime"
        )
        return self.mae.mae_model

    def find_yolo_path(self) -> str:
        """
        Автоматически найти путь к модели yolo
        Ищет по заданному пути
        """
        base_yolo_path = Path(__file__).parent.parent.parent.parent.parent
        print(base_yolo_path)
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

    async def run_yolo(self, frames: np.ndarray, fps: float) -> list[dict]:
        """
        Запуск yolo
        Возвращает список с
        class_id - что нашёл
        timestaps - нашёл на какой временном промежутке
        bbox - boundingbox(квадрат, где обнаружил объект)
        """
        assert self.yolo.yolo is not None

        results = []
        n = 2

        for i in range(0, len(frames), n):  # каждый n-й кадр
            frame = frames[i]
            timesteps = i / fps  # время в секундах

            preds = self.yolo.yolo(
                frame,
                conf=0.6,  # уверенность с которой будет показывать. Меньше conf не будет показывать
                iou=0.45,  # подавить дубликаты
                classes=[0],  # классы объектов
                half=True,  # fp16 для скорости
                show=False,  # показывает обрабатываемый ролик
                save=False,
            )
            for r in preds:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    xyxyn = box.xyxyn[0].tolist()
                    results.append(
                        {
                            "class_id": int(box.cls.item()),
                            "timesteps": float(timesteps),
                            "bbox": xyxyn,
                        }
                    )

        return results

    async def _read_video_frames(self, video_path: str) -> tuple:
        """Чтение видео и возврат кадров + информация о видео"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.processing.frame_size)
            frames.append(frame_resized)

        cap.release()

        if len(frames) == 0:
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

        return frames.reshape(
            num_chunks, self.processing.chunk_size, *self.processing.frame_size, 3
        )

    async def analyze(self, model: KlinModel) -> KlinResultDto:
        """
        Сам процессор
        Проверяет, что все модели загружены
        Читает кадры с видео
        Делит видео на чанки для обработки
        Обработка videomae
        Обработка yolo
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
        # Загрузка моделей (один раз)
        if self.mae.model is None:
            self.ensure_mae_model_loaded()
        if self.yolo.yolo is None:
            self.ensure_yolo_loaded()
        assert self.mae.model is not None
        assert self.mae.processor is not None
        assert self.yolo.yolo is not None
        start_time = time.time()
        video_name = os.path.basename(model.video_path)
        try:
            print(f"[API] Начало обработки видео: {video_name}")
            # ---- Чтение видео ----
            frames, video_info = await self._read_video_frames(model.video_path)
            fps = video_info["fps"]
            print(
                f"[API] Прочитано кадров: {len(frames)} из {video_info['total_frames']}"
            )
            # ---- Подготовка чанков для MAE ----
            chunks = await self._chunk_frames(frames)
            print(f"[API] Создано чанков: {len(chunks)}")
            # =====================================================
            # MAE + YOLO
            # =====================================================
            with torch.no_grad():
                # ---------- MAE ----------
                chunk_results = []
                actual_frames = video_info["frames_read"]
                id2label: dict = self.mae.model.config.id2label or {}
                for i, chunk in enumerate(chunks):
                    start_frame = i * self.processing.chunk_size
                    end_frame = (
                        min((i + 1) * self.processing.chunk_size, actual_frames) - 1
                    )
                    if start_frame >= actual_frames:
                        break
                    start_time_chunk = start_frame / fps if fps > 0 else 0
                    end_time_chunk = (end_frame + 1) / fps if fps > 0 else 0
                    inputs = self.mae.processor(list(chunk), return_tensors="pt").to(
                        self.processing.device
                    )
                    outputs = self.mae.model(**inputs)
                    logits = outputs.logits[0]
                    probabilities = torch.nn.functional.softmax(logits, dim=0)
                    predicted_idx = int(logits.argmax().item())
                    confident = float(probabilities[predicted_idx].item())
                    answer = id2label.get(predicted_idx, str(predicted_idx))
                    chunk_results.append(
                        {
                            "time": [start_time_chunk, end_time_chunk],
                            "answer": answer,
                            "confident": confident,
                        }
                    )

                if chunk_results:
                    all_classes = list(set(d["answer"] for d in chunk_results))
                else:
                    all_classes = []

                # ---------- YOLO ----------
                yolo_results = await self.run_yolo(frames, fps)
                # =====================================================
                # Обработка YOLO
                # =====================================================
                detected_classes = list({r["class_id"] for r in yolo_results})
                detected_objects = []
                if detected_classes:
                    names = self.yolo.yolo.names
                    detected_objects = [names[c] for c in detected_classes]
                bbox_by_time = defaultdict(list)

                for detection in yolo_results:
                    timesteps = detection["timesteps"]
                    bbox = detection["bbox"]
                    bbox_by_time[timesteps].append(bbox)

                bbox_dict = dict(bbox_by_time)

            processing_time = time.time() - start_time
            # ---- Лог ----
            print("\n" + "=" * 60)
            print("РЕЗУЛЬТАТЫ АНАЛИЗА ВИДЕО")
            print("=" * 60)
            print(f"Видео: {video_name}")
            print(f"Длительность: {video_info['duration']:.1f} сек")
            print(f"Время обработки: {processing_time:.2f} сек")
            print("=" * 60 + "\n")

            # json_result = {
            #     "mae": chunk_results,
            #     "yolo": bbox_dict,
            #     "all_classes": all_classes,
            #     "objects": detected_objects,
            #     "bbox_by_time": bbox_dict,
            # }

            return KlinResultDto(
                mae=msgspec.json.encode(chunk_results).decode("utf-8"),
                yolo=msgspec.json.encode(bbox_dict).decode("utf-8"),
                all_classes=all_classes,
                objects=detected_objects,
            )
        except Exception as e:
            logger.error(f"Ошибка обработки {e}")
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
