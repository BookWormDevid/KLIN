import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
import async_timeout
import cv2
import msgspec
import numpy as np
import torch
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

from app.application.dto import MAEResultDto
from app.application.interfaces import IMAECallbackSender, IMAEInference
from app.models import MAEModel

logger = logging.getLogger(__name__)

BASE_DIR_MAE = Path(__file__).parent.parent.parent.parent.parent
mae_dir = BASE_DIR_MAE / "videomae_results" / "videomae-ufc-crime"


class MAEProcessor(IMAEInference):
    def __init__(self) -> None:
        self.model: VideoMAEForVideoClassification | None = None
        self.processor: VideoMAEImageProcessor | None = None
        self.mae_model: str | None = None
        self.chunk_size = 16
        self.frame_size = (224, 224)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        self.mae_model = self.find_model_path()
        self.processor = VideoMAEImageProcessor.from_pretrained(
            self.mae_model, local_files_only=True
        )

        model = VideoMAEForVideoClassification.from_pretrained(
            self.mae_model, local_files_only=True
        )

        self.model = model.to(self.device)  # type: ignore
        self.model.eval()

    def find_model_path(self) -> str:
        """Автоматически найти путь к модели"""

        base_dir_mae = Path(__file__).parent.parent.parent.parent.parent
        print(base_dir_mae)
        self.mae_model = str(base_dir_mae / "videomae_results" / "videomae-ufc-crime")
        return self.mae_model

    # def save_video_bytes(self, video_bytes: bytes, suffix: str = ".mp4") -> str:
    #     if not video_bytes:
    #         raise ValueError("video_bytes пустой")
    #
    #     try:
    #         local_temp_dir = Path("tmp").resolve()
    #         local_temp_dir.mkdir(parents=True, exist_ok=True)
    #
    #         with tempfile.NamedTemporaryFile(
    #                 delete=False,
    #                 suffix=suffix,
    #                 dir=local_temp_dir
    #         ) as tmp_file:
    #
    #             tmp_file.write(video_bytes)
    #             tmp_path = tmp_file.name
    #
    #             return str(tmp_path)
    #     except Exception as e:
    #         raise RuntimeError(f"Не удалось сохранить видео во временный файл: {e}") from e

    async def _read_video_frames(self, video_path: str) -> tuple:
        """Чтение видео и возврат кадров + информация о видео"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Читаем кадры (ограничиваем для скорости)
        max_frames: int = min(total_frames, 100)
        for _i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            frames.append(frame_resized)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

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
        padding_needed = (-t) % self.chunk_size

        if padding_needed > 0:
            padding = np.zeros((padding_needed, *self.frame_size, 3), dtype=np.uint8)
            frames = np.vstack((frames, padding))

        num_chunks = len(frames) // self.chunk_size
        if num_chunks is None:
            raise ValueError("Chunk size is empty")
        return frames.reshape(num_chunks, self.chunk_size, *self.frame_size, 3)

    async def analyze(self, mae_request: MAEModel) -> MAEResultDto:
        if self.processor is None or self.model is None:
            self.ensure_model_loaded()
        assert self.processor is not None
        assert self.model is not None

        start_time = time.time()
        video_name = os.path.basename(mae_request.video_path)
        try:
            print(f"[API] Начало обработки видео: {video_name}")
            frames, video_info = await self._read_video_frames(mae_request.video_path)
            print(
                f"[API] Прочитано кадров: {len(frames)} из {video_info['total_frames']}"
            )

            chunks = await self._chunk_frames(frames)
            print(f"[API] Создано чанков: {len(chunks)}")
            all_logits = []
            with torch.no_grad():
                # Обрабатываем по одному чанку (16 кадров)
                for chunk in chunks:
                    # chunk.shape == (16, 224, 224, 3)
                    inputs = self.processor(
                        list(chunk),  # список из 16 numpy-массивов
                        return_tensors="pt",
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    all_logits.append(outputs.logits)  # (1, num_classes)

            # Агрегация результатов
            if all_logits:
                # Среднее по всем чанкам
                final_logits = torch.mean(torch.cat(all_logits, dim=0), dim=0)
                probabilities = torch.nn.functional.softmax(final_logits, dim=0)
                predicted_idx = int(final_logits.argmax().item())
                confidence = probabilities[predicted_idx].item()
            else:
                predicted_idx = 0
                confidence = 0.5

            # Получение имени класса
            id2label: dict = self.model.config.id2label or {}
            predicted_class = id2label.get(predicted_idx, str(predicted_idx))

            processing_time = time.time() - start_time

            # Вывод результатов
            print("\n" + "=" * 60)
            print("РЕЗУЛЬТАТЫ АНАЛИЗА ВИДЕО")
            print("=" * 60)
            print(f"Видео: {video_name}")
            print(f"Результат: {predicted_class}")
            print(f"Уверенность: {confidence:.2%}")
            print(f"Длительность: {video_info['duration']:.1f} сек")
            print(f"Всего кадров: {video_info['total_frames']}")
            print(f"FPS: {video_info['fps']:.2f}")
            print(f"Время обработки: {processing_time:.1f} сек")
            print("=" * 60 + "\n")

            return MAEResultDto(result=predicted_class)

        except Exception as e:
            logger.error(f"Ошибка обработки {e}")
            raise


class MAECallbackSender(IMAECallbackSender):
    async def post_consumer(self, model: MAEModel) -> None:
        if not model.response_url:
            return

        payload: dict[str, Any] = {
            "mae_id": str(model.id),
            "result": model.result,
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
                        "transcription_id=%s response_url=%s error=%s",
                        max_attempts,
                        model.id,
                        model.response_url,
                        exc,
                    )
                    return

                await asyncio.sleep(2**attempt)
