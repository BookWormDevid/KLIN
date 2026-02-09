import logging
import torch
import cv2
import os
import numpy as np
import time
import tempfile
from pathlib import Path
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)
from app.application.interfaces import IMAEInference
from app.models import MAEModel

logger = logging.getLogger(__name__)

BASE_DIR_MAE = Path(__file__).parent.parent.parent.parent.parent
mae_dir = BASE_DIR_MAE / "videomae_results" / "videomae-ufc-crime"

class MAEProcessor(IMAEInference):
    def __init__(self, chunk_size: int = 16, frame_size: tuple[int, int] = (224, 224), mae_dir: str | None = None) -> None:
        if not mae_dir:
            mae_dir = self._find_model_path()
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VideoMAEImageProcessor.from_pretrained(
            mae_dir, local_files_only=True
        )
        model: VideoMAEForVideoClassification = (
            VideoMAEForVideoClassification.from_pretrained(
                mae_dir, local_files_only=True
            )
        )
        self.model = model.to(self.device)  # type: ignore
        self.model.eval()

    def _find_model_path(self) -> str:
        """Автоматически найти путь к модели"""

        BASE_DIR_MAE = Path(__file__).parent.parent
        print(BASE_DIR_MAE)
        mae_dir = BASE_DIR_MAE / "videomae_results" / "videomae-ufc-crime"

        return str(mae_dir)

    def save_video_bytes(self, video_bytes: bytes, suffix: str = ".mp4") -> str:
        if not video_bytes:
            raise ValueError("video_bytes пустой")

        try:
            local_temp_dir = Path("tmp").resolve()
            local_temp_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                    dir=local_temp_dir
            ) as tmp_file:
                tmp_file.write(video_bytes)
                tmp_path = tmp_file.name

                return str(tmp_path)
        except Exception as e:
            raise RuntimeError(f"Не удалось сохранить видео во временный файл: {e}") from e

    def _chunk_frames(self, frames: np.ndarray) -> np.ndarray:
        """Разделение кадров на чанки"""
        t = len(frames)
        padding_needed = (-t) % self.chunk_size

        if padding_needed > 0:
            padding = np.zeros((padding_needed, *self.frame_size, 3), dtype=np.uint8)
            frames = np.vstack((frames, padding))

        num_chunks = len(frames) // self.chunk_size
        return frames.reshape(num_chunks, self.chunk_size, *self.frame_size, 3)

    def _read_video_frames(self, video_path: str) -> tuple:
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

    def predict_video(self, video_path: str, batch_size: int = 2) -> dict:
        """Предсказание для одного видео"""
        start_time = time.time()
        video_name = os.path.basename(video_path)

        try:
            print(f"[API] Начало обработки видео: {video_name}")
            frames, video_info = self._read_video_frames(video_path)
            print(f"[API] Прочитано кадров: {len(frames)} из {video_info['total_frames']}")

            chunks = self._chunk_frames(frames)
            print(f"[API] Создано чанков: {len(chunks)}")

            all_logits = []
            with torch.no_grad():
                # Обрабатываем по одному чанку (16 кадров)
                for chunk in chunks:
                    # chunk.shape == (16, 224, 224, 3)
                    inputs = self.processor(
                        list(chunk),  # список из 16 numpy-массивов
                        return_tensors="pt"
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
            if predicted_class == "nonviolent":
                predicted_class = "non_violent"

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

            return {
                "video_name": video_name,
                "video_path": video_path,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "total_frames": video_info["total_frames"],
                "video_duration": video_info["duration"],
                "video_fps": video_info["fps"],
                "processing_time": processing_time,
            }

        except Exception as e:
            print(f"[API] Ошибка обработки видео {video_name}: {e}")
            return {
                "video_name": video_name,
                "video_path": video_path,
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def analyze(self, model: MAEModel, video_bytes: bytes) -> str:
        """
            Принимает видео в виде bytes, сохраняет во временный файл,
            передаёт в predict_video и возвращает результат.
            """
        video_path = None
        try:
            # 1️⃣ Сохраняем видео во временный файл
            video_path = self.save_video_bytes(video_bytes)

            # 2️⃣ Запускаем инференс
            result = self.predict_video(video_path)

            return result

        except Exception as e:
            logger.exception("Ошибка при предсказании видео из байт")
            return {
                "video_name": "unknown",
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "error": str(e),
            }

        finally:
            # 3️⃣ Удаляем временный файл, если он существует
            if video_path and Path(video_path).exists():
                try:
                    Path(video_path).unlink()
                except Exception:
                    pass
