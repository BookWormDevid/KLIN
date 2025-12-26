import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yt_dlp
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

BASE_DIR = Path(__file__).parent.parent


class VideoClassifier:
    def __init__(self, model_path: str = "", chunk_size: int = 16, frame_size: tuple = (224, 224)):
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Определяем путь к модели
        if not model_path:
            model_path = self._find_model_path()

        print(f"[API] Загрузка модели из: {model_path}")

        # Загрузка модели и процессора
        self.processor = VideoMAEImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

        print(f"[API] Модель загружена! Доступные классы: {self.model.config.id2label}")

    def _find_model_path(self):
        """Автоматически найти путь к модели"""
        possible_paths = [
            BASE_DIR / "videomae_results" / "checkpoint14172",
            Path("videomae_results") / "checkpoint14172",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(f"Модель не найдена по пути: {possible_paths[0]}")

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
        max_frames = min(total_frames, 100)
        for i in range(max_frames):
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
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'frames_read': len(frames)
        }

        return np.array(frames, dtype=np.uint8), video_info

    def _download_video_from_url(self, url: str) -> str:
        """Скачать видео по URL"""
        print(f"[API] Загрузка видео по URL: {url}")
        
        # Создаем временный файл
        temp_dir = tempfile.gettempdir()
        temp_filename = f"video_{int(time.time())}.mp4"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        
        # Настройки yt-dlp
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': temp_filepath,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise ValueError(f"Не удалось загрузить видео по URL: {url}")
                
                # Проверяем, что файл скачался
                if os.path.exists(temp_filepath):
                    file_size = os.path.getsize(temp_filepath) / (1024 * 1024)
                    print(f"[API] Видео скачано: {temp_filepath} ({file_size:.1f} MB)")
                    return temp_filepath
                else:
                    raise ValueError("Файл не скачан")
                    
        except Exception:
            # Пробуем альтернативный подход для прямых ссылок на видео
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Некорректный URL: {url}")
            
            print("[API] Попытка прямой загрузки...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, temp_filepath)
                if os.path.exists(temp_filepath):
                    file_size = os.path.getsize(temp_filepath) / (1024 * 1024)
                    print(f"[API] Видео скачано напрямую: {temp_filepath} ({file_size:.1f} MB)")
                    return temp_filepath
                else:
                    raise ValueError("Не удалось скачать видео")
            except Exception as e2:
                raise ValueError(f"Ошибка загрузки видео: {str(e2)}")

    def _chunk_frames(self, frames: np.ndarray) -> np.ndarray:
        """Разделение кадров на чанки"""
        t = len(frames)
        padding_needed = (-t) % self.chunk_size

        if padding_needed > 0:
            padding = np.zeros((padding_needed, *self.frame_size, 3), dtype=np.uint8)
            frames = np.concatenate([frames, padding], axis=0)

        num_chunks = len(frames) // self.chunk_size
        return frames.reshape(num_chunks, self.chunk_size, *self.frame_size, 3)

    def predict_video(self, video_path: str, batch_size: int = 2) -> dict:
        """Предсказание для одного видео"""
        start_time = time.time()
        video_name = os.path.basename(video_path)
        
        try:
            # Чтение и обработка видео
            print(f"[API] Начало обработки видео: {video_name}")
            frames, video_info = self._read_video_frames(video_path)
            print(f"[API] Прочитано кадров: {len(frames)} из {video_info['total_frames']}")
            
            chunks = self._chunk_frames(frames)
            print(f"[API] Создано чанков: {len(chunks)}")

            # Пакетная обработка чанков
            all_predictions = []
            with torch.no_grad():
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_frames = [list(chunk) for chunk in batch_chunks]

                    inputs = self.processor(batch_frames, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    all_predictions.append(outputs.logits.cpu())

            # Агрегация результатов
            if len(all_predictions) > 0:
                final_logits = torch.mean(torch.cat(all_predictions), dim=0)
                probabilities = torch.nn.functional.softmax(final_logits, dim=0)
                predicted_idx = final_logits.argmax().item()
                confidence = probabilities[predicted_idx].item()
            else:
                predicted_idx = 0
                confidence = 0.5

            # Получение имени класса
            predicted_class = self.model.config.id2label.get(predicted_idx, "unknown")
            # Преобразование nonviolent -> non_violent для совместимости
            if predicted_class == "nonviolent":
                predicted_class = "non_violent"

            processing_time = time.time() - start_time

            # Вывод в терминал
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
                'video_name': video_name,
                'video_path': video_path,
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'total_frames': video_info['total_frames'],
                'video_duration': video_info['duration'],
                'video_fps': video_info['fps'],
                'processing_time': processing_time
            }

        except Exception as e:
            print(f"[API] Ошибка обработки видео {video_name}: {e}")
            return {
                'video_name': video_name,
                'video_path': video_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def predict_video_from_url(self, url: str) -> dict:
        """Предсказание для видео по URL"""
        temp_filepath = None
        try:
            print(f"[API] Загрузка видео по URL: {url}")
            
            # Скачиваем видео по URL
            temp_filepath = self._download_video_from_url(url)
            
            # Анализируем скачанное видео
            result = self.predict_video(temp_filepath)
            
            # Очищаем временный файл
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                
            # Добавляем информацию об URL
            result['url'] = url
            return result
            
        except Exception as e:
            # Очищаем временный файл в случае ошибки
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
                    
            print(f"[API] Ошибка обработки URL {url}: {e}")
            return {
                'video_name': 'url_video',
                'url': url,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }