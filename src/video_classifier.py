import os
import torch
import cv2
import numpy as np
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from tqdm import tqdm
import pandas as pd
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent

class VideoFolderClassifier:
    def __init__(self, model_path: str = None, chunk_size: int = 16, frame_size: tuple = (224, 224)):
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Определяем путь к модели
        if model_path is None:
            # Автоматически ищем модель в текущей директории или родительских
            model_path = self._find_model_path()

        print(f"🔄 Загрузка модели из: {model_path}")

        # Загрузка модели и процессора
        self.processor = VideoMAEImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

        print(f"✅ Модель загружена! Классы: {self.model.config.id2label}")

    def _find_model_path(self):
        """Автоматически найти путь к модели"""
        possible_paths = [
            os.path.join(BASE_DIR, "models", "videomae-large")
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                return path

        # Если не нашли, запросим у пользователя
        print("❌ Модель не найдена автоматически.")
        model_path = input("📁 Введите полный путь к папке с моделью: ")
        if os.path.exists(model_path):
            return model_path
        else:
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    def _read_video_frames(self, video_path: str) -> np.ndarray:
        """Read video and return frames as numpy array"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Читаем каждый кадр
        for _ in range(min(total_frames, 1000)):  # ограничим максимальное количество кадров
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            frames.append(frame_resized)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

        return np.array(frames, dtype=np.uint8)

    def _chunk_frames(self, frames: np.ndarray) -> np.ndarray:
        """Split frames into chunks with padding if needed"""
        t = len(frames)
        padding_needed = (-t) % self.chunk_size

        if padding_needed > 0:
            padding = np.zeros((padding_needed, *self.frame_size, 3), dtype=np.uint8)
            frames = np.concatenate([frames, padding], axis=0)

        num_chunks = len(frames) // self.chunk_size
        return frames.reshape(num_chunks, self.chunk_size, *self.frame_size, 3)

    def predict_video(self, video_path: str, batch_size: int = 4) -> dict:
        """Predict single video"""
        try:
            # Process video
            frames = self._read_video_frames(video_path)
            chunks = self._chunk_frames(frames)

            print(f"📹 {os.path.basename(video_path)}: {len(frames)} frames -> {len(chunks)} chunks")

            # Batch process chunks
            all_predictions = []
            with torch.no_grad():
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_frames = [list(chunk) for chunk in batch_chunks]

                    inputs = self.processor(batch_frames, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    all_predictions.append(outputs.logits.cpu())

            # Aggregate results
            final_logits = torch.mean(torch.cat(all_predictions), dim=0)
            probabilities = torch.nn.functional.softmax(final_logits, dim=0)
            predicted_idx = final_logits.argmax().item()
            confidence = probabilities[predicted_idx].item()

            return {
                'video_name': os.path.basename(video_path),
                'video_path': video_path,
                'predicted_class': self.model.config.id2label[predicted_idx],
                'confidence': confidence,
                'num_frames': len(frames),
                'num_chunks': len(chunks)
            }

        except Exception as e:
            print(f"❌ Ошибка обработки {os.path.basename(video_path)}: {e}")
            return {
                'video_name': os.path.basename(video_path),
                'video_path': video_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }

    def predict_folder(self, folder_path: str, output_file: str = None, batch_size: int = 4) -> pd.DataFrame:
        """Predict all videos in folder"""

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")

        # Find all video files
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        video_files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))

        print(f"🎯 Найдено видео файлов: {len(video_files)}")

        if len(video_files) == 0:
            print("❌ Видео файлы не найдены!")
            return pd.DataFrame()

        # Process all videos
        results = []
        for video_path in tqdm(video_files, desc="Обработка видео"):
            result = self.predict_video(video_path, batch_size)
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save results if output file specified
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"💾 Результаты сохранены в: {output_file}")

        # Print summary
        self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print processing summary"""
        if len(df) == 0:
            return

        successful = df[df['predicted_class'] != 'ERROR']
        errors = df[df['predicted_class'] == 'ERROR']

        print(f"\n📊 СВОДКА ОБРАБОТКИ:")
        print(f"   Всего видео: {len(df)}")
        print(f"   Успешно обработано: {len(successful)}")
        print(f"   Ошибок: {len(errors)}")

        if len(successful) > 0:
            print(f"   Средняя уверенность: {successful['confidence'].mean():.4f}")

            # Class distribution
            class_counts = successful['predicted_class'].value_counts()
            print(f"   Распределение классов:")
            for class_name, count in class_counts.items():
                percentage = (count / len(successful)) * 100
                print(f"     {class_name}: {count} видео ({percentage:.1f}%)")

        if len(errors) > 0:
            print(f"\n❌ Ошибки обработки:")
            for _, error_row in errors.iterrows():
                print(f"   {error_row['video_name']}: {error_row.get('error', 'Unknown error')}")


# Исправленная функция для быстрого запуска
def process_video_folder_simple(folder_path, model_path=None, output_file="video_results.csv"):
    """Простая функция для обработки папки с видео"""

    # Если путь к модели не указан, используем относительный путь
    if model_path is None:
        # Попробуем найти модель относительно текущей директории
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "videomae_results", "checkpoint-14172"),
            os.path.join(current_dir, "checkpoint-14172"),
            os.path.join(os.path.dirname(current_dir), "videomae-base-finetuned-klin",
                         "checkpoint-24536")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

    if model_path is None:
        print("❌ Укажите путь к модели:")
        model_path = input("Введите полный путь к папке с моделью: ")

    print(f"🔧 Используется модель: {model_path}")
    print(f"📁 Обрабатываемая папка: {folder_path}")

    # Инициализация классификатора
    classifier = VideoFolderClassifier(model_path)

    # Обработка папки
    results = classifier.predict_folder(
        folder_path=folder_path,
        output_file=output_file
    )

    return results


# Основной запуск
if __name__ == "__main__":
    # Укажите путь к вашей папке с видео
    video_folder = r"/home/cipher/Documents/VS_code/KLIN/data/raw/KLIN/Test/violent"

    # Укажите путь к модели (если нужно)
    model_path = "/home/cipher/Documents/VS_code/KLIN/videomae_results/checkpoint-28344" # Автоматический поиск

    results = process_video_folder_simple(
        folder_path=video_folder,
        model_path=model_path,
        output_file="video_classification_results.csv"
    )

    # Показать результаты
    if not results.empty:
        print("\n📋 Первые результаты:")
        print(results.head(10))