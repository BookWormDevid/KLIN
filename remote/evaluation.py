import os
import torch
import cv2
import numpy as np
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from tqdm import tqdm
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent


class VideoClassifier:
    def __init__(self, model_path: str = None, chunk_size: int = 16, frame_size: tuple = (224, 224)):
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Определяем путь к модели
        if model_path is None:
            model_path = self._find_model_path()

        print(f"🔄 Загрузка модели из: {model_path}")

        # Загрузка модели и процессора
        self.processor = VideoMAEImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

        print(f"✅ Модель загружена! Доступные классы: {self.model.config.id2label}")

    def _find_model_path(self):
        """Автоматически найти путь к модели"""
        possible_paths = [
            os.path.join(BASE_DIR, "models", "videomae-large")
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                return path

        print("❌ Модель не найдена автоматически.")
        model_path = input("📁 Введите полный путь к папке с моделью: ")
        if os.path.exists(model_path):
            return model_path
        else:
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    def _read_video_frames(self, video_path: str) -> np.ndarray:
        """Чтение видео и возврат кадров"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"📹 Видео: {os.path.basename(video_path)}")
        print(f"   Количество кадров: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Длительность: {duration:.2f} секунд")

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
        """Разделение кадров на чанки"""
        t = len(frames)
        padding_needed = (-t) % self.chunk_size

        if padding_needed > 0:
            padding = np.zeros((padding_needed, *self.frame_size, 3), dtype=np.uint8)
            frames = np.concatenate([frames, padding], axis=0)

        num_chunks = len(frames) // self.chunk_size
        return frames.reshape(num_chunks, self.chunk_size, *self.frame_size, 3)

    def predict_video(self, video_path: str, batch_size: int = 4) -> dict:
        """Предсказание для одного видео"""
        try:
            # Чтение и обработка видео
            frames = self._read_video_frames(video_path)
            chunks = self._chunk_frames(frames)

            print(f"   Обработано кадров: {len(frames)}")
            print(f"   Создано чанков: {len(chunks)}")

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
            final_logits = torch.mean(torch.cat(all_predictions), dim=0)
            probabilities = torch.nn.functional.softmax(final_logits, dim=0)
            predicted_idx = final_logits.argmax().item()
            confidence = probabilities[predicted_idx].item()

            # Получение всех вероятностей классов
            class_probs = {}
            for idx, class_name in self.model.config.id2label.items():
                class_probs[class_name] = probabilities[idx].item()

            result = {
                'video_name': os.path.basename(video_path),
                'video_path': video_path,
                'predicted_class': self.model.config.id2label[predicted_idx],
                'confidence': confidence,
                'all_predictions': class_probs,
                'num_frames': len(frames),
                'num_chunks': len(chunks)
            }

            return result

        except Exception as e:
            print(f"❌ Ошибка обработки {os.path.basename(video_path)}: {e}")
            return {
                'video_name': os.path.basename(video_path),
                'video_path': video_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }


def process_video(input_path: str, model_path: str = None):
    """Основная функция обработки видео или папки с видео"""

    # Если путь к модели не указан, пробуем найти автоматически
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "models", "KLIN-model"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

    if model_path is None or not os.path.exists(model_path):
        print("❌ Укажите путь к модели:")
        model_path = input("Введите полный путь к папке с моделью: ")

    print(f"🔧 Используется модель: {model_path}")

    # Инициализация классификатора
    classifier = VideoClassifier(model_path)

    # Проверяем, является ли путь файлом или папкой
    if os.path.isfile(input_path):
        # Обработка одного файла
        print("\n" + "=" * 50)
        print(f"🎬 ОБРАБОТКА ВИДЕОФАЙЛА")
        print("=" * 50)

        result = classifier.predict_video(input_path)

        if result.get('error'):
            print(f"\n❌ Ошибка: {result['error']}")
            return

        print("\n" + "=" * 50)
        print(f"📊 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
        print("=" * 50)
        print(f"📁 Видео: {result['video_name']}")
        print(f"🎯 Предсказанный класс: {result['predicted_class']}")
        print(f"📈 Уверенность: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")

        # Вывод всех вероятностей
        print("\n📋 Вероятности всех классов:")
        all_probs = result['all_predictions']
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        for class_name, prob in sorted_probs:
            percentage = prob * 100
            if prob == result['confidence']:
                print(f"   ✅ {class_name}: {prob:.4f} ({percentage:.2f}%)")
            else:
                print(f"   📊 {class_name}: {prob:.4f} ({percentage:.2f}%)")

    elif os.path.isdir(input_path):
        # Обработка папки
        print("\n" + "=" * 50)
        print(f"📁 ОБРАБОТКА ПАПКИ С ВИДЕО")
        print("=" * 50)
        print(f"Папка: {input_path}")

        # Поиск видеофайлов
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        video_files = []

        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))

        print(f"Найдено видео файлов: {len(video_files)}")

        if len(video_files) == 0:
            print("❌ Видео файлы не найдены!")
            return

        # Обработка всех видео
        results = []
        for video_path in tqdm(video_files, desc="Обработка видео"):
            result = classifier.predict_video(video_path)
            results.append(result)

            if not result.get('error'):
                print(
                    f"\n📹 {os.path.basename(video_path)}: {result['predicted_class']} ({result['confidence'] * 100:.1f}%)")
            else:
                print(f"\n❌ {os.path.basename(video_path)}: Ошибка - {result.get('error', 'Unknown')}")

        # Вывод статистики
        successful = [r for r in results if not r.get('error')]
        errors = [r for r in results if r.get('error')]

        print("\n" + "=" * 50)
        print(f"📈 СВОДНАЯ СТАТИСТИКА")
        print("=" * 50)
        print(f"Всего видео: {len(results)}")
        print(f"Успешно обработано: {len(successful)}")
        print(f"Ошибок: {len(errors)}")

        if successful:
            avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
            print(f"Средняя уверенность: {avg_confidence:.4f} ({avg_confidence * 100:.2f}%)")

            # Распределение по классам
            class_dist = {}
            for r in successful:
                cls = r['predicted_class']
                class_dist[cls] = class_dist.get(cls, 0) + 1

            print("\n📊 Распределение по классам:")
            for cls, count in sorted(class_dist.items()):
                percentage = (count / len(successful)) * 100
                print(f"   {cls}: {count} видео ({percentage:.1f}%)")
    else:
        print(f"❌ Путь не существует: {input_path}")


# Пример использования
if __name__ == "__main__":
    # Можно указать как путь к файлу, так и путь к папке
    input_path = r"C:\Users\meksi\Documents\GitHub\KLIN\data\raw\KLIN\Test"

    # Путь к модели
    model_path = r"C:\Users\meksi\Documents\GitHub\KLIN\models\KLIN-model"

    # Запуск обработки
    process_video(
        input_path=input_path,
        model_path=model_path
    )