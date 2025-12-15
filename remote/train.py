"""
VideoMAE обучение с OpenCV для загрузки видео
Специально для Windows и RTX 5060
"""

import os
import pathlib

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import evaluate
import mlflow
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
from typing import List, Dict, Optional

# ==================== КОНФИГУРАЦИЯ ====================
BASE_DIR = pathlib.Path(__file__).parent.parent
TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "KLIN", "Train")
VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "KLIN", "Val")
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "KLIN", "Test")

OUTPUT_DIR = "./videomae_results"
MODEL_NAME = os.path.join(BASE_DIR, "models", "videomae-large")  # Используем онлайн модель
NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 2  # Для RTX 5060 начните с 2
EPOCHS = 8


# ==================== COLLATE ФУНКЦИЯ ====================
def collate_fn(batch):
    """Собирает батч - должна быть на верхнем уровне"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


# ==================== ДАТАСЕТ С OPENCV ====================
class OpenCVVideoDataset(Dataset):
    """Датасет для видео с использованием OpenCV"""

    def __init__(self,
                 root_dir: str,
                 num_frames: int = 16,
                 img_size: int = 224,
                 mode: str = 'train',
                 max_samples: Optional[int] = None):
        """
        Args:
            root_dir: Папка с подпапками 'nonviolent' и 'violent'
            num_frames: Количество кадров для каждого видео
            img_size: Размер кадра (квадратный)
            mode: 'train' или 'val'
            max_samples: Максимальное количество видео (для тестирования)
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.mode = mode

        # Собираем все видео файлы
        self.samples: List[Dict] = []
        self._collect_video_paths(max_samples)

        # Трансформации
        self.transform = self._get_transforms()

        # Статистика
        self.successful_loads = 0
        self.failed_loads = 0

    def _collect_video_paths(self, max_samples: Optional[int]):
        """Собирает пути ко всем видео файлам"""
        print(f"Сканирование {self.root_dir}...")

        for label_idx, label_name in enumerate(['nonviolent', 'violent']):
            label_dir = os.path.join(self.root_dir, label_name)

            if not os.path.exists(label_dir):
                print(f"  Предупреждение: папка {label_dir} не существует")
                continue

            # Получаем все видео файлы
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
            video_files = []

            for root, _, files in os.walk(label_dir):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        video_files.append(os.path.join(root, file))

            print(f"  Найдено {len(video_files)} видео в {label_name}")

            # Ограничиваем если нужно
            if max_samples and len(video_files) > max_samples // 2:
                video_files = video_files[:max_samples // 2]

            # Добавляем в samples
            for video_path in video_files:
                self.samples.append({
                    'path': video_path,
                    'label': label_idx,
                    'label_name': label_name
                })

        print(f"Итого: {len(self.samples)} видео в датасете")

    def _get_transforms(self):
        """Создает трансформации для видео"""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),  # Сначала увеличиваем
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet статистика
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:  # val/test
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),  # Прямой ресайз
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _load_video_frames(self, video_path: str) -> Optional[np.ndarray]:
        """Загружает и выбирает кадры из видео"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.failed_loads += 1
                return None

            # Получаем информацию о видео
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0:
                cap.release()
                self.failed_loads += 1
                return None

            # Стратегия выбора кадров
            if total_frames <= self.num_frames:
                # Если кадров меньше нужного - берем все
                frame_indices = list(range(total_frames))
                # Дублируем последний кадр
                while len(frame_indices) < self.num_frames:
                    frame_indices.append(frame_indices[-1])
            else:
                if self.mode == 'train':
                    # Случайный отрезок
                    start_idx = np.random.randint(0, total_frames - self.num_frames)
                    frame_indices = list(range(start_idx, start_idx + self.num_frames))
                else:
                    # Равномерное распределение
                    step = total_frames / self.num_frames
                    frame_indices = [int(i * step) for i in range(self.num_frames)]

            # Читаем выбранные кадры
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Конвертируем BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Базовый ресайз для экономии памяти
                    frame_resized = cv2.resize(frame_rgb, (256, 256))
                    frames.append(frame_resized)
                else:
                    # Черный кадр если не прочитался
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))

            cap.release()
            self.successful_loads += 1

            return np.array(frames)  # (T, H, W, C)

        except Exception as e:
            self.failed_loads += 1
            print(f"Ошибка загрузки {os.path.basename(video_path)}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        video_info = self.samples[idx]
        video_path = video_info['path']
        label = video_info['label']

        # Пробуем загрузить видео
        frames = self._load_video_frames(video_path)

        if frames is None:
            # Возвращаем нулевой тензор при ошибке
            dummy_video = torch.zeros((self.num_frames, 3, self.img_size, self.img_size))
            return {
                'pixel_values': dummy_video,
                'labels': torch.tensor(label, dtype=torch.long)
            }

        # Применяем трансформации к каждому кадру
        transformed_frames = []
        for frame in frames:  # frame shape: (H, W, C)
            transformed = self.transform(frame)  # (C, H, W)
            transformed_frames.append(transformed)

        # Собираем все кадры вместе
        video_tensor = torch.stack(transformed_frames)  # (T, C, H, W)

        # Проверяем размеры
        if video_tensor.shape[1:] != (3, self.img_size, self.img_size):
            # Принудительно ресайзим если нужно
            video_tensor = torch.nn.functional.interpolate(
                video_tensor,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )

        return {
            'pixel_values': video_tensor,
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def print_stats(self):
        """Печатает статистику загрузки"""
        total = self.successful_loads + self.failed_loads
        if total > 0:
            success_rate = (self.successful_loads / total) * 100
            print(f"Статистика загрузки: {self.successful_loads}/{total} "
                  f"({success_rate:.1f}% успешно)")


# ==================== МЕТРИКИ ====================
def compute_metrics(eval_pred):
    """Вычисляет метрики для оценки"""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


# ==================== ОСНОВНОЙ КОД ====================
def main():
    # ==================== НАСТРОЙКА MLflow ====================
    MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow")
    os.makedirs(MLFLOW_DIR, exist_ok=True)

    # Явный backend для хранения логов
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
    mlflow.set_experiment("videomae_violence_detection")

    # ==================== START RUN ====================
    mlflow.start_run(run_name="videomae-training")
    try:
        # Проверка оборудования
        print("=" * 50)
        print("ПРОВЕРКА ОБОРУДОВАНИЯ")
        print("=" * 50)
        print(f"PyTorch версия: {torch.__version__}")
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA версия: {torch.version.cuda}")
            fp16_enabled = True
            device = torch.device("cuda")
        else:
            print("ВНИМАНИЕ: CUDA не доступна! Обучение на CPU будет медленным.")
            fp16_enabled = False
            device = torch.device("cpu")

        print(f"\nИспользуемое устройство: {device}")

        # Логируем параметры в MLflow
        mlflow.log_params({
            "model": MODEL_NAME,
            "num_frames": NUM_FRAMES,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "device": str(device),
            "fp16": fp16_enabled,
        })

        # ==================== СОЗДАНИЕ ДАТАСЕТОВ ====================
        print("\n" + "=" * 50)
        print("СОЗДАНИЕ ДАТАСЕТОВ")
        print("=" * 50)

        print("Создание тренировочного датасета...")
        train_dataset = OpenCVVideoDataset(
            root_dir=TRAIN_DIR,
            num_frames=NUM_FRAMES,
            img_size=IMG_SIZE,
            mode='train',
            max_samples=1000
        )

        print("Создание валидационного датасета...")
        val_dataset = OpenCVVideoDataset(
            root_dir=VAL_DIR,
            num_frames=NUM_FRAMES,
            img_size=IMG_SIZE,
            mode='val',
            max_samples=200
        )

        print("Создание тестового датасета...")
        test_dataset = OpenCVVideoDataset(
            root_dir=TEST_DIR,
            num_frames=NUM_FRAMES,
            img_size=IMG_SIZE,
            mode='val',
            max_samples=200
        )

        print(f"\nРазмеры датасетов:")
        print(f"  Тренировочный: {len(train_dataset)} видео")
        print(f"  Валидационный: {len(val_dataset)} видео")
        print(f"  Тестовый: {len(test_dataset)} видео")

        # Тестируем загрузку одного элемента
        print("\nТестируем загрузку одного видео...")
        test_sample = train_dataset[0]
        print(f"  Форма видео: {test_sample['pixel_values'].shape}")
        print(f"  Метка: {test_sample['labels'].item()} "
              f"({'violent' if test_sample['labels'].item() == 1 else 'nonviolent'})")
        print(f"  Min значение пикселей: {test_sample['pixel_values'].min():.3f}")
        print(f"  Max значение пикселей: {test_sample['pixel_values'].max():.3f}")
        print(f"  Mean значение пикселей: {test_sample['pixel_values'].mean():.3f}")

        # ==================== ЗАГРУЗКА МОДЕЛИ ====================
        print("\n" + "=" * 50)
        print("ЗАГРУЗКА МОДЕЛИ")
        print("=" * 50)

        print(f"Загрузка VideoMAE модели: {MODEL_NAME}")
        image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

        model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            label2id={"nonviolent": 0, "violent": 1},
            id2label={0: "nonviolent", 1: "violent"},
            ignore_mismatched_sizes=True,
        )

        print("\nТестирование модели на одном примере...")
        model.eval()
        with torch.no_grad():
            test_input = test_sample['pixel_values'].unsqueeze(0).to(device)
            model = model.to(device)
            test_output = model(test_input)
            print(f"  Форма входа: {test_input.shape}")
            print(f"  Форма выхода: {test_output.logits.shape}")
            print(f"  Предсказание: {test_output.logits}")

        model = model.to(device)
        print(f"\nМодель перемещена на: {next(model.parameters()).device}")

        # ==================== НАСТРОЙКА ОБУЧЕНИЯ ====================
        print("\n" + "=" * 50)
        print("НАСТРОЙКА ОБУЧЕНИЯ")
        print("=" * 50)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            adam_epsilon=1e-8,
            adam_beta1=0.9,
            adam_beta2=0.999,
            logging_dir=f"{OUTPUT_DIR}/logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=fp16_enabled,
            dataloader_num_workers=0, # не менять
            remove_unused_columns=False,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            report_to=[],  # отключаем автоматическое логирование, чтобы не было конфликтов
        )

        # ==================== TRAINER ====================
        print("\nСоздание Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=image_processor,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )

        # ==================== ОБУЧЕНИЕ ====================
        print("\n" + "=" * 50)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 50)

        print("\nОценка до обучения...")
        eval_results = trainer.evaluate()
        print(f"Начальная точность: {eval_results['eval_accuracy']:.4f}")

        trainer.train()
        print("\n✓ Обучение успешно завершено!")

        # Статистика загрузки
        print("\nСтатистика загрузки видео:")
        train_dataset.print_stats()
        val_dataset.print_stats()

        # ==================== ТЕСТИРОВАНИЕ ====================
        print("\n" + "=" * 50)
        print("ТЕСТИРОВАНИЕ")
        print("=" * 50)

        test_results = trainer.evaluate(test_dataset)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_results.items()})
        print("\nРезультаты тестирования:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")

        # ==================== СОХРАНЕНИЕ МОДЕЛИ ====================
        print("\n" + "=" * 50)
        print("СОХРАНЕНИЕ МОДЕЛИ")
        print("=" * 50)

        trainer.save_model(OUTPUT_DIR)
        image_processor.save_pretrained(OUTPUT_DIR)
        print(f"Модель сохранена в: {os.path.abspath(OUTPUT_DIR)}")

    finally:
        # ==================== КОРРЕКТНО ЗАКРЫВАЕМ RUN ====================
        mlflow.end_run()
        print(f"MLflow Run завершён. ID: {mlflow.active_run()}")  # None после end_run



def debug_mode():
    """Режим отладки при возникновении ошибок"""
    print("\n=== РЕЖИМ ОТЛАДКИ ===")

    # Создаем мини-датасет
    debug_dataset = OpenCVVideoDataset(
        root_dir=TRAIN_DIR,
        num_frames=8,  # Меньше кадров для отладки
        img_size=224,
        mode='train',
        max_samples=10  # Только 10 видео
    )

    # Проверяем загрузку
    print(f"Размер отладочного датасета: {len(debug_dataset)}")

    for i in range(min(3, len(debug_dataset))):
        sample = debug_dataset[i]
        print(f"\nПример {i}:")
        print(f"  Форма: {sample['pixel_values'].shape}")  # Должно быть (8, 3, 224, 224)
        print(f"  Метка: {sample['labels'].item()}")
        print(f"  Min: {sample['pixel_values'].min():.3f}, "
              f"Max: {sample['pixel_values'].max():.3f}, "
              f"Mean: {sample['pixel_values'].mean():.3f}")

    # Проверяем DataLoader
    debug_loader = DataLoader(
        debug_dataset,
        batch_size=2,
        collate_fn=collate_fn,
        num_workers=0
    )

    print("\nТестируем DataLoader...")
    for batch_idx, batch in enumerate(debug_loader):
        print(f"\nБатч {batch_idx}:")
        print(f"  pixel_values shape: {batch['pixel_values'].shape}")  # Должно быть (2, 8, 3, 224, 224)
        print(f"  labels: {batch['labels']}")

        if batch_idx >= 2:
            break

    print("\n✓ Отладка завершена. Если все работает, увеличьте max_samples.")


if __name__ == "__main__":
    # Обработка Ctrl+C
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем.")
    except Exception as e:
        print(f"\n\nКритическая ошибка: {e}")
        import traceback

        traceback.print_exc()