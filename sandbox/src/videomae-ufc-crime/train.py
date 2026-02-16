import os

import cv2
import evaluate  # type: ignore[import-untyped]
import mlflow.pytorch as mlflow_pytorch
import numpy as np
from datasets import Dataset, DatasetDict  # type: ignore[import-untyped]
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

import mlflow

# ──────────────────────────────────────────────────────────────────────────────
# Маппинг классов
# ──────────────────────────────────────────────────────────────────────────────
label2id = {
    "Abuse": 0,
    "Arrest": 1,
    "Arson": 2,
    "Assault": 3,
    "Burglary": 4,
    "Explosion": 5,
    "Fighting": 6,
    "Normal_Videos": 7,
    "RoadAccidents": 8,
    "Robbery": 9,
    "Shooting": 10,
    "Shoplifting": 11,
    "Stealing": 12,
    "Vandalism": 13,
}
id2label = {v: k for k, v in label2id.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Пути к данным
# ──────────────────────────────────────────────────────────────────────────────
train_dir = "ucf_crime_dataset/train"
val_dir = "ucf_crime_dataset/val"

# ──────────────────────────────────────────────────────────────────────────────
# Processor
# ──────────────────────────────────────────────────────────────────────────────
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")


# ──────────────────────────────────────────────────────────────────────────────
# Сэмплирование кадров
# ──────────────────────────────────────────────────────────────────────────────
def sample_frames(video_path, num_frames=16):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        if len(frames) == 0:
            return None
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return frames
    except Exception as e:
        print(f"Ошибка загрузки {video_path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Preprocess
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(examples):
    frames_list = []
    labels = examples["label"]
    for path in examples["video_path"]:
        frames = sample_frames(path)
        if frames is None:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 16
        frames_list.append(frames)

    inputs = processor(frames_list, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"], "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка путей
# ──────────────────────────────────────────────────────────────────────────────
def get_video_paths_and_labels(root_dir):
    paths, labels = [], []
    class_names = sorted(os.listdir(root_dir))
    for _label_idx, class_name in enumerate(class_names):
        if class_name not in label2id:
            print(f"Предупреждение: неизвестный класс {class_name}")
            continue
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for video in os.listdir(class_dir):
            if video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                paths.append(os.path.join(class_dir, video))
                labels.append(label2id[class_name])
    return {"video_path": paths, "label": labels}


train_data = get_video_paths_and_labels(train_dir)
val_data = get_video_paths_and_labels(val_dir)

dataset = DatasetDict(
    {"train": Dataset.from_dict(train_data), "validation": Dataset.from_dict(val_data)}
)

dataset = dataset.map(
    preprocess, batched=True, batch_size=4, remove_columns=["video_path"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Модель
# ──────────────────────────────────────────────────────────────────────────────
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-large",
    num_labels=14,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Метрика
# ──────────────────────────────────────────────────────────────────────────────
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    result = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    if result is None:
        return {}
    return dict(result)


# ──────────────────────────────────────────────────────────────────────────────
# Training Arguments
# ──────────────────────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="./videomae_ucf_finetuned",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    seed=42,
    report_to=["mlflow"],
)

# ──────────────────────────────────────────────────────────────────────────────
# MLflow + Trainer
# ──────────────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")  # ← поменяйте если используете сервер
mlflow.set_experiment("VideoMAE-UCF-Crime-Finetune")

with mlflow.start_run(run_name="videomae-large-ucf-crime"):
    # Логируем основные параметры
    mlflow.log_params(
        {
            "model_name": "MCG-NJU/videomae-large",
            "num_labels": 14,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "effective_batch_size": args.per_device_train_batch_size
            * args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "optimizer": args.optim,
            "fp16": args.fp16,
            "seed": args.seed,
        }
    )

    # Добавляем теги (удобно для фильтрации)
    mlflow.set_tag("dataset", "UCF-Crime")
    mlflow.set_tag("task", "video-classification-14-classes")
    mlflow.set_tag("sampling", "uniform-16-frames")

    # Создаём Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Включаем автологгер для transformers → mlflow будет логировать loss, accuracy и т.д.
    mlflow_pytorch.autolog(log_models=False)  # модели сохраним вручную

    print("Начало обучения...")
    trainer.train()

    # Финальная оценка
    print("Оценка на валидации...")
    results = trainer.evaluate()
    print(results)

    # Логируем финальные метрики
    mlflow.log_metrics(results)

    # Сохраняем модель и processor как артефакты
    print("Сохранение модели и процессора в MLflow...")
    trainer.save_model("./videomae_ucf_finetuned/final_model")
    processor.save_pretrained("./videomae_ucf_finetuned/final_model")

    mlflow.log_artifacts("./videomae_ucf_finetuned/final_model", artifact_path="model")

print("Обучение завершено. Результаты и модель доступны в MLflow.")
