import os
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import decord
from decord import VideoReader, cpu

# ──────────────────────────────────────────────────────────────────────────────
# Маппинг классов (как в OPear, проверьте на точность)
# ──────────────────────────────────────────────────────────────────────────────
label2id = {
    "Abuse": 0,
    "Arrest": 1,
    "Arson": 2,
    "Assault": 3,
    "Burglary": 4,
    "Explosion": 5,
    "Fighting": 6,
    "Normal_Videos_event": 7,  # Важно: как в модели OPear
    "RoadAccidents": 8,
    "Robbery": 9,
    "Shooting": 10,
    "Shoplifting": 11,
    "Stealing": 12,
    "Vandalism": 13
}
id2label = {v: k for k, v in label2id.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Пути к датасету (поменяйте на свои)
# ──────────────────────────────────────────────────────────────────────────────
train_dir = "ucf_crime_dataset/train"  # папка с подпапками классов
val_dir = "ucf_crime_dataset/val"  # аналогично

# ──────────────────────────────────────────────────────────────────────────────
# Processor для предобработки видео
# ──────────────────────────────────────────────────────────────────────────────
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")


# ──────────────────────────────────────────────────────────────────────────────
# Функция сэмплинга 16 кадров из видео
# ──────────────────────────────────────────────────────────────────────────────
def sample_frames(video_path, num_frames=16):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return None
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # [num_frames, H, W, 3]
        frames = list(frames)  # list of arrays для processor
        return frames
    except Exception as e:
        print(f"Ошибка загрузки {video_path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Preprocess функция для batched .map
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(examples):
    frames_list = []
    labels = examples["label"]
    for path in examples["video_path"]:
        frames = sample_frames(path)
        if frames is None:
            # Placeholder: чёрные кадры
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 16
        frames_list.append(frames)

    # Processor обрабатывает list of lists frames
    inputs = processor(frames_list, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"], "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка датасета из папок (custom, поскольку video_folder не стандартный)
# ──────────────────────────────────────────────────────────────────────────────
def get_video_paths_and_labels(root_dir):
    paths, labels = [], []
    class_names = sorted(os.listdir(root_dir))  # сортируем для consistent label
    for label, class_name in enumerate(class_names):
        if class_name not in label2id:
            print(f"Предупреждение: класс {class_name} не в label2id")
            continue
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for video in os.listdir(class_dir):
            if video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                paths.append(os.path.join(class_dir, video))
                labels.append(label2id[class_name])  # используем label2id
    return {"video_path": paths, "label": labels}


train_data = get_video_paths_and_labels(train_dir)
val_data = get_video_paths_and_labels(val_dir)

dataset = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "validation": Dataset.from_dict(val_data)
})

# Применяем preprocess (batched для скорости, batch_size под GPU)
dataset = dataset.map(
    preprocess,
    batched=True,
    batch_size=4,  # подстройте под вашу память
    remove_columns=["video_path"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Модель
# ──────────────────────────────────────────────────────────────────────────────
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-large",
    num_labels=14,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Метрика
# ──────────────────────────────────────────────────────────────────────────────
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


# ──────────────────────────────────────────────────────────────────────────────
# TrainingArguments (точно как в OPear)
# ──────────────────────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="./videomae_ucf_finetuned",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # effective bs=8
    num_train_epochs=4,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # если CUDA поддерживает
    seed=42,
    report_to="none"  # или "wandb" / "tensorboard"
)

# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,  # сохраняет processor для inference
    compute_metrics=compute_metrics
)

# Запуск обучения
trainer.train()

# После обучения: оценка на val
results = trainer.evaluate()
print(results)

# Сохранение модели (опционально)
trainer.save_model("./videomae_ucf_finetuned/final_model")