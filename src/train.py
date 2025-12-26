import os
import pathlib

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

import mlflow

# ==================== КОНФИГУРАЦИЯ ====================
BASE_DIR = pathlib.Path(__file__).parent.parent
TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "KLIN", "Train")
VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "KLIN", "Val")
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "KLIN", "Test")

OUTPUT_DIR = "./videomae_results"
MODEL_NAME = os.path.join(BASE_DIR, "models", "videomae-large")
NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 2
EPOCHS = 8
CLASS_WEIGHTS = [1.0, 10.0]  # nonviolent, violent

# ==================== COLLATE ФУНКЦИЯ ====================
def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}

# ==================== ДАТАСЕТ ====================
class OpenCVVideoDataset(Dataset):
    """Video dataset using OpenCV with oversampling for violent class."""

    def __init__(self, root_dir: str, num_frames: int = 16, img_size: int = 224, 
                 mode: str = 'train', max_samples: int | None = None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.mode = mode
        self.samples: list[dict] = []
        self._collect_video_paths(max_samples)
        self.transform = self._get_transforms()
        self.successful_loads = 0
        self.failed_loads = 0

    def _collect_video_paths(self, max_samples: int | None):
        for label_idx, label_name in enumerate(['nonviolent', 'violent']):
            label_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
            video_files = [os.path.join(root, f) 
                           for root, _, files in os.walk(label_dir)
                           for f in files if f.lower().endswith(video_extensions)]
            
            # Oversample violent class
            if label_name == 'violent':
                video_files = video_files * 3

            if max_samples and len(video_files) > max_samples // 2:
                video_files = video_files[:max_samples // 2]

            for video_path in video_files:
                self.samples.append({'path': video_path, 'label': label_idx, 'label_name': label_name})

    def _get_transforms(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

    def _load_video_frames(self, video_path: str, label:int) -> np.ndarray | None:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None

            # Temporal sampling
            if total_frames <= self.num_frames:
                frame_indices = list(range(total_frames))
                while len(frame_indices) < self.num_frames:
                    frame_indices.append(frame_indices[-1])
            else:
                if self.mode == 'train':
                    start_idx = np.random.randint(0, max(1, total_frames - self.num_frames))
                    frame_indices = list(range(start_idx, start_idx + self.num_frames))
                else:
                    step = total_frames / self.num_frames
                    frame_indices = [int(i * step) for i in range(self.num_frames)]

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (256, 256))
                    frames.append(frame_resized)
            cap.release()
            if len(frames) < self.num_frames:
                return None
            return np.array(frames)
        except:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_info = self.samples[idx]
        frames = self._load_video_frames(video_info['path'], video_info['label'])
        if frames is None:
            # retry with random other sample
            return self.__getitem__(np.random.randint(len(self)))
        transformed_frames = [self.transform(f) for f in frames]
        video_tensor = torch.stack(transformed_frames)
        return {'pixel_values': video_tensor, 'labels': torch.tensor(video_info['label'], dtype=torch.long)}

    def print_stats(self):
        total = self.successful_loads + self.failed_loads
        if total > 0:
            print(f"Loaded: {self.successful_loads}/{total} videos ({self.successful_loads/total*100:.1f}%)")

# ==================== METRICS ====================
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = (preds == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[1], average='binary')
    return {
        "accuracy": acc,
        "violent_precision": precision,
        "violent_recall": recall,
        "violent_f1": f1
    }

# ==================== CUSTOM TRAINER WITH WEIGHTED LOSS ====================
from torch.nn import CrossEntropyLoss


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor(CLASS_WEIGHTS).to(logits.device)
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==================== MAIN ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow/mlflow.db")
    mlflow.set_experiment("videomae_violence_detection")
    mlflow.start_run()

    # Datasets
    train_dataset = OpenCVVideoDataset(TRAIN_DIR, NUM_FRAMES, IMG_SIZE, 'train')
    val_dataset = OpenCVVideoDataset(VAL_DIR, NUM_FRAMES, IMG_SIZE, 'val')
    test_dataset = OpenCVVideoDataset(TEST_DIR, NUM_FRAMES, IMG_SIZE, 'val')

    # Model
    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        label2id={"nonviolent":0,"violent":1},
        id2label={0:"nonviolent",1:"violent"},
        ignore_mismatched_sizes=True
    ).to(device)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="violent_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{OUTPUT_DIR}/logs",
        gradient_accumulation_steps=2,
        gradient_checkpointing=True
    )

    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Test
    test_results = trainer.evaluate(test_dataset)
    print("Test results:", test_results)
    mlflow.log_metrics({f"test_{k}": v for k,v in test_results.items()})

    # Save
    trainer.save_model(OUTPUT_DIR)
    image_processor.save_pretrained(OUTPUT_DIR)
    mlflow.end_run()
    print("Model saved.")

if __name__ == "__main__":
    main()
