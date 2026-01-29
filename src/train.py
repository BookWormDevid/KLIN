import logging
import pathlib
import random
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

import mlflow

# ======================= CONFIG =======================
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "raw" / "KLIN" / "Train"
VAL_DIR = BASE_DIR / "data" / "raw" / "KLIN" / "Val"
TEST_DIR = BASE_DIR / "data" / "raw" / "KLIN" / "Test"

OUTPUT_DIR = BASE_DIR / "videomae_results"
MODEL_NAME = str(BASE_DIR / "models" / "videomae-large")

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 2
EPOCHS = 8
CLASS_WEIGHTS = [1.0, 5.0]  # [nonviolent, violent]
MAX_RETRIES = 5
RNG_SEED = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================
# ======================= DATASET ======================
# ======================================================
class BetterVideoDataset(Dataset):
    """
    Returns:
        {
            "pixel_values": Tensor[T, C, H, W],
            "labels": Tensor([])
        }
    """

    def __init__(
        self,
        root_dir: pathlib.Path | str,
        num_frames: int = NUM_FRAMES,
        img_size: int = IMG_SIZE,
        mode: str = "train",
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        assert mode in {"train", "val", "test"}
        self.root_dir = pathlib.Path(root_dir)
        self.num_frames = int(num_frames)
        self.img_size = int(img_size)
        self.mode = mode
        self.samples: list[dict[str, Any]] = []
        self.rng = random.Random(seed if seed is not None else RNG_SEED)
        self._collect_paths(max_samples)
        self.transform = self._build_transforms()

    def _collect_paths(self, max_samples: int | None) -> None:
        classes = ["nonviolent", "violent"]
        exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
        for label, name in enumerate(classes):
            cls_dir = self.root_dir / name
            if not cls_dir.exists():
                logger.warning("Class directory does not exist: %s", cls_dir)
                continue
            files = [p for p in cls_dir.rglob("*") if p.suffix.lower() in exts]
            files = sorted(files)
            if max_samples is not None:
                per_class = max_samples // len(classes)
                files = files[:per_class]
            for p in sorted(files):
                self.samples.append({"path": p, "label": int(label)})

    def _build_transforms(self) -> transforms.Compose:
        common = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if self.mode == "train":
            aug = [
                transforms.Resize(256),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
            return transforms.Compose([transforms.ToPILImage(), *aug, *common[1:]])
        else:
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.img_size, self.img_size)),
                    *common[1:],
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_all_frames(path: pathlib.Path) -> list[np.ndarray] | None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return None
        frames: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames:
            return None
        return frames

    def _sample_indices(self, total: int) -> list[int]:
        if total <= self.num_frames:
            idxs = list(range(total))
            while len(idxs) < self.num_frames:
                idxs.append(idxs[-1])
            return idxs
        if self.mode == "train":
            start = random.randint(0, total - self.num_frames)
            return list(range(start, start + self.num_frames))
        else:
            step = total / float(self.num_frames)
            return [int(i * step) for i in range(self.num_frames)]

    def _load_clip(self, path: pathlib.Path) -> np.ndarray | None:
        frames = self._read_all_frames(path)
        if frames is None:
            return None
        idxs = self._sample_indices(len(frames))
        sampled = [frames[i] for i in idxs]
        return np.stack(sampled)  # (T, H, W, C)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tries = 0
        cur = int(idx)
        while tries < MAX_RETRIES:
            sample = self.samples[cur]
            clip_np = self._load_clip(sample["path"])
            if clip_np is None:
                cur = random.randint(0, len(self.samples) - 1)
                tries += 1
                continue
            frames = [self.transform(f) for f in clip_np]
            video = torch.stack(frames)  # (T, C, H, W)
            label = torch.tensor(sample["label"], dtype=torch.long)
            return {"pixel_values": video, "labels": label}
        raise RuntimeError("Too many video decode failures")


# ======================================================
# ======================= COLLATE ======================
# ======================================================
def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch]).contiguous()
    labels = torch.stack([b["labels"] for b in batch]).contiguous()
    return {"pixel_values": pixel_values, "labels": labels}


# ======================================================
# =================== WEIGHTED TRAINER =================
# ======================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs_copy = dict(inputs)
        labels = inputs_copy.pop("labels", None)
        outputs = model(**inputs_copy)
        logits = outputs.logits  # (batch, num_classes)

        # Weighted loss on device
        device = logits.device
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ======================================================
# ======================= METRICS ======================
# ======================================================
def compute_metrics(eval_pred) -> dict[str, float]:
    if isinstance(eval_pred, tuple) or isinstance(eval_pred, list):
        logits, labels = eval_pred
    else:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # numerically stable softmax
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shift)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    preds = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[1], average="binary", zero_division=0
    )

    macro_f1 = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )[2]

    bal_acc = balanced_accuracy_score(labels, preds)

    try:
        auroc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auroc = float("nan")

    return {
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "violent_precision": float(precision),
        "violent_recall": float(recall),
        "violent_f1": float(f1),
        "auroc": float(auroc),
    }


# ======================================================
# ================= MULTI-CLIP EVAL ====================
# ======================================================
@torch.no_grad()
def multi_clip_evaluate(
    trainer: Trainer, dataset: BetterVideoDataset, n_clips: int = 5
) -> dict[str, float]:
    model = trainer.model
    assert model is not None
    model.eval()
    device = next(model.parameters()).device

    all_preds: list[int] = []
    all_labels: list[int] = []

    for sample in dataset.samples:
        path = sample["path"]
        label = sample["label"]
        clips: list[torch.Tensor] = []
        for _ in range(n_clips):
            try:
                arr = dataset._load_clip(path)
            except Exception:
                arr = None
            if arr is None:
                continue
            frames = [dataset.transform(f) for f in arr]
            clips.append(torch.stack(frames))
        if not clips:
            continue
        batch = torch.stack(clips).to(device)  # (n, T, C, H, W)
        outputs = model(pixel_values=batch)
        probs = torch.softmax(outputs.logits, dim=-1).mean(dim=0)
        pred = int(torch.argmax(probs).cpu().item())
        all_preds.append(pred)
        all_labels.append(label)

    if not all_preds:
        raise RuntimeError("No predictions in multi-clip eval")

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)

    acc = accuracy_score(labels_arr, preds_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_arr, preds_arr, labels=[1], average="binary", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "violent_precision": float(precision),
        "violent_recall": float(recall),
        "violent_f1": float(f1),
    }


# ======================================================
# =================== FREEZE HELPERS ===================
# ======================================================
def freeze_backbone(model: VideoMAEForVideoClassification) -> None:
    backbone = model.videomae
    for p in backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: VideoMAEForVideoClassification) -> None:
    backbone = model.videomae
    for p in backbone.parameters():
        p.requires_grad = True


# ======================================================
# ========================= MAIN =======================
# ======================================================
def main() -> None:
    # reproducibility
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow/mlflow.db")
    mlflow.set_experiment("videomae_violence_detection")
    mlflow.start_run()

    mlflow.log_params(
        {
            "num_frames": NUM_FRAMES,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "class_weights": CLASS_WEIGHTS,
            "model_name": MODEL_NAME,
        }
    )

    train_dataset = BetterVideoDataset(TRAIN_DIR, mode="train", seed=RNG_SEED)
    val_dataset = BetterVideoDataset(VAL_DIR, mode="val", seed=RNG_SEED)
    test_dataset = BetterVideoDataset(TEST_DIR, mode="test", seed=RNG_SEED)

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        label2id={"nonviolent": 0, "violent": 1},
        id2label={0: "nonviolent", 1: "violent"},
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="violent_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        save_total_limit=2,
        report_to=["mlflow"],
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed")

    logger.info("Running multi-clip evaluation on validation set...")
    mc_metrics = multi_clip_evaluate(trainer, val_dataset, n_clips=5)
    logger.info("Multi-clip validation metrics: %s", mc_metrics)
    mlflow.log_metrics({f"multiclip_val_{k}": v for k, v in mc_metrics.items()})

    logger.info("Running test evaluation (single-clip)...")
    test_metrics = trainer.evaluate(test_dataset)
    logger.info("Test metrics: %s", test_metrics)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    image_processor.save_pretrained(str(OUTPUT_DIR))

    mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="model")

    logger.info("Model and processor saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
