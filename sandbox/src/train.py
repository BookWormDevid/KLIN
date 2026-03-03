import importlib
import logging
import math
import os
import pathlib
import random
import sys
from collections import Counter
from typing import Any

import numpy as np
import torch
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    set_seed,
)

import mlflow


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "KLIN",
    PROJECT_ROOT / "data" / "raw" / "klin",
]
DATASET_DIR_NAME = "Anomaly-detection-dataset"
SPLITS_REL = pathlib.Path(
    DATASET_DIR_NAME,
    "UCF_Crimes-Train-Test-Split",
    "Action_Regnition_splits",
)
CLASS_IDS_FILE = "ClassIDs.txt"

LOCAL_MODEL_LARGE_CKPT = PROJECT_ROOT / "models" / "videomae-large"
LOCAL_MODEL_BASE_CKPT = PROJECT_ROOT / "models" / "videomae-base"
MODEL_CKPT_EXPLICIT = os.getenv("MODEL_CKPT") is not None
DEFAULT_MODEL_CKPT = os.getenv(
    "MODEL_CKPT",
    (
        str(LOCAL_MODEL_LARGE_CKPT)
        if LOCAL_MODEL_LARGE_CKPT.exists()
        else "MCG-NJU/videomae-large"
    ),
)
SAFE_MODEL_CKPT = os.getenv(
    "SAFE_MODEL_CKPT",
    (
        str(LOCAL_MODEL_BASE_CKPT)
        if LOCAL_MODEL_BASE_CKPT.exists()
        else "MCG-NJU/videomae-base"
    ),
)

TRAIN_SPLIT_FILE = os.getenv("TRAIN_SPLIT_FILE", "train_001.txt")
TEST_SPLIT_FILE = os.getenv("TEST_SPLIT_FILE", "test_001.txt")

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "30"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.1"))
SEED = int(os.getenv("SEED", "42"))

FPS = float(os.getenv("FPS", "30"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "800"))

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE_VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if CUDA_AVAILABLE
    else None
)
MEMORY_SAFE_MODE = os.getenv("MEMORY_SAFE_MODE", "auto").strip().lower()
AUTO_SAFE_MODE = DEVICE_VRAM_GB is not None and DEVICE_VRAM_GB <= 16.5
RUNTIME_SAFE_MODE = MEMORY_SAFE_MODE == "on" or (
    MEMORY_SAFE_MODE == "auto" and AUTO_SAFE_MODE
)
MODEL_CKPT = (
    DEFAULT_MODEL_CKPT
    if MODEL_CKPT_EXPLICIT or not RUNTIME_SAFE_MODE
    else SAFE_MODEL_CKPT
)

DEFAULT_TRAIN_BATCH_SIZE = 1 if RUNTIME_SAFE_MODE else 2
DEFAULT_EVAL_BATCH_SIZE = 1 if RUNTIME_SAFE_MODE else DEFAULT_TRAIN_BATCH_SIZE
DEFAULT_GRAD_ACC_STEPS = 32 if RUNTIME_SAFE_MODE else 4

TRAIN_BATCH_SIZE = max(
    1,
    int(os.getenv("TRAIN_BATCH_SIZE", str(DEFAULT_TRAIN_BATCH_SIZE))),
)
EVAL_BATCH_SIZE = max(
    1,
    int(os.getenv("EVAL_BATCH_SIZE", str(DEFAULT_EVAL_BATCH_SIZE))),
)
GRAD_ACC_STEPS = max(
    1,
    int(os.getenv("GRAD_ACC_STEPS", str(DEFAULT_GRAD_ACC_STEPS))),
)
DATALOADER_NUM_WORKERS = max(
    0,
    int(os.getenv("DATALOADER_NUM_WORKERS", "0")),
)
USE_FP16 = os.getenv("USE_FP16", "1").strip().lower() in {"1", "true", "yes"}
USE_BF16 = os.getenv(
    "USE_BF16",
    "1" if CUDA_AVAILABLE and torch.cuda.is_bf16_supported() else "0",
).strip().lower() in {"1", "true", "yes"}
USE_BF16 = USE_BF16 and CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
if USE_BF16:
    USE_FP16 = False
GRADIENT_CHECKPOINTING = os.getenv(
    "GRADIENT_CHECKPOINTING",
    "1" if RUNTIME_SAFE_MODE else "0",
).strip().lower() in {"1", "true", "yes"}
FREEZE_BACKBONE = os.getenv(
    "FREEZE_BACKBONE",
    "1" if RUNTIME_SAFE_MODE else "0",
).strip().lower() in {"1", "true", "yes"}
OPTIMIZER = os.getenv(
    "OPTIMIZER",
    "adafactor",
)
EVAL_ACCUMULATION_STEPS = max(
    1,
    int(os.getenv("EVAL_ACCUMULATION_STEPS", "1")),
)
DEFAULT_EVAL_STRATEGY = "epoch"
DEFAULT_SAVE_STRATEGY = "no" if RUNTIME_SAFE_MODE else "epoch"
EVAL_STRATEGY = os.getenv("EVAL_STRATEGY", DEFAULT_EVAL_STRATEGY)
SAVE_STRATEGY = os.getenv("SAVE_STRATEGY", DEFAULT_SAVE_STRATEGY)
LOAD_BEST_MODEL_AT_END = (
    EVAL_STRATEGY != "no"
    and SAVE_STRATEGY != "no"
    and EVAL_STRATEGY == SAVE_STRATEGY
    and os.getenv("LOAD_BEST_MODEL_AT_END", "1").strip().lower() in {"1", "true", "yes"}
)
if RUNTIME_SAFE_MODE:
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    GRAD_ACC_STEPS = max(GRAD_ACC_STEPS, 32)
    GRADIENT_CHECKPOINTING = True
    FREEZE_BACKBONE = True
    OPTIMIZER = "adafactor"
    if "SAMPLE_RATE" not in os.environ:
        SAMPLE_RATE = 8

MAX_CLIP_DURATION_SECONDS = float(
    os.getenv(
        "MAX_CLIP_DURATION_SECONDS",
        "12" if RUNTIME_SAFE_MODE else "120",
    )
)
SAFE_CUDA_MEMORY_FRACTION = float(os.getenv("SAFE_CUDA_MEMORY_FRACTION", "0.9"))
if CUDA_AVAILABLE and RUNTIME_SAFE_MODE:
    try:
        torch.cuda.set_per_process_memory_fraction(SAFE_CUDA_MEMORY_FRACTION, 0)
    except (RuntimeError, ValueError):
        pass

OUTPUT_DIR = pathlib.Path(
    os.getenv(
        "OUTPUT_DIR",
        str(PROJECT_ROOT / "videomae_results" / "videomae"),
    )
)

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow" / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "VideoMAE-UCF-Crime",
)
MLFLOW_RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "videomae-large")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
NORMAL_CLASS_NAME = "Normal_Videos_event"
NORMAL_VIDEO_DIR_CANDIDATES: list[str | pathlib.Path] = [
    "Normal_Videos_for_Event_Recognition",
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
    pathlib.Path("Testing_Normal_Videos", "Testing_Normal_Videos_Anomaly"),
]

VideoLabelPair = tuple[str, dict[str, int]]
VideoLabelPairs = list[VideoLabelPair]
VideoIndex = dict[tuple[str, str], pathlib.Path]


def resolve_dataset_root() -> pathlib.Path:
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate / DATASET_DIR_NAME
    expected = " or ".join(str(path) for path in DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(f"Dataset root not found. Expected under: {expected}")


def resolve_split_file(splits_dir: pathlib.Path, split_file: str) -> pathlib.Path:
    candidate = pathlib.Path(split_file)
    if candidate.is_absolute():
        return candidate
    return splits_dir / candidate


def parse_class_mapping(class_file: pathlib.Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for line in class_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        mapping[parts[0]] = int(parts[1]) - 1
    if not mapping:
        raise ValueError(f"No classes parsed from {class_file}")
    return mapping


def iter_video_files(root_dir: pathlib.Path) -> list[pathlib.Path]:
    if not root_dir.exists():
        return []
    return sorted(
        path
        for path in root_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_anomaly_video_index(dataset_root: pathlib.Path) -> VideoIndex:
    index: VideoIndex = {}
    for part_dir in sorted(dataset_root.glob("Anomaly-Videos-Part-*")):
        for class_dir in sorted(part_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for video_path in iter_video_files(class_dir):
                index[(class_dir.name, video_path.name)] = video_path

    for normal_rel_dir in NORMAL_VIDEO_DIR_CANDIDATES:
        normal_dir = dataset_root / str(normal_rel_dir)
        for video_path in iter_video_files(normal_dir):
            index[(NORMAL_CLASS_NAME, video_path.name)] = video_path

    if not index:
        raise FileNotFoundError(
            f"No anomaly videos found in expected dirs under: {dataset_root}"
        )
    return index


def parse_split_entry(entry: str) -> tuple[str, str] | None:
    class_name, sep, video_name = entry.partition("/")
    if not sep:
        return None
    class_name = class_name.strip()
    video_name = video_name.strip()
    if not class_name or not video_name:
        return None
    return class_name, video_name


def load_video_label_pairs(
    split_file: pathlib.Path,
    class_mapping: dict[str, int],
    video_index: VideoIndex,
) -> VideoLabelPairs:
    pairs: VideoLabelPairs = []
    malformed = 0
    unknown_class = 0
    missing_video = 0

    for raw_line in split_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parsed = parse_split_entry(line)
        if parsed is None:
            malformed += 1
            continue

        class_name, video_name = parsed
        label = class_mapping.get(class_name)
        if label is None:
            unknown_class += 1
            continue

        video_path = video_index.get((class_name, video_name))
        if video_path is None:
            missing_video += 1
            continue

        pairs.append((str(video_path), {"label": label}))

    logger.info(
        "Loaded %d samples from %s (malformed=%d unknown_class=%d missing_video=%d)",
        len(pairs),
        split_file.name,
        malformed,
        unknown_class,
        missing_video,
    )

    if not pairs:
        raise ValueError(f"Split {split_file} produced zero valid samples")
    return pairs


def split_train_val(
    pairs: VideoLabelPairs,
    val_ratio: float,
    seed: int,
) -> tuple[VideoLabelPairs, VideoLabelPairs]:
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if len(pairs) < 2:
        raise ValueError("Need at least 2 samples to split train/val")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_size = min(max(1, val_size), len(shuffled) - 1)
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]
    return train_pairs, val_pairs


def import_pytorchvideo_components() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        data_module = importlib.import_module("pytorchvideo.data")
        transforms_module = importlib.import_module("pytorchvideo.transforms")
    except ModuleNotFoundError as exc:
        missing_module = getattr(exc, "name", "") or ""
        if missing_module != "torchvision.transforms.functional_tensor":
            raise
        functional_tensor = importlib.import_module(
            "torchvision.transforms._functional_tensor"
        )
        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
        data_module = importlib.import_module("pytorchvideo.data")
        transforms_module = importlib.import_module("pytorchvideo.transforms")

    return (
        data_module.LabeledVideoDataset,
        data_module.make_clip_sampler,
        transforms_module.ApplyTransformToKey,
        transforms_module.Normalize,
        transforms_module.RandomShortSideScale,
        transforms_module.UniformTemporalSubsample,
    )


def build_datasets(
    train_pairs: VideoLabelPairs,
    val_pairs: VideoLabelPairs,
    test_pairs: VideoLabelPairs,
    image_processor: VideoMAEImageProcessor,
    model: VideoMAEForVideoClassification,
) -> tuple[Any, Any, Any]:
    tv_transforms = importlib.import_module("torchvision.transforms")
    (
        LabeledVideoDataset,
        make_clip_sampler,
        ApplyTransformToKey,
        Normalize,
        RandomShortSideScale,
        UniformTemporalSubsample,
    ) = import_pytorchvideo_components()

    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        image_size = int(image_processor.size["shortest_edge"])
        resize_to = (image_size, image_size)
    else:
        resize_to = (
            int(image_processor.size["height"]),
            int(image_processor.size["width"]),
        )

    num_frames = int(model.config.num_frames)
    raw_clip_duration = num_frames * SAMPLE_RATE / FPS
    clip_duration = min(raw_clip_duration, MAX_CLIP_DURATION_SECONDS)

    train_short_side_min = 224 if RUNTIME_SAFE_MODE else 256
    train_short_side_max = 256 if RUNTIME_SAFE_MODE else 320

    train_transform = tv_transforms.Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=tv_transforms.Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        tv_transforms.Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(
                            min_size=train_short_side_min,
                            max_size=train_short_side_max,
                        ),
                        tv_transforms.Resize(resize_to),
                        tv_transforms.RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    eval_transform = tv_transforms.Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=tv_transforms.Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        tv_transforms.Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        tv_transforms.Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    train_dataset = LabeledVideoDataset(
        labeled_video_paths=train_pairs,
        clip_sampler=make_clip_sampler("random", clip_duration),
        transform=train_transform,
        decode_audio=False,
    )
    val_dataset = LabeledVideoDataset(
        labeled_video_paths=val_pairs,
        clip_sampler=make_clip_sampler("uniform", clip_duration),
        transform=eval_transform,
        decode_audio=False,
    )
    test_dataset = LabeledVideoDataset(
        labeled_video_paths=test_pairs,
        clip_sampler=make_clip_sampler("uniform", clip_duration),
        transform=eval_transform,
        decode_audio=False,
    )
    logger.info(
        "Temporal sampling requested_rate=%d effective_clip_duration_s=%.2f",
        SAMPLE_RATE,
        clip_duration,
    )

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = np.asarray(eval_pred.label_ids)
    accuracy = float(
        importlib.import_module("sklearn.metrics").accuracy_score(labels, predictions)
    )
    return {"accuracy": accuracy}


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    if CUDA_AVAILABLE and (USE_FP16 or USE_BF16):
        pixel_values = pixel_values.to(torch.bfloat16 if USE_BF16 else torch.float16)
    labels = torch.as_tensor(
        [example["label"] for example in examples], dtype=torch.long
    )
    return {"pixel_values": pixel_values, "labels": labels}


def resolve_max_steps(
    train_dataset: Any, train_pairs_count: int, num_epochs: int
) -> int:
    num_videos = getattr(train_dataset, "num_videos", None)
    if not isinstance(num_videos, int) or num_videos <= 0:
        num_videos = train_pairs_count
    steps_per_epoch = max(1, math.ceil(num_videos / TRAIN_BATCH_SIZE))
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch / GRAD_ACC_STEPS))
    return max(1, optimizer_steps_per_epoch * num_epochs)


def resolve_model_loading_dtype() -> torch.dtype:
    if CUDA_AVAILABLE and USE_BF16:
        return torch.bfloat16
    return torch.float32


def build_training_args(output_dir: pathlib.Path, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        remove_unused_columns=False,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        logging_steps=10,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        fp16=USE_FP16 and CUDA_AVAILABLE,
        bf16=USE_BF16 and CUDA_AVAILABLE,
        fp16_full_eval=USE_FP16 and CUDA_AVAILABLE,
        bf16_full_eval=USE_BF16 and CUDA_AVAILABLE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        optim=OPTIMIZER,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=CUDA_AVAILABLE and not RUNTIME_SAFE_MODE,
        dataloader_persistent_workers=(
            DATALOADER_NUM_WORKERS > 0 and not RUNTIME_SAFE_MODE
        ),
        eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
        max_steps=max_steps,
        seed=SEED,
        report_to=["mlflow"],
        save_total_limit=2,
    )


def summarize_class_counts(
    pairs: VideoLabelPairs, id2label: dict[int, str]
) -> dict[str, int]:
    counts = Counter(payload["label"] for _path, payload in pairs)
    return {
        id2label.get(label_id, str(label_id)): int(count)
        for label_id, count in sorted(counts.items())
    }


def main() -> None:
    set_seed(SEED)

    dataset_root = resolve_dataset_root()
    splits_dir = (
        dataset_root / "UCF_Crimes-Train-Test-Split" / "Action_Regnition_splits"
    )
    class_file = splits_dir / CLASS_IDS_FILE

    train_file = resolve_split_file(splits_dir, TRAIN_SPLIT_FILE)
    test_file = resolve_split_file(splits_dir, TEST_SPLIT_FILE)

    if not class_file.exists():
        raise FileNotFoundError(f"Missing class mapping file: {class_file}")
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train split file: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing test split file: {test_file}")

    class_mapping = parse_class_mapping(class_file)
    id2label = {idx: label for label, idx in class_mapping.items()}

    video_index = build_anomaly_video_index(dataset_root)
    train_pairs_full = load_video_label_pairs(train_file, class_mapping, video_index)
    test_pairs = load_video_label_pairs(test_file, class_mapping, video_index)
    train_pairs, val_pairs = split_train_val(train_pairs_full, VAL_RATIO, SEED)

    logger.info(
        "Sizes -> train=%d val=%d test=%d",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )
    model_dtype = resolve_model_loading_dtype()
    logger.info(
        (
            "Runtime safe_mode=%s vram_gb=%s train_bs=%d eval_bs=%d grad_acc=%d "
            "fp16=%s bf16=%s model_dtype=%s grad_ckpt=%s freeze_backbone=%s "
            "optimizer=%s"
        ),
        RUNTIME_SAFE_MODE,
        f"{DEVICE_VRAM_GB:.2f}" if DEVICE_VRAM_GB is not None else "n/a",
        TRAIN_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        GRAD_ACC_STEPS,
        USE_FP16 and CUDA_AVAILABLE,
        USE_BF16 and CUDA_AVAILABLE,
        str(model_dtype).replace("torch.", ""),
        GRADIENT_CHECKPOINTING,
        FREEZE_BACKBONE,
        OPTIMIZER,
    )

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        label2id=class_mapping,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    if FREEZE_BACKBONE:
        for parameter in model.videomae.parameters():
            parameter.requires_grad = False
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_dataset, val_dataset, test_dataset = build_datasets(
        train_pairs,
        val_pairs,
        test_pairs,
        image_processor,
        model,
    )

    max_steps = resolve_max_steps(train_dataset, len(train_pairs), NUM_EPOCHS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=build_training_args(OUTPUT_DIR, max_steps),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.set_tag("dataset", "UCF-Crime")
        mlflow.set_tag("task", "video-classification")
        mlflow.set_tag("pipeline", "notebook_style")

        mlflow.log_params(
            {
                "model_ckpt": MODEL_CKPT,
                "train_split_file": train_file.name,
                "test_split_file": test_file.name,
                "num_classes": len(class_mapping),
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "warmup_ratio": WARMUP_RATIO,
                "val_ratio": VAL_RATIO,
                "seed": SEED,
                "sample_rate": SAMPLE_RATE,
                "max_clip_duration_seconds": MAX_CLIP_DURATION_SECONDS,
                "fps": FPS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACC_STEPS,
                "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
                "eval_strategy": EVAL_STRATEGY,
                "save_strategy": SAVE_STRATEGY,
                "load_best_model_at_end": LOAD_BEST_MODEL_AT_END,
                "max_steps": max_steps,
                "fp16": USE_FP16 and CUDA_AVAILABLE,
                "bf16": USE_BF16 and CUDA_AVAILABLE,
                "model_loading_dtype": str(model_dtype).replace("torch.", ""),
                "gradient_checkpointing": GRADIENT_CHECKPOINTING,
                "freeze_backbone": FREEZE_BACKBONE,
                "optimizer": OPTIMIZER,
                "memory_safe_mode": MEMORY_SAFE_MODE,
                "runtime_safe_mode": RUNTIME_SAFE_MODE,
                "safe_cuda_memory_fraction": SAFE_CUDA_MEMORY_FRACTION,
                "device_vram_gb": (
                    float(DEVICE_VRAM_GB) if DEVICE_VRAM_GB is not None else -1.0
                ),
                "output_dir": str(OUTPUT_DIR),
            }
        )

        mlflow.log_metrics(
            {
                "train_pairs_total": float(len(train_pairs)),
                "val_pairs_total": float(len(val_pairs)),
                "test_pairs_total": float(len(test_pairs)),
            }
        )

        train_summary = summarize_class_counts(train_pairs, id2label)
        val_summary = summarize_class_counts(val_pairs, id2label)
        test_summary = summarize_class_counts(test_pairs, id2label)
        for class_name, count in train_summary.items():
            mlflow.log_metric(f"train_class_{class_name}", float(count))
        for class_name, count in val_summary.items():
            mlflow.log_metric(f"val_class_{class_name}", float(count))
        for class_name, count in test_summary.items():
            mlflow.log_metric(f"test_class_{class_name}", float(count))

        logger.info("Starting training...")
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        train_result = trainer.train()
        train_metrics = {
            key: float(value)
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float, np.floating))
        }
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        val_metrics = trainer.evaluate(
            eval_dataset=val_dataset, metric_key_prefix="val"
        )
        test_metrics = trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test",
        )

        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in val_metrics.items()
                if isinstance(value, (int, float, np.floating))
            }
        )
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in test_metrics.items()
                if isinstance(value, (int, float, np.floating))
            }
        )

        trainer.save_model(str(OUTPUT_DIR))
        image_processor.save_pretrained(str(OUTPUT_DIR))
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="model")
        mlflow.log_artifact(str(class_file), artifact_path="metadata")
        mlflow.log_artifact(str(train_file), artifact_path="metadata")
        mlflow.log_artifact(str(test_file), artifact_path="metadata")

    logger.info("Training completed. Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
