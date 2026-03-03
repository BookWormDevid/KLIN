import importlib
import logging
import math
import os
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
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
    PROJECT_ROOT / "data" / "raw" / "klin",
    PROJECT_ROOT / "data" / "raw" / "KLIN",
]
DATASET_DIR_NAME = "Anomaly-detection-dataset"
SPLITS_DIR_REL = pathlib.Path(
    DATASET_DIR_NAME,
    "UCF_Crimes-Train-Test-Split",
    "Action_Regnition_splits",
)
CLASS_IDS_FILE = "ClassIDs.txt"

LOCAL_MODEL_CKPT = PROJECT_ROOT / "models" / "videomae-large"
MODEL_CKPT = (
    str(LOCAL_MODEL_CKPT) if LOCAL_MODEL_CKPT.exists() else "MCG-NJU/videomae-large"
)
OUTPUT_DIR = PROJECT_ROOT / "videomae_results" / "videomae-UCF-crime"

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "30"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.1"))
SEED = int(os.getenv("SEED", "42"))

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "4"))
FPS = float(os.getenv("FPS", "30"))
TEMPORAL_MODE = os.getenv("TEMPORAL_MODE", "hybrid").strip().lower()
GLOBAL_SAMPLE_RATE = int(os.getenv("GLOBAL_SAMPLE_RATE", "64"))
EVAL_CLIPS_PER_VIDEO = max(1, int(os.getenv("EVAL_CLIPS_PER_VIDEO", "2")))

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
NORMAL_CLASS_NAME = "Normal_Videos_event"
INCLUDE_NORMAL_CLASS = os.getenv("INCLUDE_NORMAL_CLASS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
NORMAL_VIDEO_DIR_CANDIDATES: list[str | pathlib.Path] = [
    "Normal_Videos_for_Event_Recognition",
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
    pathlib.Path("Testing_Normal_Videos", "Testing_Normal_Videos_Anomaly"),
]

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow" / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "VideoMAE-UCF-Crime-All-Action-Splits",
)
MLFLOW_RUN_NAME = os.getenv(
    "MLFLOW_RUN_NAME",
    "videomae-large-action-recognition-all-splits",
)

FOLD_RE = re.compile(r"^(train|test)_(\d+)\.txt$")

VideoLabelItem = tuple[str, dict[Any, Any] | None]
VideoLabelPairs = list[VideoLabelItem]
VideoIndex = dict[tuple[str, str], pathlib.Path]


@dataclass(frozen=True)
class RuntimeConfig:
    train_batch_size: int
    eval_batch_size: int
    grad_accum_steps: int
    eval_accumulation_steps: int
    dataloader_num_workers: int
    gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    optimizer: str
    device_name: str
    total_vram_gb: float | None


@dataclass(frozen=True)
class TrainingContext:
    class_mapping: dict[str, int]
    id2label: dict[int, str]
    video_index: VideoIndex
    image_processor: VideoMAEImageProcessor
    runtime_config: RuntimeConfig


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_runtime_config() -> RuntimeConfig:
    cpu_count = os.cpu_count() or 2
    default_workers = min(4, max(1, cpu_count // 2))

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        device_name = props.name

        if total_vram_gb <= 8.5:
            train_batch_default = 1
            eval_batch_default = 1
            grad_acc_default = 8
            eval_acc_steps_default = 8
        elif total_vram_gb <= 12.5:
            train_batch_default = 1
            eval_batch_default = 1
            grad_acc_default = 4
            eval_acc_steps_default = 4
        else:
            train_batch_default = 2
            eval_batch_default = 2
            grad_acc_default = 2
            eval_acc_steps_default = 2

        bf16_enabled = torch.cuda.is_bf16_supported() and env_flag("USE_BF16", True)
        fp16_enabled = (not bf16_enabled) and env_flag("USE_FP16", True)
        optimizer_default = "adamw_torch_fused"
        gradient_checkpointing_default = True
    else:
        total_vram_gb = None
        device_name = "cpu"
        train_batch_default = 1
        eval_batch_default = 1
        grad_acc_default = 1
        eval_acc_steps_default = 1
        bf16_enabled = False
        fp16_enabled = False
        optimizer_default = "adamw_torch"
        gradient_checkpointing_default = False

    train_batch_size = max(
        1,
        int(os.getenv("TRAIN_BATCH_SIZE", str(train_batch_default))),
    )
    eval_batch_size = max(
        1,
        int(os.getenv("EVAL_BATCH_SIZE", str(eval_batch_default))),
    )
    grad_accum_steps = max(
        1,
        int(os.getenv("GRAD_ACC_STEPS", str(grad_acc_default))),
    )
    eval_accumulation_steps = max(
        1,
        int(os.getenv("EVAL_ACCUMULATION_STEPS", str(eval_acc_steps_default))),
    )
    dataloader_num_workers = max(
        0,
        int(os.getenv("DATALOADER_NUM_WORKERS", str(default_workers))),
    )

    return RuntimeConfig(
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        grad_accum_steps=grad_accum_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        gradient_checkpointing=env_flag(
            "GRADIENT_CHECKPOINTING",
            gradient_checkpointing_default,
        ),
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        optimizer=os.getenv("TRAIN_OPTIMIZER", optimizer_default),
        device_name=device_name,
        total_vram_gb=total_vram_gb,
    )


def configure_torch_backend() -> None:
    if torch.cuda.is_available() and env_flag("ENABLE_TF32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def get_torchvision_transforms() -> Any:
    return importlib.import_module("torchvision.transforms")


def sklearn_train_test_split(
    data: VideoLabelPairs,
    test_size: float,
    random_state: int,
    stratify: list[int] | None,
) -> tuple[VideoLabelPairs, VideoLabelPairs]:
    model_selection = importlib.import_module("sklearn.model_selection")
    train_pairs, val_pairs = model_selection.train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_pairs), list(val_pairs)


def sklearn_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metrics_module = importlib.import_module("sklearn.metrics")
    return float(metrics_module.accuracy_score(y_true, y_pred))


def resolve_klin_root() -> pathlib.Path:
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    expected = " or ".join(str(path) for path in DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(f"Could not find dataset root. Expected: {expected}")


def parse_class_mapping(
    class_file: pathlib.Path,
    include_normal_class: bool,
) -> dict[str, int]:
    raw_mapping: dict[str, int] = {}
    for line in class_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        raw_mapping[parts[0]] = int(parts[1]) - 1

    if not raw_mapping:
        raise ValueError(f"No class ids parsed from {class_file}")

    filtered = sorted(raw_mapping.items(), key=lambda item: item[1])
    if not include_normal_class:
        filtered = [item for item in filtered if item[0] != NORMAL_CLASS_NAME]

    remapped = {class_name: idx for idx, (class_name, _old_id) in enumerate(filtered)}

    if not remapped:
        raise ValueError("Class mapping is empty after filtering")
    return remapped


def iter_video_files(root_dir: pathlib.Path) -> list[pathlib.Path]:
    if not root_dir.exists():
        return []
    return sorted(
        path
        for path in root_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_video_index(dataset_root: pathlib.Path) -> VideoIndex:
    index: VideoIndex = {}

    anomaly_roots = sorted(dataset_root.glob("Anomaly-Videos-Part-*"))
    for anomaly_root in anomaly_roots:
        for class_dir in sorted(anomaly_root.iterdir()):
            if not class_dir.is_dir():
                continue
            for video_path in iter_video_files(class_dir):
                key = (class_dir.name, video_path.name)
                index.setdefault(key, video_path)

    for normal_rel_dir in NORMAL_VIDEO_DIR_CANDIDATES:
        normal_dir = dataset_root / str(normal_rel_dir)
        for video_path in iter_video_files(normal_dir):
            key = (NORMAL_CLASS_NAME, video_path.name)
            index.setdefault(key, video_path)

    if not index:
        raise FileNotFoundError(
            f"No videos found under expected dataset root: {dataset_root}"
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

    labels: list[int] = []
    for _video_path, payload in pairs:
        if not payload or "label" not in payload:
            raise ValueError("Each sample must include an integer 'label'")
        labels.append(int(payload["label"]))

    class_counts = Counter(labels)
    use_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2
    stratify = labels if use_stratify else None

    try:
        train_pairs, val_pairs = sklearn_train_test_split(
            data=pairs,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_pairs, val_pairs = sklearn_train_test_split(
            data=pairs,
            test_size=val_ratio,
            random_state=seed,
            stratify=None,
        )

    return train_pairs, val_pairs


def import_pytorchvideo_components() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        data_module = importlib.import_module("pytorchvideo.data")
        transforms_module = importlib.import_module("pytorchvideo.transforms")
    except ModuleNotFoundError as exc:
        missing_module = getattr(exc, "name", "") or ""
        if missing_module != "torchvision.transforms.functional_tensor":
            raise ModuleNotFoundError(
                "pytorchvideo is required for this training pipeline. "
                "Install it with `uv add pytorchvideo` or `pip install pytorchvideo`."
            ) from exc

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


def discover_fold_split_files(
    splits_dir: pathlib.Path,
) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    train_files: dict[str, pathlib.Path] = {}
    test_files: dict[str, pathlib.Path] = {}

    for file_path in sorted(splits_dir.glob("*.txt")):
        match = FOLD_RE.match(file_path.name)
        if not match:
            continue
        split_type, fold_id = match.group(1), match.group(2)
        if split_type == "train":
            train_files[fold_id] = file_path
        else:
            test_files[fold_id] = file_path

    fold_ids = sorted(set(train_files) & set(test_files))
    if not fold_ids:
        raise FileNotFoundError(
            f"No complete train/test fold pairs found in {splits_dir}"
        )

    return [
        (fold_id, train_files[fold_id], test_files[fold_id]) for fold_id in fold_ids
    ]


def resolve_temporal_sample_rates() -> tuple[int, int]:
    train_sample_rate = max(1, SAMPLE_RATE)
    global_sample_rate = max(1, GLOBAL_SAMPLE_RATE)

    if TEMPORAL_MODE == "local":
        return train_sample_rate, train_sample_rate
    if TEMPORAL_MODE == "global":
        return global_sample_rate, global_sample_rate
    if TEMPORAL_MODE == "hybrid":
        return train_sample_rate, max(train_sample_rate, global_sample_rate)

    raise ValueError(
        f"Unsupported TEMPORAL_MODE={TEMPORAL_MODE}. Use local, global, or hybrid."
    )


def build_eval_clip_sampler(
    make_clip_sampler: Any,
    eval_clip_duration: float,
) -> Any:
    try:
        return make_clip_sampler(
            "constant_clips_per_video",
            eval_clip_duration,
            EVAL_CLIPS_PER_VIDEO,
        )
    except Exception:
        return make_clip_sampler("uniform", eval_clip_duration)


def build_datasets(
    train_pairs: VideoLabelPairs,
    val_pairs: VideoLabelPairs,
    test_pairs: VideoLabelPairs,
    image_processor: VideoMAEImageProcessor,
    model: VideoMAEForVideoClassification,
) -> tuple[Any, Any, Any]:
    tv_transforms = get_torchvision_transforms()
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
    train_sample_rate, eval_sample_rate = resolve_temporal_sample_rates()
    train_clip_duration = num_frames * train_sample_rate / FPS
    eval_clip_duration = num_frames * eval_sample_rate / FPS

    def make_transform(is_train: bool) -> Any:
        ops: list[Any] = [
            UniformTemporalSubsample(num_frames),
            tv_transforms.Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
        ]
        if is_train:
            ops.extend(
                [
                    RandomShortSideScale(min_size=256, max_size=320),
                    tv_transforms.Resize(resize_to),
                    tv_transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            ops.append(tv_transforms.Resize(resize_to))

        return tv_transforms.Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=tv_transforms.Compose(ops),
                )
            ]
        )

    train_transform = make_transform(is_train=True)
    eval_transform = make_transform(is_train=False)

    train_dataset = LabeledVideoDataset(
        labeled_video_paths=train_pairs,
        clip_sampler=make_clip_sampler("random", train_clip_duration),
        transform=train_transform,
        decode_audio=False,
    )
    eval_sampler = build_eval_clip_sampler(make_clip_sampler, eval_clip_duration)
    val_dataset = LabeledVideoDataset(
        labeled_video_paths=val_pairs,
        clip_sampler=eval_sampler,
        transform=eval_transform,
        decode_audio=False,
    )
    test_dataset = LabeledVideoDataset(
        labeled_video_paths=test_pairs,
        clip_sampler=build_eval_clip_sampler(make_clip_sampler, eval_clip_duration),
        transform=eval_transform,
        decode_audio=False,
    )
    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = np.asarray(eval_pred.label_ids)
    accuracy = sklearn_accuracy(labels, predictions)
    return {"accuracy": float(accuracy)}


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.as_tensor(
        [example["label"] for example in examples], dtype=torch.long
    )
    return {"pixel_values": pixel_values, "labels": labels}


def build_training_args(
    fold_output_dir: pathlib.Path,
    runtime_config: RuntimeConfig,
    train_size: int,
) -> TrainingArguments:
    effective_batch = runtime_config.train_batch_size * runtime_config.grad_accum_steps
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch))
    max_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(max_steps * WARMUP_RATIO)

    return TrainingArguments(
        output_dir=str(fold_output_dir),
        gradient_accumulation_steps=runtime_config.grad_accum_steps,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=runtime_config.train_batch_size,
        per_device_eval_batch_size=runtime_config.eval_batch_size,
        fp16=runtime_config.fp16,
        bf16=runtime_config.bf16,
        gradient_checkpointing=runtime_config.gradient_checkpointing,
        optim=runtime_config.optimizer,
        dataloader_num_workers=runtime_config.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=runtime_config.dataloader_num_workers > 0,
        eval_accumulation_steps=runtime_config.eval_accumulation_steps,
        max_steps=max_steps,
        seed=SEED,
        report_to=["mlflow"],
        save_total_limit=2,
    )


def build_model(context: TrainingContext) -> VideoMAEForVideoClassification:
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        label2id=context.class_mapping,
        id2label=context.id2label,
        ignore_mismatched_sizes=True,
    )
    if context.runtime_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def extract_numeric_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            numeric[f"{prefix}_{key}"] = float(value)
    return numeric


def train_one_fold(
    fold_id: str,
    train_file: pathlib.Path,
    test_file: pathlib.Path,
    context: TrainingContext,
) -> dict[str, float]:
    train_pairs = load_video_label_pairs(
        train_file,
        context.class_mapping,
        context.video_index,
    )
    test_pairs = load_video_label_pairs(
        test_file,
        context.class_mapping,
        context.video_index,
    )
    train_pairs, val_pairs = split_train_val(train_pairs, VAL_RATIO, SEED)

    logger.info(
        "Fold %s sizes -> train=%d val=%d test=%d",
        fold_id,
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )

    model = build_model(context)

    train_dataset, val_dataset, test_dataset = build_datasets(
        train_pairs,
        val_pairs,
        test_pairs,
        context.image_processor,
        model,
    )

    fold_output_dir = OUTPUT_DIR / f"fold_{fold_id}"
    trainer = Trainer(
        model=model,
        args=build_training_args(
            fold_output_dir,
            context.runtime_config,
            train_size=len(train_pairs),
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=context.image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    with mlflow.start_run(run_name=f"fold_{fold_id}", nested=True):
        mlflow.set_tag("fold_id", fold_id)
        mlflow.log_params(
            {
                "fold_id": fold_id,
                "train_split_file": train_file.name,
                "test_split_file": test_file.name,
                "num_classes": len(context.class_mapping),
                "train_size": len(train_pairs),
                "val_size": len(val_pairs),
                "test_size": len(test_pairs),
            }
        )

        logger.info("Starting training for fold %s...", fold_id)
        train_result = trainer.train()
        mlflow.log_metrics(extract_numeric_metrics(train_result.metrics, "train"))

        val_metrics = trainer.evaluate(eval_dataset=val_dataset)
        test_metrics = trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test",
        )
        mlflow.log_metrics(extract_numeric_metrics(val_metrics, "val"))
        mlflow.log_metrics(extract_numeric_metrics(test_metrics, "test"))

        fold_output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(fold_output_dir))
        context.image_processor.save_pretrained(str(fold_output_dir))
        mlflow.log_artifacts(
            str(fold_output_dir),
            artifact_path=f"models/fold_{fold_id}",
        )

    logger.info("Fold %s val metrics: %s", fold_id, val_metrics)
    logger.info("Fold %s test metrics: %s", fold_id, test_metrics)

    return {
        "fold_val_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
        "fold_test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
    }


def main() -> None:
    configure_torch_backend()
    set_seed(SEED)

    klin_root = resolve_klin_root()
    dataset_root = klin_root / DATASET_DIR_NAME
    splits_dir = klin_root / SPLITS_DIR_REL

    class_file = splits_dir / CLASS_IDS_FILE
    if not class_file.exists():
        raise FileNotFoundError(f"Missing class mapping file: {class_file}")

    fold_split_files = discover_fold_split_files(splits_dir)
    class_mapping = parse_class_mapping(
        class_file,
        include_normal_class=INCLUDE_NORMAL_CLASS,
    )
    id2label = {idx: label for label, idx in class_mapping.items()}
    video_index = build_video_index(dataset_root)
    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    runtime_config = resolve_runtime_config()
    train_sample_rate, eval_sample_rate = resolve_temporal_sample_rates()
    precision_mode = (
        "bf16" if runtime_config.bf16 else "fp16" if runtime_config.fp16 else "fp32"
    )
    context = TrainingContext(
        class_mapping=class_mapping,
        id2label=id2label,
        video_index=video_index,
        image_processor=image_processor,
        runtime_config=runtime_config,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    logger.info(
        "Found %d folds: %s",
        len(fold_split_files),
        ", ".join(fold_id for fold_id, _train, _test in fold_split_files),
    )
    logger.info(
        "Runtime device=%s vram_gb=%s train_bs=%d eval_bs=%d grad_acc=%d precision=%s",
        runtime_config.device_name,
        f"{runtime_config.total_vram_gb:.2f}"
        if runtime_config.total_vram_gb is not None
        else "n/a",
        runtime_config.train_batch_size,
        runtime_config.eval_batch_size,
        runtime_config.grad_accum_steps,
        precision_mode,
    )
    logger.info(
        (
            "Temporal mode=%s train_sample_rate=%d eval_sample_rate=%d "
            "eval_clips_per_video=%d"
        ),
        TEMPORAL_MODE,
        train_sample_rate,
        eval_sample_rate,
        EVAL_CLIPS_PER_VIDEO,
    )

    fold_results: list[dict[str, float]] = []
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.set_tag("dataset", "UCF-Crime")
        mlflow.set_tag("task", "video-classification")
        mlflow.set_tag(
            "splits_mode",
            "all_action_recognition_splits",
        )
        mlflow.log_params(
            {
                "model_ckpt": MODEL_CKPT,
                "include_normal_class": INCLUDE_NORMAL_CLASS,
                "num_classes": len(class_mapping),
                "num_folds": len(fold_split_files),
                "train_batch_size": runtime_config.train_batch_size,
                "eval_batch_size": runtime_config.eval_batch_size,
                "num_epochs": NUM_EPOCHS,
                "grad_accum_steps": runtime_config.grad_accum_steps,
                "learning_rate": LEARNING_RATE,
                "warmup_ratio": WARMUP_RATIO,
                "val_ratio": VAL_RATIO,
                "seed": SEED,
                "sample_rate": SAMPLE_RATE,
                "temporal_mode": TEMPORAL_MODE,
                "train_sample_rate": train_sample_rate,
                "eval_sample_rate": eval_sample_rate,
                "global_sample_rate": GLOBAL_SAMPLE_RATE,
                "eval_clips_per_video": EVAL_CLIPS_PER_VIDEO,
                "fps": FPS,
                "gradient_checkpointing": runtime_config.gradient_checkpointing,
                "precision_mode": precision_mode,
                "optimizer": runtime_config.optimizer,
                "dataloader_num_workers": runtime_config.dataloader_num_workers,
                "eval_accumulation_steps": runtime_config.eval_accumulation_steps,
                "device_name": runtime_config.device_name,
                "total_vram_gb": runtime_config.total_vram_gb,
                "output_dir": str(OUTPUT_DIR),
            }
        )

        for fold_id, train_file, test_file in fold_split_files:
            fold_metrics = train_one_fold(
                fold_id=fold_id,
                train_file=train_file,
                test_file=test_file,
                context=context,
            )
            fold_results.append(fold_metrics)
            mlflow.log_metrics(
                {
                    f"fold_{fold_id}_val_accuracy": fold_metrics["fold_val_accuracy"],
                    f"fold_{fold_id}_test_accuracy": fold_metrics["fold_test_accuracy"],
                }
            )

        val_scores = [item["fold_val_accuracy"] for item in fold_results]
        test_scores = [item["fold_test_accuracy"] for item in fold_results]

        if val_scores:
            mlflow.log_metric("cv_val_accuracy_mean", float(np.mean(val_scores)))
            mlflow.log_metric("cv_val_accuracy_std", float(np.std(val_scores)))
        if test_scores:
            mlflow.log_metric("cv_test_accuracy_mean", float(np.mean(test_scores)))
            mlflow.log_metric("cv_test_accuracy_std", float(np.std(test_scores)))

    logger.info("Completed training across %d folds", len(fold_split_files))


if __name__ == "__main__":
    main()
