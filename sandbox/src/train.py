import logging
import os
import pathlib
import random
import re
from typing import Any

import numpy as np
import torch
from torchvision.transforms import (  # type: ignore[import-untyped]
    Compose,
    Lambda,
    RandomHorizontalFlip,
    Resize,
)
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
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

BATCH_SIZE = 4
NUM_EPOCHS = 30
GRAD_ACC_STEPS = 2
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
VAL_RATIO = 0.1
SEED = 42

SAMPLE_RATE = 800
FPS = 30

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
NORMAL_CLASS_NAME = "Normal_Videos_event"
# Default mode includes Normal class (14 classes); override via INCLUDE_NORMAL_CLASS.
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

    # Reindex classes to contiguous ids 0..N-1 for stable CE loss.
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
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]
    return train_pairs, val_pairs


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


def build_datasets(
    train_pairs: VideoLabelPairs,
    val_pairs: VideoLabelPairs,
    test_pairs: VideoLabelPairs,
    image_processor: VideoMAEImageProcessor,
    model: VideoMAEForVideoClassification,
) -> tuple[Any, Any, Any]:
    try:
        from pytorchvideo.data import (  # type: ignore[import-untyped]
            LabeledVideoDataset,
            make_clip_sampler,
        )
        from pytorchvideo.transforms import (  # type: ignore[import-untyped]
            ApplyTransformToKey,
            Normalize,
            RandomShortSideScale,
            UniformTemporalSubsample,
        )
    except ModuleNotFoundError as exc:
        missing_module = getattr(exc, "name", "") or ""
        if missing_module == "torchvision.transforms.functional_tensor":
            import sys

            try:
                import torchvision.transforms._functional_tensor as _functional_tensor  # type: ignore[import-untyped]
            except ModuleNotFoundError as shim_exc:
                raise ModuleNotFoundError(
                    "pytorchvideo is installed, but current torchvision is not "
                    "compatible and no fallback tensor functional API was found."
                ) from shim_exc

            sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor
            logger.warning(
                "Applied torchvision compatibility shim for pytorchvideo "
                "(functional_tensor -> _functional_tensor)."
            )
            from pytorchvideo.data import (  # type: ignore[import-untyped]
                LabeledVideoDataset,
                make_clip_sampler,
            )
            from pytorchvideo.transforms import (  # type: ignore[import-untyped]
                ApplyTransformToKey,
                Normalize,
                RandomShortSideScale,
                UniformTemporalSubsample,
            )
        else:
            raise ModuleNotFoundError(
                "pytorchvideo is required for this training pipeline. "
                "Install it with `uv add pytorchvideo` or `pip install pytorchvideo`."
            ) from exc

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
    clip_duration = num_frames * SAMPLE_RATE / FPS

    def make_transform(is_train: bool) -> Compose:
        ops: list[Any] = [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
        ]
        if is_train:
            ops.extend(
                [
                    RandomShortSideScale(min_size=256, max_size=320),
                    Resize(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            ops.append(Resize(resize_to))

        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(ops),
                )
            ]
        )

    train_transform = make_transform(is_train=True)
    eval_transform = make_transform(is_train=False)

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
    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = float(np.mean(predictions == eval_pred.label_ids))
    return {"accuracy": accuracy}


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def build_training_args(
    train_size: int,
    fold_output_dir: pathlib.Path,
) -> TrainingArguments:
    max_steps = max(1, (train_size // BATCH_SIZE) * NUM_EPOCHS)
    return TrainingArguments(
        output_dir=str(fold_output_dir),
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        max_steps=max_steps,
        seed=SEED,
        report_to=["mlflow"],
        save_total_limit=2,
    )


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
    class_mapping: dict[str, int],
    video_index: VideoIndex,
) -> dict[str, float]:
    train_pairs = load_video_label_pairs(train_file, class_mapping, video_index)
    test_pairs = load_video_label_pairs(test_file, class_mapping, video_index)
    train_pairs, val_pairs = split_train_val(train_pairs, VAL_RATIO, SEED)

    logger.info(
        "Fold %s sizes -> train=%d val=%d test=%d",
        fold_id,
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    id2label = {idx: label for label, idx in class_mapping.items()}
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        label2id=class_mapping,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    train_dataset, val_dataset, test_dataset = build_datasets(
        train_pairs,
        val_pairs,
        test_pairs,
        image_processor,
        model,
    )

    fold_output_dir = OUTPUT_DIR / f"fold_{fold_id}"
    trainer = Trainer(
        model=model,
        args=build_training_args(len(train_pairs), fold_output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
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
                "num_classes": len(class_mapping),
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
        image_processor.save_pretrained(str(fold_output_dir))
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
    video_index = build_video_index(dataset_root)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    logger.info(
        "Found %d folds: %s",
        len(fold_split_files),
        ", ".join(fold_id for fold_id, _train, _test in fold_split_files),
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
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "grad_accum_steps": GRAD_ACC_STEPS,
                "learning_rate": LEARNING_RATE,
                "warmup_ratio": WARMUP_RATIO,
                "val_ratio": VAL_RATIO,
                "seed": SEED,
                "output_dir": str(OUTPUT_DIR),
            }
        )

        for fold_id, train_file, test_file in fold_split_files:
            fold_metrics = train_one_fold(
                fold_id=fold_id,
                train_file=train_file,
                test_file=test_file,
                class_mapping=class_mapping,
                video_index=video_index,
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
