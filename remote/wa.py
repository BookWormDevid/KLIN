import warnings
import torch

# Подавить все предупреждения IterableDataset
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
import pathlib
import os
import numpy as np
import pytorchvideo
from pytorchvideo import data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
import evaluated


dataset_root_path = "C:/Users/DEvA/Videos/Video_For_AI/klin_processed"
dataset_root_path = pathlib.Path(dataset_root_path)


class_labels = ['violent', 'nonviolent']
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(label2id, id2label)

model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt, local_files_only=True)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    local_files_only=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model.device)

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps




train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

train_dataset = pytorchvideo.data.labeled_video_dataset(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)



val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

val_dataset = pytorchvideo.data.labeled_video_dataset(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.labeled_video_dataset(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)


# Add this after creating your datasets
def check_dataset(dataset, name):
    print(f"\n{name} dataset info:")
    print(f"Number of videos: {dataset.num_videos}")
    try:
        sample = next(iter(dataset))
        print(f"Sample keys: {sample.keys()}")
        print(f"Video shape: {sample['video'].shape}")

    except Exception as e:
        print(f"Error loading sample from {name}: {e}")

check_dataset(train_dataset, "Train")
check_dataset(val_dataset, "Validation")
check_dataset(test_dataset, "Test")


model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-klin"
num_epochs = 8

args = TrainingArguments(
    output_dir=new_model_name,
    remove_unused_columns=False,
    learning_rate=1e-5,
    eval_strategy='epoch',
    save_strategy='epoch',
    adam_epsilon=1e-8,
    adam_beta1=0.9,
    adam_beta2=0.999,
    push_to_hub=False,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    lr_scheduler_type='linear',
    dataloader_drop_last=False,
    dataloader_pin_memory=False,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
)



metric = evaluated.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}




trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,

)
train_results = trainer.train()