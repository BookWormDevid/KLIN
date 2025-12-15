import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class FightDetectionDataset(Dataset):
    """
    Multi-dataset loader for fight detection
    Supports: RWF-2000, Anomaly Detection, Violence Detection datasets
    """

    def __init__(self,
                 dataset_paths: Dict[str, str],
                 clip_length: int = 16,
                 frame_rate: int = 10,
                 resolution: Tuple[int, int] = (224, 224),
                 mode: str = 'train',
                 datasets_to_use: List[str] = None):
        """
        Args:
            dataset_paths: Dict with dataset names and paths
                Example: {'rwf': '/path/to/rwf', 'anomaly': '/path/to/anomaly'}
            clip_length: Number of frames per clip
            frame_rate: Target FPS for frame sampling
            resolution: Target frame resolution (H, W)
            mode: 'train', 'val', or 'test'
            datasets_to_use: Which datasets to include (None = all)
        """
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.mode = mode

        # Define datasets mapping
        self.dataset_configs = {
            'rwf': {
                'path': dataset_paths.get('rwf'),
                'classes': {'Fight': 1, 'NonFight': 0},
                'structure': 'class_folders'  # Fight/NonFight folders
            },
            'anomaly': {
                'path': dataset_paths.get('anomaly'),
                'classes': {'Fighting': 1, 'Normal_Videos_event': 0},
                'structure': 'class_folders'  # Individual class folders
            },
            'violence': {
                'path': dataset_paths.get('violence'),
                'classes': {'Violent': 1, 'nonviolent': 0},
                'structure': 'class_folders'
            }
        }

        # Filter datasets if specified
        if datasets_to_use:
            self.dataset_configs = {k: v for k, v in self.dataset_configs.items()
                                    if k in datasets_to_use and v['path'] is not None}

        # Build samples list
        self.samples = self._build_samples()

        # Data augmentation
        self.transform = self._get_transforms()

        print(f"Loaded {len(self.samples)} samples from datasets: {list(self.dataset_configs.keys())}")

    def _build_samples(self) -> List[Dict]:
        """Build unified list of samples from all datasets"""
        samples = []

        for dataset_name, config in self.dataset_configs.items():
            dataset_path = config['path']
            if not os.path.exists(dataset_path):
                print(f"Warning: {dataset_name} path {dataset_path} does not exist")
                continue

            if config['structure'] == 'class_folders':
                samples.extend(self._load_class_folder_structure(dataset_name, dataset_path, config['classes']))

        # Shuffle samples
        random.shuffle(samples)
        return samples


    def _load_class_folder_structure(self, dataset_name: str, base_path: str,
                                     class_mapping: Dict[str, int]) -> List[Dict]:
        """Load datasets organized in class folders"""
        samples = []

        for class_name, label in class_mapping.items():
            class_path = os.path.join(base_path, class_name)

            if not os.path.exists(class_path):
                print(f"Warning: Class path {class_path} does not exist")
                continue

            # Support multiple video extensions
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

            for video_file in os.listdir(class_path):
                if any(video_file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(class_path, video_file)

                    sample = {
                        'video_path': video_path,
                        'label': label,
                        'dataset': dataset_name,
                        'class_name': class_name
                    }
                    samples.append(sample)

        print(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples

    def _get_transforms(self):
        """Get data augmentation transforms for train/val modes"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.resolution[0], self.resolution[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.resolution[0], self.resolution[1]),
                A.Normalize(),
                ToTensorV2(),
            ])

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load and sample frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate sampling interval
        if original_fps <= 0:
            original_fps = 30  # fallback

        sample_interval = max(1, int(original_fps / self.frame_rate))

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames according to target FPS
            if frame_count % sample_interval == 0:
                # Convert BGR to RGB and grayscale to 3-channel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if len(frame.shape) == 2:  # Grayscale
                    frame = np.stack([frame] * 3, axis=-1)
                frames.append(frame)

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        return frames

    def _sample_clip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample a clip of specified length from frames"""
        if len(frames) <= self.clip_length:
            # Pad with last frame if video is too short
            padded_frames = frames + [frames[-1]] * (self.clip_length - len(frames))
            return padded_frames[:self.clip_length]
        else:
            # Random sampling for training, center sampling for validation
            if self.mode == 'train':
                start_idx = random.randint(0, len(frames) - self.clip_length)
            else:
                start_idx = (len(frames) - self.clip_length) // 2

            return frames[start_idx:start_idx + self.clip_length]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        try:
            # Load and sample frames
            frames = self._load_video_frames(sample['video_path'])
            clip_frames = self._sample_clip(frames)

            # Apply transforms to each frame
            transformed_frames = []
            for frame in clip_frames:
                transformed = self.transform(image=frame)['image']
                transformed_frames.append(transformed)

            # Stack frames: [C, T, H, W]
            video_tensor = torch.stack(transformed_frames, dim=1)

            label = torch.tensor(sample['label'], dtype=torch.float32)

            return video_tensor, label

        except Exception as e:
            print(f"Error loading sample {sample['video_path']}: {e}")
            # Return a dummy sample (you might want better error handling)
            dummy_video = torch.zeros(3, self.clip_length, self.resolution[0], self.resolution[1])
            dummy_label = torch.tensor(0.0, dtype=torch.float32)
            return dummy_video, dummy_label


class DataLoaderFactory:
    """Factory for creating data loaders with different configurations"""

    @staticmethod
    def create_loaders(dataset_paths: Dict[str, str],
                       batch_size: int = 32,
                       clip_length: int = 16,
                       frame_rate: int = 10,
                       resolution: Tuple[int, int] = (224, 224),
                       train_ratio: float = 0.8,
                       num_workers: int = 4):
        """
        Create train and validation data loaders
        """
        # Create full dataset
        full_dataset = FightDetectionDataset(
            dataset_paths=dataset_paths,
            clip_length=clip_length,
            frame_rate=frame_rate,
            resolution=resolution,
            mode='train'  # We'll split manually
        )

        # Split dataset
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Set modes
        train_dataset.dataset.mode = 'train'
        val_dataset.dataset.mode = 'val'

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader
