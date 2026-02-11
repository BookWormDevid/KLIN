import os
import pathlib
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

BASE_DIR = pathlib.Path(__file__).parent.parent


class VideoFolderClassifier:
    def __init__(
        self,
        model_path: str | None = None,
        chunk_size: int = 16,
        frame_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not model_path:
            model_path = self._find_model_path()

        self.processor: VideoMAEImageProcessor = VideoMAEImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        model: VideoMAEForVideoClassification = (
            VideoMAEForVideoClassification.from_pretrained(
                model_path, local_files_only=True
            )
        )
        self.model = model.to(self.device)  # type: ignore
        self.model.eval()

    def _find_model_path(self) -> str:
        possible_paths = [os.path.join(BASE_DIR, "models", "videomae-large")]
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(
                os.path.join(path, "config.json")
            ):
                return path
        raise FileNotFoundError("model not found")

    def _read_video_frames(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        frames: list[np.ndarray] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = min(total_frames, 1000)
        for _ in range(limit):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            frames.append(frame_resized)
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"no frames in video: {video_path}")
        return np.array(frames, dtype=np.uint8)

    def _chunk_frames(self, frames: np.ndarray) -> np.ndarray:
        t = len(frames)
        padding_needed = (-t) % self.chunk_size
        if padding_needed > 0:
            padding = np.zeros((padding_needed, *self.frame_size, 3), dtype=np.uint8)
            frames = np.concatenate([frames, padding], axis=0)
        num_chunks = len(frames) // self.chunk_size
        return frames.reshape(num_chunks, self.chunk_size, *self.frame_size, 3)

    def predict_video(self, video_path: str, batch_size: int = 4) -> dict[str, Any]:
        try:
            frames = self._read_video_frames(video_path)
            chunks = self._chunk_frames(frames)
            all_predictions: list[torch.Tensor] = []

            with torch.no_grad():
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i : i + batch_size]

                    # Формируем список списков кадров в формате uint8
                    images = [
                        [
                            frame.astype(np.uint8) for frame in clip
                        ]  # clip shape: (chunk_size, H, W, 3)
                        for clip in batch_chunks
                    ]

                    # Проверка форм
                    for clip in images:
                        assert len(clip) == self.chunk_size, (
                            f"chunk size mismatch: {len(clip)}"
                        )
                        for frame in clip:
                            assert frame.shape[:2] == self.frame_size, (
                                f"frame size mismatch: {frame.shape[:2]}"
                            )
                            assert frame.shape[2] == 3, (
                                f"frame channels mismatch: {frame.shape[2]}"
                            )

                    # Подготовка входа для модели
                    inputs = self.processor(images, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    all_predictions.append(outputs.logits.cpu())

            if not all_predictions:
                raise RuntimeError("no predictions produced")

            # Усреднение логитов по всем чанкам
            logits = torch.cat(all_predictions, dim=0)
            final_logits = torch.mean(logits, dim=0)
            probabilities = torch.nn.functional.softmax(final_logits, dim=0)
            predicted_idx = int(final_logits.argmax().item())
            confidence = float(probabilities[predicted_idx].item())

            id2label = getattr(self.model.config, "id2label", None) or {}
            predicted_class = id2label.get(predicted_idx, str(predicted_idx))

            return {
                "video_name": os.path.basename(video_path),
                "video_path": video_path,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "num_frames": len(frames),
                "num_chunks": len(chunks),
            }

        except Exception as e:
            return {
                "video_name": os.path.basename(video_path),
                "video_path": video_path,
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "error": str(e),
            }

    def predict_folder(
        self, folder_path: str, output_file: str = "", batch_size: int = 4
    ) -> pd.DataFrame:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"folder not found: {folder_path}")
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
        video_files: list[str] = []
        for root, _dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
        results: list[dict[str, Any]] = []
        for video_path in tqdm(video_files, desc="processing videos"):
            results.append(self.predict_video(video_path, batch_size))
        df = pd.DataFrame(results)
        if output_file:
            df.to_csv(output_file, index=False, encoding="utf-8")
        return df


def process_video_folder_simple(
    folder_path: str,
    model_path: str | None = None,
    output_file: str = "video_results.csv",
) -> pd.DataFrame:
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "videomae_results", "checkpoint-14172"),
            os.path.join(current_dir, "checkpoint-14172"),
            os.path.join(
                os.path.dirname(current_dir),
                "videomae-base-finetuned-klin",
                "checkpoint-24536",
            ),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    if model_path is None:
        raise FileNotFoundError("model path required")
    classifier = VideoFolderClassifier(model_path)
    results = classifier.predict_folder(
        folder_path=folder_path, output_file=output_file
    )
    return results


if __name__ == "__main__":
    video_path = r"C:\Users\meksi\Desktop\d\Fighting023_x264.mp4"
    model_path = (
        r"C:\Users\meksi\Documents\GitHub\KLIN\videomae_results\videomae-ufc-crime"
    )

    classifier = VideoFolderClassifier(model_path=model_path)
    result = classifier.predict_video(video_path)
    print(result)
