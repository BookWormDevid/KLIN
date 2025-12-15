import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = "1"

import cv2
import torch
import numpy as np
import time
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


path_to_model = Path (__file__).parent / "videomae-base-finetuned-klin/checkpoint-24536"
print(str(path_to_model))

class VideoClassifier:
    def __init__(
        self,
        model_name: str = str(path_to_model),
        chunk_size: int = 16,
        frame_size: tuple = (224, 224)
    ):
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[VideoClassifier] Loading model on {self.device}...")
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name, local_files_only=True).to(self.device)
        self.model.eval()
        print("[VideoClassifier] Model loaded.")

    def _read_video_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            frames.append(frame_resized)

        cap.release()

        duration = frame_count / fps if fps > 0 else None
        return np.array(frames), fps, duration, frame_count

    def _chunk_frames(self, frames: np.ndarray):
        t = len(frames)
        padding_needed = (-t) % self.chunk_size

        if padding_needed > 0:
            padding = np.zeros((padding_needed, *self.frame_size, 3), dtype=np.uint8)
            frames = np.concatenate([frames, padding], axis=0)

        num_chunks = len(frames) // self.chunk_size
        return frames.reshape(num_chunks, self.chunk_size, *self.frame_size, 3)

    def predict(self, video_path: str, batch_size: int = 8):
        start_time = time.time()

        # Read video
        frames, fps, duration, total_frames = self._read_video_frames(video_path)
        chunks = self._chunk_frames(frames)

        all_logits = []

        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_list = [list(x) for x in batch]  # required by VideoMAE
                inputs = self.processor(batch_list, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                all_logits.append(outputs.logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        final_logits = torch.mean(logits, dim=0)

        predicted_idx = final_logits.argmax().item()
        confidence = torch.softmax(final_logits, dim=0)[predicted_idx].item() * 100

        processing_time = time.time() - start_time

        return {
            "prediction": self.model.config.id2label[predicted_idx].lower(),
            "confidence": round(confidence, 2),
            "processing_time": round(processing_time, 2),
            "details": {
                "frames_analyzed": int(total_frames),
                "video_duration": round(duration, 2) if duration else None,
                "fps": round(fps, 2),
                "chunks_used": int(len(chunks))
            }
        }
