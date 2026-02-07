import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# Папка с тестовыми видео
video_folder = r"C:\Users\meksi\Desktop\d"          # ← поменяй на свою

# Устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"/videomae_results/videomae-ufc-crime"

processor = VideoMAEImageProcessor.from_pretrained(model_path, local_files_only=True)
model = VideoMAEForVideoClassification.from_pretrained(
    model_path,
    local_files_only=True,
    ignore_mismatched_sizes=True
).to(device)
model.eval()

# Маппинги берём прямо из модели
id2label = model.config.id2label
# label2id = model.config.label2id   # если понадобится

print("Классы модели:", id2label)

# ────────────────────────────────────────────────
# Функция загрузки и предобработки видео
# ────────────────────────────────────────────────
def load_and_process_video(video_path, processor, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    # Равномерно сэмплируем num_frames кадров
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    # Если кадров меньше — дублируем последний
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Теперь frames — list из PIL-like массивов (H,W,3)

    # Processor сам сделает resize, normalize, to tensor и stack
    inputs = processor(
        frames,               # list of frames
        return_tensors="pt"
    )

    # inputs["pixel_values"] → [1, num_frames, 3, 224, 224]
    return inputs["pixel_values"].squeeze(0)  # [num_frames, 3, H, W]

# ────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, video_folder, processor):
        self.video_files = [
            os.path.join(video_folder, f)
            for f in os.listdir(video_folder)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        self.processor = processor

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        path = self.video_files[idx]
        video_tensor = load_and_process_video(path, self.processor)
        if video_tensor is None:
            # Пропускаем битые файлы
            return {"video": None, "filename": os.path.basename(path)}
        return {"video": video_tensor, "filename": os.path.basename(path)}

# ────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────
test_dataset = VideoDataset(video_folder, processor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for batch in test_loader:
        video_tensor = batch["video"]          # [1, num_frames, 3, H, W]
        filename = batch["filename"][0]

        if video_tensor is None:
            print(f"Video {filename} — ошибка загрузки, пропускаем")
            continue

        inputs = {"pixel_values": video_tensor.to(device)}

        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_label = id2label[pred_idx]
        confidence = probs[0, pred_idx].item()

        print(f"Файл: {filename}")
        print(f"→ Предсказание: {pred_label} (id {pred_idx}) | уверенность: {confidence:.3f}")
        print("-" * 60)