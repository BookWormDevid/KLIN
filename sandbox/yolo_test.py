from pathlib import Path

from ultralytics import YOLO

parent = Path(__file__).resolve().parent

model = YOLO(parent / "models/yolo26n.pt")


model.predict(source=parent / "videos/fi004.mp4", show=True, conf=0.6)
