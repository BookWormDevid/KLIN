from ultralytics import YOLO
from pathlib import Path


parent = Path(__file__).resolve().parent
model = YOLO(parent / "yolo_small_weights.pt")


results = model.predict(source=parent /"videos/fi004.mp4", show=True, conf=0.6)


print(results)