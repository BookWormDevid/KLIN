from pathlib import Path

from ultralytics import YOLO

parent = Path(__file__).resolve().parent.parent

model = YOLO(parent / "models/yolo26x.pt")


model.predict(source=parent / r"C:\Users\meksi\Desktop\d\Fighting023_x264.mp4", show=True, conf=0.6)
