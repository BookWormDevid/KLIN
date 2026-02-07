from pathlib import Path

from ultralytics import YOLO

parent = Path(__file__).resolve().parent.parent

model = YOLO(parent / "models/yolo26n.pt")


model.predict(source=parent / r"C:\Users\meksi\Desktop\d\Fighting004_x264.mp4", show=True, conf=0.6)
