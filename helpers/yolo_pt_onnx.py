import pathlib

from ultralytics import YOLO


dir = pathlib.Path(__file__).parent.parent
print(dir)
tdir = dir / "models" / "yolov8x.pt"
model = YOLO(tdir)
model.export(format="onnx", opset=17, dynamic=False, simplify=True)
