from pathlib import Path

from ultralytics import YOLO


dir = Path(__file__).parent.parent

model = YOLO("models/yolov8x.pt")

out_dir = Path("model_repository/yolo_person/1")
out_dir.mkdir(parents=True, exist_ok=True)

exported_path = Path(
    model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        device="cpu",
    )
)

target_path = out_dir / "model.onnx"
exported_path.replace(target_path)

print(f"Saved to: {target_path.resolve()}")
