from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "yolov8x.pt"

OUTPUT_DIR = ROOT_DIR / "model_repository" / "yolo_person" / "1"
OUTPUT_PATH = OUTPUT_DIR / "model.onnx"


def build_model() -> YOLO:
    return YOLO(MODEL_PATH)


def export_model(model: YOLO) -> Path:
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=640,
            opset=17,
            simplify=True,
            device="cpu",
            # ВАЖНО для Triton batching
            dynamic=True,
            # фиксируем batch dimension
            batch=1,
        )
    )

    exported_path.replace(OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    target_path = export_model(model)

    print(f"Saved to: {target_path.resolve()}")


if __name__ == "__main__":
    main()
