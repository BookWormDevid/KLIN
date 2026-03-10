from pathlib import Path

import torch
from transformers import VideoMAEForVideoClassification


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "videomae-UCF-crime"
OUTPUT_DIR = ROOT_DIR / "model_repository" / "videomae_crime" / "1"
OUTPUT_PATH = OUTPUT_DIR / "model.onnx"


def build_model() -> VideoMAEForVideoClassification:
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model


def export_model(model: VideoMAEForVideoClassification) -> None:
    dummy_input = torch.randn(1, 16, 3, 224, 224)

    torch.onnx.export(
        model,
        (dummy_input,),
        OUTPUT_PATH.resolve(),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=18,
        external_data=True,
        dynamo=True,
        dynamic_shapes={"pixel_values": {0: "batch"}},
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    export_model(model)

    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
