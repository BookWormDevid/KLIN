import pathlib

import torch
from transformers import VideoMAEForVideoClassification


dir = pathlib.Path(__file__).parent.parent
model_path = dir / "models" / "videomae-ucf-crime"

model = VideoMAEForVideoClassification.from_pretrained(model_path)
model.eval()

dummy_input = torch.randn(1, 16, 3, 224, 224)

torch.onnx.export(
    model,
    (dummy_input,),
    (dir / "model_repository/videomae_crime/1/model.onnx").resolve(),
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=18,
)
