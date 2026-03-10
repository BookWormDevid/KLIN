from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import cast

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT_DIR / "models" / "pre_trained_x3d_model.pt"
X3D_MODULE_PATH = ROOT_DIR / "model_repository" / "x3d_violence" / "1" / "x3d_net.py"
OUTPUT_DIR = ROOT_DIR / "model_repository" / "x3d_violence" / "1"
OUTPUT_PATH = OUTPUT_DIR / "model.onnx"


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        logits = cast(torch.Tensor, self.model(video))
        return logits.squeeze(-1)


def load_x3d_module(module_path: Path):
    spec = spec_from_file_location("x3d_net_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load X3D module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_state_dict(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned_weights: dict[str, torch.Tensor] = {}

    for key, value in weights.items():
        key = key.replace("module.", "")
        if "split_bn" in key:
            continue
        cleaned_weights[key] = value

    return cleaned_weights


def build_model() -> ExportWrapper:
    x3d = load_x3d_module(X3D_MODULE_PATH)
    weights = torch.load(WEIGHTS_PATH, map_location="cpu")

    model = x3d.generate_model(
        x3d_version="M",
        n_classes=2,
        n_input_channels=3,
        dropout=0,
        base_bn_splits=1,
    )

    model.load_state_dict(clean_state_dict(weights), strict=False)
    model.eval()

    wrapped_model = ExportWrapper(model)
    wrapped_model.eval()
    return wrapped_model


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    dummy_input = torch.randn(1, 3, 16, 224, 224)

    torch.onnx.export(
        model,
        (dummy_input,),
        OUTPUT_PATH.resolve(),
        input_names=["video"],
        output_names=["logits"],
        opset_version=18,
        external_data=True,
        dynamo=True,
        dynamic_shapes={"video": {0: "batch"}},
    )

    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
