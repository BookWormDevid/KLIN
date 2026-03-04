"""
Utilities for exporting production models to ONNX format for Triton.

Supported exporters:
- VideoMAE (Hugging Face directory with config + safetensors/bin)
- X3D (PyTorch checkpoint from .pt/.pth)
- YOLO (Ultralytics .pt)
"""

from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path
from typing import Any, Protocol, cast

import onnx
import torch
from torch import nn
from transformers import VideoMAEForVideoClassification
from ultralytics import YOLO


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = DEFAULT_REPO_ROOT / "model_repository"

DEFAULT_VIDEOMAE_OUT = DEFAULT_MODEL_REPO / "videomae_crime" / "1" / "model.onnx"
DEFAULT_X3D_OUT = DEFAULT_MODEL_REPO / "x3d_violence" / "1" / "model.onnx"
DEFAULT_YOLO_OUT = DEFAULT_MODEL_REPO / "yolo_person" / "1" / "model.onnx"


class X3DHubProtocol(Protocol):
    """Typed protocol for the required pytorchvideo hub APIs."""

    def x3d_s(
        self,
        *,
        pretrained: bool = False,
        progress: bool = True,
        **kwargs: Any,
    ) -> nn.Module: ...

    def x3d_m(
        self,
        *,
        pretrained: bool = False,
        progress: bool = True,
        **kwargs: Any,
    ) -> nn.Module: ...


class VideoMAEOnnxWrapper(nn.Module):
    """Wraps HF VideoMAE to export only logits tensor."""

    def __init__(self, model: VideoMAEForVideoClassification) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns only logits."""
        return cast(torch.Tensor, self.model(pixel_values=pixel_values).logits)


def ensure_parent_dir(path: Path) -> None:
    """Create target parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_inspect_onnx_io(onnx_path: Path) -> None:
    """
    Print ONNX input/output names when parsing libraries are available.

    This helps align Triton `config.pbtxt` names with exported graph names.
    """
    ort_exc: Exception | None = None
    try:
        ort_module = importlib.import_module("onnxruntime")
        inference_session = cast(Any, ort_module.InferenceSession)
        session = inference_session(onnx_path.as_posix())
        input_names = [item.name for item in session.get_inputs()]
        output_names = [item.name for item in session.get_outputs()]
        print(f"[ONNX] Inputs: {input_names}")
        print(f"[ONNX] Outputs: {output_names}")
        return
    except Exception as exc:
        ort_exc = exc

    try:
        onnx_module = importlib.import_module("onnx")
        onnx_load = cast(Any, onnx_module.load)
        model = onnx_load(onnx_path.as_posix())
        input_names = [item.name for item in model.graph.input]
        output_names = [item.name for item in model.graph.output]
        print(f"[ONNX] Inputs: {input_names}")
        print(f"[ONNX] Outputs: {output_names}")
    except Exception as exc:
        if ort_exc is not None:
            print(f"[ONNX] onnxruntime inspection failed: {ort_exc}")
        print(f"[ONNX] onnx inspection failed: {exc}")
        print(
            "[ONNX] Unable to inspect graph names. "
            "Install `onnxruntime` or `onnx` for I/O inspection."
        )


def _ensure_positive_int(name: str, value: int) -> None:
    """
    Validate positive integer CLI argument.
    """
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def maybe_rewrite_onnx_ir_version(
    output_path: Path, target_ir_version: int | None
) -> None:
    """
    Rewrite ONNX IR version after export for runtime compatibility.

    Useful when exporting with recent ONNX packages but deploying to older
    runtimes (for example Triton 24.01 / ORT with max supported IR=9).
    """
    if target_ir_version is None:
        return

    _ensure_positive_int("--target-ir-version", target_ir_version)
    model = onnx.load(output_path.as_posix(), load_external_data=True)
    source_ir = int(model.ir_version)
    if source_ir == target_ir_version:
        print(f"[ONNX] IR version already {target_ir_version}, no rewrite needed.")
        return

    model.ir_version = target_ir_version
    external_data_path = output_path.with_name(f"{output_path.name}.data")
    if external_data_path.exists():
        onnx.save_model(
            model,
            output_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save(model, output_path.as_posix())

    onnx.checker.check_model(output_path.as_posix())
    print(f"[ONNX] Rewrote IR version: {source_ir} -> {target_ir_version}")


def _validate_output_path(path: Path) -> None:
    """
    Validate ONNX output file path.
    """
    if path.suffix.lower() != ".onnx":
        raise ValueError(f"Output path must end with .onnx: {path}")


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate CLI args before running exporters.
    """
    _ensure_positive_int("--opset", args.opset)

    if args.command == "videomae":
        if not args.model_dir.is_dir():
            raise ValueError(f"Model directory not found: {args.model_dir}")
        _ensure_positive_int("--frames", args.frames)
        _ensure_positive_int("--height", args.height)
        _ensure_positive_int("--width", args.width)
        if args.target_ir_version is not None:
            _ensure_positive_int("--target-ir-version", args.target_ir_version)
        _validate_output_path(args.output)
        return

    if args.command == "x3d":
        if not args.checkpoint.is_file():
            raise ValueError(f"Checkpoint file not found: {args.checkpoint}")
        _ensure_positive_int("--num-classes", args.num_classes)
        _ensure_positive_int("--frames", args.frames)
        _ensure_positive_int("--height", args.height)
        _ensure_positive_int("--width", args.width)
        if args.target_ir_version is not None:
            _ensure_positive_int("--target-ir-version", args.target_ir_version)
        _validate_output_path(args.output)
        return

    if args.command == "yolo":
        if not args.weights.is_file():
            raise ValueError(f"Weights file not found: {args.weights}")
        _ensure_positive_int("--imgsz", args.imgsz)
        if args.target_ir_version is not None:
            _ensure_positive_int("--target-ir-version", args.target_ir_version)
        _validate_output_path(args.output)
        return

    raise ValueError(f"Unsupported command: {args.command}")


def load_pytorchvideo_hub() -> X3DHubProtocol:
    """Load pytorchvideo hub lazily to avoid static untyped imports in mypy."""
    module = importlib.import_module("pytorchvideo.models.hub")
    required = ("x3d_s", "x3d_m")
    for name in required:
        if not hasattr(module, name):
            raise ImportError(f"pytorchvideo.models.hub missing '{name}'")
    return cast(X3DHubProtocol, module)


def export_videomae_to_onnx(
    *,
    model_dir: Path,
    output_path: Path,
    frames: int,
    height: int,
    width: int,
    opset: int,
    dynamic_batch: bool,
    local_files_only: bool,
    target_ir_version: int | None,
) -> None:
    """Export VideoMAE model directory to ONNX."""
    print(f"[VideoMAE] Loading model from: {model_dir}")
    model = VideoMAEForVideoClassification.from_pretrained(
        model_dir.as_posix(),
        local_files_only=local_files_only,
    )
    model.eval()
    wrapper = VideoMAEOnnxWrapper(model).eval()

    ensure_parent_dir(output_path)
    dummy = torch.randn(1, frames, 3, height, width, dtype=torch.float32)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {"pixel_values": {0: "batch"}, "logits": {0: "batch"}}

    print(f"[VideoMAE] Exporting to: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path.as_posix(),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    maybe_rewrite_onnx_ir_version(output_path, target_ir_version)
    maybe_inspect_onnx_io(output_path)
    print("[VideoMAE] Done.")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Extract state_dict from typical checkpoint structures.

    Supports:
    - raw state dict
    - {"state_dict": ...}
    - {"model_state_dict": ...}
    - {"model": ...}
    """
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state dict")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = value

    if not state_dict:
        raise ValueError("No tensor weights found in checkpoint")
    return state_dict


def build_x3d_model(
    *,
    variant: str,
    num_classes: int,
    frames: int,
    crop_size: int,
) -> nn.Module:
    """Build X3D model (small or medium) via pytorchvideo hub."""
    pytorchvideo_hub = load_pytorchvideo_hub()

    if variant == "x3d_s":
        return pytorchvideo_hub.x3d_s(
            pretrained=False,
            progress=True,
            model_num_class=num_classes,
            input_clip_length=frames,
            input_crop_size=crop_size,
            head_activation=nn.Identity,
        )
    if variant == "x3d_m":
        return pytorchvideo_hub.x3d_m(
            pretrained=False,
            progress=True,
            model_num_class=num_classes,
            input_clip_length=frames,
            input_crop_size=crop_size,
            head_activation=nn.Identity,
        )
    raise ValueError(f"Unsupported X3D variant: {variant}")


def export_x3d_to_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
    variant: str,
    num_classes: int,
    frames: int,
    height: int,
    width: int,
    opset: int,
    strict_load: bool,
    dynamic_batch: bool,
    target_ir_version: int | None,
) -> None:
    """Export X3D checkpoint (.pt/.pth) to ONNX."""
    print(f"[X3D] Building model variant: {variant}")
    model = build_x3d_model(
        variant=variant,
        num_classes=num_classes,
        frames=frames,
        crop_size=max(height, width),
    )

    print(f"[X3D] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    if missing:
        print(f"[X3D] Missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[X3D] Unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    model.eval()
    ensure_parent_dir(output_path)
    dummy = torch.randn(1, 3, frames, height, width, dtype=torch.float32)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    print(f"[X3D] Exporting to: {output_path}")
    torch.onnx.export(
        model,
        (dummy,),
        output_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    maybe_rewrite_onnx_ir_version(output_path, target_ir_version)
    maybe_inspect_onnx_io(output_path)
    print("[X3D] Done.")


def export_yolo_to_onnx(
    *,
    weights_path: Path,
    output_path: Path,
    imgsz: int,
    opset: int,
    dynamic: bool,
    simplify: bool,
    half: bool,
    target_ir_version: int | None,
) -> None:
    """Export YOLO .pt weights to ONNX and move result to Triton path."""
    print(f"[YOLO] Loading weights: {weights_path}")
    model = YOLO(weights_path.as_posix())
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
    )

    src = Path(str(exported_path))
    ensure_parent_dir(output_path)
    shutil.copy2(src, output_path)
    print(f"[YOLO] Exported from: {src}")
    print(f"[YOLO] Saved to: {output_path}")
    maybe_rewrite_onnx_ir_version(output_path, target_ir_version)
    maybe_inspect_onnx_io(output_path)
    print("[YOLO] Done.")


def add_videomae_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register CLI subcommand for VideoMAE export."""
    parser = subparsers.add_parser("videomae", help="Export VideoMAE model dir to ONNX")
    parser.add_argument(
        "--model-dir", required=True, type=Path, help="HF model directory path"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEOMAE_OUT)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--target-ir-version",
        type=int,
        default=None,
        help="Rewrite exported ONNX IR version (e.g. 9 for Triton 24.01)",
    )
    parser.add_argument(
        "--static-batch", action="store_true", help="Disable dynamic batch axis"
    )
    parser.add_argument(
        "--allow-remote-files",
        action="store_true",
        help="Allow loading weights from remote (sets local_files_only=False)",
    )


def add_x3d_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register CLI subcommand for X3D export."""
    parser = subparsers.add_parser(
        "x3d", help="Export X3D checkpoint (.pt/.pth) to ONNX"
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to .pt/.pth checkpoint"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_X3D_OUT)
    parser.add_argument("--variant", choices=("x3d_s", "x3d_m"), default="x3d_s")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--target-ir-version",
        type=int,
        default=None,
        help="Rewrite exported ONNX IR version (e.g. 9 for Triton 24.01)",
    )
    parser.add_argument(
        "--strict-load", action="store_true", help="Use strict state_dict loading"
    )
    parser.add_argument(
        "--static-batch", action="store_true", help="Disable dynamic batch axis"
    )


def add_yolo_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register CLI subcommand for YOLO export."""
    parser = subparsers.add_parser("yolo", help="Export Ultralytics YOLO .pt to ONNX")
    parser.add_argument(
        "--weights", required=True, type=Path, help="Path to YOLO .pt file"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_YOLO_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--target-ir-version",
        type=int,
        default=None,
        help="Rewrite exported ONNX IR version (e.g. 9 for Triton 24.01)",
    )
    parser.add_argument(
        "--static-batch", action="store_true", help="Disable dynamic axes in export"
    )
    parser.add_argument(
        "--no-simplify", action="store_true", help="Disable ONNX graph simplification"
    )
    parser.add_argument(
        "--half", action="store_true", help="Use FP16 during export where supported"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Export VideoMAE, X3D or YOLO models to ONNX for Triton"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_videomae_subparser(subparsers)
    add_x3d_subparser(subparsers)
    add_yolo_subparser(subparsers)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "videomae":
        export_videomae_to_onnx(
            model_dir=args.model_dir,
            output_path=args.output,
            frames=args.frames,
            height=args.height,
            width=args.width,
            opset=args.opset,
            dynamic_batch=not args.static_batch,
            local_files_only=not args.allow_remote_files,
            target_ir_version=args.target_ir_version,
        )
        return

    if args.command == "x3d":
        export_x3d_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            variant=args.variant,
            num_classes=args.num_classes,
            frames=args.frames,
            height=args.height,
            width=args.width,
            opset=args.opset,
            strict_load=args.strict_load,
            dynamic_batch=not args.static_batch,
            target_ir_version=args.target_ir_version,
        )
        return

    if args.command == "yolo":
        export_yolo_to_onnx(
            weights_path=args.weights,
            output_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=not args.static_batch,
            simplify=not args.no_simplify,
            half=args.half,
            target_ir_version=args.target_ir_version,
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
