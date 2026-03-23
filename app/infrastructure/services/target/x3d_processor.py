from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]
from numpy.typing import NDArray

from app.infrastructure.helpers import PrepareForTriton


@dataclass
class X3DProcessor:
    """Triton-backed X3D stage."""

    triton: grpcclient.InferenceServerClient
    prepare: PrepareForTriton
    model_name: str = "x3d_violence"

    def infer_prepared(self, video_arr: np.ndarray) -> NDArray[np.float32]:
        inputs = grpcclient.InferInput("video", video_arr.shape, "FP32")
        inputs.set_data_from_numpy(video_arr)
        outputs = grpcclient.InferRequestedOutput("logits")

        result = self.triton.infer(
            model_name=self.model_name,
            inputs=[inputs],
            outputs=[outputs],
        )
        return np.asarray(result.as_numpy("logits")[0], dtype=np.float32)

    def infer_clip(self, frames: list[np.ndarray]) -> NDArray[np.float32]:
        video_arr = self.prepare.prepare_x3d_for_triton(frames)
        return self.infer_prepared(video_arr)
