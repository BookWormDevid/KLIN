from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]
from numpy.typing import NDArray

from app.infrastructure.helpers import PrepareForTriton, stable_softmax


@dataclass
class MaeProcessor:
    """Triton-backed VideoMAE stage."""

    triton: grpcclient.InferenceServerClient
    prepare: PrepareForTriton
    model_name: str = "videomae_crime"

    def infer_probs(self, chunk_frames: list[np.ndarray]) -> NDArray[np.float32]:
        img = self.prepare.prepare_mae_chunk_for_triton(chunk_frames)
        inputs = grpcclient.InferInput("pixel_values", img.shape, "FP32")
        inputs.set_data_from_numpy(img)
        outputs = grpcclient.InferRequestedOutput("logits")

        result = self.triton.infer(
            model_name=self.model_name,
            inputs=[inputs],
            outputs=[outputs],
        )
        logits = np.asarray(result.as_numpy("logits")[0], dtype=np.float32)
        return stable_softmax(logits)
