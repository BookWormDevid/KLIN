"""
Triton-процессор для стадии VideoMAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]
from numpy.typing import NDArray

from app.infrastructure.helpers import (
    PrepareForTriton,
    infer_single_output,
    stable_softmax,
)


@dataclass
class MaeProcessor:
    """
    Triton-backed VideoMAE.
    """

    triton: grpcclient.InferenceServerClient
    prepare: PrepareForTriton
    model_name: str = "videomae_crime"

    def infer_probs(self, chunk_frames: list[np.ndarray]) -> NDArray[np.float32]:
        """
        Запускает инференс VideoMAE и возвращает вероятности классов.
        """
        img = self.prepare.prepare_mae_chunk_for_triton(chunk_frames)
        logits = infer_single_output(
            self.triton,
            model_name=self.model_name,
            input_name="pixel_values",
            input_data=img,
            output_name="logits",
        )
        return stable_softmax(logits)
