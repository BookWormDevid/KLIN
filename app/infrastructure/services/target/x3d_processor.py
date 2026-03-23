"""
Triton-процессор для стадии X3D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]
from numpy.typing import NDArray

from app.infrastructure.helpers import PrepareForTriton, infer_single_output


@dataclass
class X3DProcessor:
    """
    Triton-backed X3D.
    """

    triton: grpcclient.InferenceServerClient
    prepare: PrepareForTriton
    model_name: str = "x3d_violence"

    def infer_prepared(self, video_arr: np.ndarray) -> NDArray[np.float32]:
        """
        Выполняет X3D-инференс на уже подготовленном тензоре.
        """
        return infer_single_output(
            self.triton,
            model_name=self.model_name,
            input_name="video",
            input_data=video_arr,
            output_name="logits",
        )

    def infer_clip(self, frames: list[np.ndarray]) -> NDArray[np.float32]:
        """
        Подготавливает клип и запускает X3D-инференс.
        """
        video_arr = self.prepare.prepare_x3d_for_triton(frames)
        return self.infer_prepared(video_arr)
