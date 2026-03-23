from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]
from numpy.typing import NDArray


@dataclass
class YoloProcessor:
    """Triton-backed YOLO stage."""

    triton: grpcclient.InferenceServerClient
    model_name: str = "yolo_person"

    def infer_batch(self, batch_imgs: np.ndarray) -> list[NDArray[np.float32]]:
        inputs = grpcclient.InferInput("images", batch_imgs.shape, "FP32")
        inputs.set_data_from_numpy(batch_imgs)
        outputs = grpcclient.InferRequestedOutput("output0")

        result = self.triton.infer(
            model_name=self.model_name,
            inputs=[inputs],
            outputs=[outputs],
        )

        raw_output: NDArray[np.float32] = result.as_numpy("output0")
        return [
            np.asarray(raw_output[batch_idx].transpose(1, 0), dtype=np.float32)
            for batch_idx in range(raw_output.shape[0])
        ]
