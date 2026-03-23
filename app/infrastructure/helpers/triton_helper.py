"""
Подготовка передачи данных для triton
"""

from typing import cast

import cv2
import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore[import-untyped]
from numpy.typing import NDArray


class PrepareForTriton:
    """
    Класс содержит методы для подачи нормализованных данных для triton
    """

    @staticmethod
    def prepare_x3d_for_triton(frames: list[np.ndarray]) -> np.ndarray:
        """
        Класс для подготовки данных для модели x3d
        """
        frames_np = np.array(frames).astype(np.float32) / 255.0
        frames_np = np.transpose(frames_np, (3, 0, 1, 2))
        frames_np = np.expand_dims(frames_np, axis=0)
        return frames_np

    @staticmethod
    def prepare_yolo_frame_for_triton(frame: np.ndarray) -> np.ndarray:
        """
        Класс для подготовки данных для модели yolo
        """
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def prepare_mae_chunk_for_triton(frames: list[np.ndarray]) -> np.ndarray:
        """
        Класс для подготовки данных для модели mae
        """
        frames_np = np.array(frames).astype(np.float32) / 255.0
        frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # T C H W
        frames_np = np.expand_dims(frames_np, axis=0)  # B T C H W
        return frames_np


def infer_single_output(
    triton: grpcclient.InferenceServerClient,
    *,
    model_name: str,
    input_name: str,
    input_data: np.ndarray,
    output_name: str,
) -> NDArray[np.float32]:
    """
    Run a Triton request with one input tensor and one output tensor.
    """

    inputs = grpcclient.InferInput(input_name, input_data.shape, "FP32")
    inputs.set_data_from_numpy(input_data)
    outputs = grpcclient.InferRequestedOutput(output_name)

    result = triton.infer(
        model_name=model_name,
        inputs=[inputs],
        outputs=[outputs],
    )
    return cast(
        NDArray[np.float32],
        np.asarray(result.as_numpy(output_name)[0], dtype=np.float32),
    )
