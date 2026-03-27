import uuid
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from app.infrastructure.helpers.triton_helper import (
    PrepareForTriton,
    infer_single_output,
)
from app.infrastructure.services.inference_stub import ApiInferenceStub
from app.infrastructure.services.target.mae_processor import MaeProcessor
from app.infrastructure.services.target.x3d_processor import X3DProcessor
from app.infrastructure.services.target.yolo_processor import YoloProcessor
from app.models import KlinModel, ProcessingState


class FakeInferInput:
    def __init__(self, name: str, shape: tuple[int, ...], datatype: str) -> None:
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data: NDArray[Any] | None = None

    def set_data_from_numpy(self, data: np.ndarray) -> None:
        self.data = data


class FakeInferRequestedOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeInferResult:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


def test_prepare_for_triton_builds_expected_tensor_shapes() -> None:
    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8) * 255,
    ]

    x3d = PrepareForTriton.prepare_x3d_for_triton(frames)
    mae = PrepareForTriton.prepare_mae_chunk_for_triton(frames)
    yolo = PrepareForTriton.prepare_yolo_frame_for_triton(frames[0])

    assert x3d.shape == (1, 3, 2, 4, 4)
    assert mae.shape == (1, 2, 3, 4, 4)
    assert yolo.shape == (1, 3, 640, 640)
    assert x3d.dtype == np.float32
    assert mae.dtype == np.float32
    assert yolo.dtype == np.float32


def test_infer_single_output_wraps_triton_input_and_casts_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triton = MagicMock()
    triton.infer.return_value = FakeInferResult(
        {"logits": np.array([[1.5, 2.5]], dtype=np.float64)}
    )
    monkeypatch.setattr(
        "app.infrastructure.helpers.triton_helper.grpcclient.InferInput",
        FakeInferInput,
    )
    monkeypatch.setattr(
        "app.infrastructure.helpers.triton_helper.grpcclient.InferRequestedOutput",
        FakeInferRequestedOutput,
    )

    result = infer_single_output(
        triton,
        model_name="x3d_violence",
        input_name="video",
        input_data=np.ones((1, 3, 2, 4, 4), dtype=np.float32),
        output_name="logits",
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([1.5, 2.5], dtype=np.float32))
    triton.infer.assert_called_once()


def test_x3d_processor_uses_prepared_and_raw_inference_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triton = MagicMock()
    prepare = MagicMock()
    expected_logits = np.array([0.1, 0.9], dtype=np.float32)
    processor = X3DProcessor(triton=triton, prepare=prepare)
    infer_prepared = MagicMock(return_value=expected_logits)
    monkeypatch.setattr(processor, "infer_prepared", infer_prepared)
    prepare.prepare_x3d_for_triton.return_value = np.ones(
        (1, 3, 2, 4, 4), dtype=np.float32
    )

    result = processor.infer_clip([np.zeros((4, 4, 3), dtype=np.uint8)])

    assert result is expected_logits
    prepare.prepare_x3d_for_triton.assert_called_once()
    infer_prepared.assert_called_once()


def test_mae_processor_prepares_chunk_and_returns_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triton = MagicMock()
    prepare = MagicMock()
    prepare.prepare_mae_chunk_for_triton.return_value = np.ones(
        (1, 2, 3, 4, 4), dtype=np.float32
    )
    infer_mock = MagicMock(return_value=np.array([1.0, 3.0], dtype=np.float32))
    monkeypatch.setattr(
        "app.infrastructure.services.target.mae_processor.infer_single_output",
        infer_mock,
    )
    processor = MaeProcessor(triton=triton, prepare=prepare)

    probs = processor.infer_probs([np.zeros((4, 4, 3), dtype=np.uint8)])

    assert probs.dtype == np.float32
    assert pytest.approx(float(np.sum(probs)), rel=1e-6) == 1.0
    assert probs[1] > probs[0]
    infer_mock.assert_called_once()


def test_yolo_processor_transposes_each_batch_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triton = MagicMock()
    raw_output = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    triton.infer.return_value = FakeInferResult({"output0": raw_output})
    monkeypatch.setattr(
        "app.infrastructure.services.target.yolo_processor.grpcclient.InferInput",
        FakeInferInput,
    )
    monkeypatch.setattr(
        "app.infrastructure.services.target.yolo_processor.grpcclient.InferRequestedOutput",
        FakeInferRequestedOutput,
    )
    processor = YoloProcessor(triton=triton)
    batch = np.ones((2, 3, 640, 640), dtype=np.float32)

    result = processor.infer_batch(batch)

    assert len(result) == 2
    np.testing.assert_array_equal(result[0], raw_output[0].transpose(1, 0))
    np.testing.assert_array_equal(result[1], raw_output[1].transpose(1, 0))


@pytest.mark.anyio
async def test_api_inference_stub_explains_where_inference_runs() -> None:
    model = KlinModel(
        id=uuid.uuid4(),
        response_url=None,
        video_path="/tmp/video.mp4",
        state=ProcessingState.PENDING,
    )
    stub = ApiInferenceStub()

    assert "queue worker" in stub.unavailable_reason().lower()

    with pytest.raises(RuntimeError, match=str(model.id)):
        await stub.analyze(model)
