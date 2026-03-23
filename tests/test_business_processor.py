import numpy as np
import pytest

from app.infrastructure.helpers import TimeRangeHelper
from app.infrastructure.services.target import BusinessProcessor


def build_business_model() -> BusinessProcessor:
    return BusinessProcessor(
        mae_classes={0: "Normal", 1: "Fight"},
        yolo_classes={0: "person"},
        allowed_classes={0},
        yolo_conf=0.6,
    )


def test_classify_x3d_logits_returns_binary_result() -> None:
    model = build_business_model()

    result = model.classify_x3d_logits(np.array([0.1, 2.1], dtype=np.float32))

    assert list(result.keys()) == ["True"]
    assert result["True"] == pytest.approx(0.8808, rel=1e-3)


def test_build_mae_result_maps_class_and_time_range() -> None:
    model = build_business_model()

    result = model.build_mae_result(
        np.array([0.2, 0.8], dtype=np.float32),
        start_frame=0,
        end_frame=29,
        fps=30.0,
        timerange=TimeRangeHelper(),
    )

    assert result["time"] == [0.0, 1.0]
    assert result["answer"] == "Fight"
    assert result["confident"] == pytest.approx(0.8)


def test_parse_yolo_detection_filters_by_class_and_threshold() -> None:
    model = build_business_model()

    good_pred = np.array([10.0, 10.0, 4.0, 6.0, 0.95], dtype=np.float32)
    low_conf_pred = np.array([10.0, 10.0, 4.0, 6.0, 0.20], dtype=np.float32)

    assert model.parse_yolo_detection(low_conf_pred) is None
    assert model.parse_yolo_detection(good_pred) == (
        0,
        [8.0, 7.0, 12.0, 13.0],
    )
