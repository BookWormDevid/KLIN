import pytest
from unittest.mock import MagicMock
import numpy as np

from app.infrastructure.services.target import InferenceProcessor


@pytest.mark.asyncio
async def test_process_video_stream_mocked(monkeypatch):
    processor = InferenceProcessor()

    # мок моделей
    processor.mae.model = MagicMock()
    processor.mae.processor = MagicMock()
    processor.yolo.yolo = MagicMock()

    processor._run_yolo_on_frame = MagicMock(return_value=[])
    processor._predict_mae_chunk = MagicMock(return_value={
        "time": [0, 1],
        "answer": "test",
        "confident": 0.9
    })

    fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self):
            self.count = 0

        def isOpened(self):
            return True

        def read(self):
            if self.count < 5:
                self.count += 1
                return True, fake_frame
            return False, None

        def get(self, prop):
            return 5

        def release(self):
            pass

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

    mae_results, yolo_bbox, objects, video_info = \
        await processor._process_video_stream("fake.mp4")

    assert isinstance(mae_results, list)