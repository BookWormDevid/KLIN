"""
Тесты логики стриминговой обработки видео
"""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.infrastructure.services.target import InferenceProcessor


class FakeVideoCapture:
    """Фейковый класс, имитирующий cv2.VideoCapture для тестов."""

    def __init__(self, frame_count: int = 10, fps: float = 30.0):
        self.count = 0
        self.frame_count = frame_count
        self.fps = fps
        self._opened = True

    def is_opened(self) -> bool:
        """Проверяет, открыто ли видео."""
        return self._opened

    def __getattr__(self, attr: str):
        """Совместимость с API cv2.VideoCapture."""
        if attr == "isOpened":
            return self.is_opened
        raise AttributeError(attr)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Читает следующий кадр.
        Возвращает (True, кадр) или (False, None) при конце видео.
        """
        if self.count < self.frame_count:
            self.count += 1
            fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            return True, fake_frame
        self._opened = False
        return False, None

    def get(self, prop: int) -> float:
        """Возвращает свойства видео."""
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def release(self) -> None:
        """Освобождает ресурсы (в фейке — ничего)."""
        self._opened = False


class InferenceProcessorTestAdapter(InferenceProcessor):
    """Адаптер для тестирования protected-методов через публичные обертки."""

    def set_yolo_runner(self, runner: MagicMock) -> MagicMock:
        """Подменяет обработчик YOLO."""
        self._run_yolo_on_frame = runner
        return runner

    def set_mae_predictor(self, predictor: MagicMock) -> MagicMock:
        """Подменяет предиктор MAE."""
        self._predict_mae_chunk = predictor
        return predictor

    async def process_video_stream(self, video_path: str):
        """Публичная обертка над потоковой обработкой видео."""
        return await self._process_video_stream(video_path)


@pytest.mark.asyncio
async def test_process_video_stream_mocked(monkeypatch):
    """
    Тест стриминговой обработки видео с моками моделей и cv2.VideoCapture.

    Проверяет:
    - корректный вызов _predict_mae_chunk
    - обработку небольшого количества кадров
    - правильный формат возвращаемых данных
    - вызов YOLO на нужных кадрах (если stride позволяет)
    """
    processor = InferenceProcessorTestAdapter()

    # Мокаем зависимости
    processor.mae.model = MagicMock()
    processor.mae.processor = MagicMock()
    processor.yolo.yolo = MagicMock()

    run_yolo_mock = processor.set_yolo_runner(MagicMock(return_value=[]))
    predict_mae_mock = processor.set_mae_predictor(
        MagicMock(
            return_value={
                "time": [0.0, 1.0],
                "answer": "test_label",
                "confident": 0.92,
            }
        )
    )

    # Подменяем VideoCapture
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda _path: FakeVideoCapture(frame_count=10, fps=30.0),
    )

    # Запускаем обработку
    (
        mae_results,
        yolo_bbox,
        detected_objects,
        video_info,
    ) = await processor.process_video_stream("fake.mp4")

    # Проверки
    assert isinstance(mae_results, list)
    assert len(mae_results) >= 1, "Должен быть хотя бы один чанк MAE"

    # Проверяем вызовы MAE
    assert predict_mae_mock.call_count >= 1
    assert predict_mae_mock.call_count <= 2
    # при chunk_size=16 и 10 кадрах → 1 чанк + паддинг

    # Проверяем структуру результата MAE
    first_result = mae_results[0]
    assert "time" in first_result
    assert "answer" in first_result
    assert "confident" in first_result
    assert isinstance(first_result["time"], list)
    assert len(first_result["time"]) == 2
    assert isinstance(first_result["confident"], float)
    assert first_result["confident"] > 0

    # Проверяем YOLO — при stride=2 должен быть вызван ~5 раз
    expected_yolo_calls = (
        10 + processor.processing.yolo_stride - 1
    ) // processor.processing.yolo_stride
    assert run_yolo_mock.call_count == expected_yolo_calls

    # Проверяем информацию о видео
    assert isinstance(video_info, dict)
    assert video_info["total_frames"] == 10
    assert video_info["fps"] == 30.0
    assert video_info["frames_read"] == 10

    # Проверяем, что при пустых детекциях YOLO возвращает ожидаемые пустые значения
    assert yolo_bbox == {}
    assert detected_objects == []
