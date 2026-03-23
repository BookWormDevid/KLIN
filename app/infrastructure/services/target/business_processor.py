"""
Бизнес-постобработка результатов отдельных ML-стадий.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.infrastructure.helpers import TimeRangeHelper, stable_softmax


@dataclass
class BusinessProcessor:
    """
    Бизнес-правила и нормализация результатов ML-пайплайна.
    """

    mae_classes: dict[int, str]
    yolo_classes: dict[int, str]
    allowed_classes: set[int]
    yolo_conf: float
    timerange: TimeRangeHelper = field(default_factory=TimeRangeHelper)

    def classify_x3d_logits(self, logits: np.ndarray) -> dict[str, float]:
        """
        Преобразует логиты X3D в бинарное решение и confidence.
        """

        probs = stable_softmax(logits)
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        return {str(bool(pred)): confidence}

    def build_mae_result(
        self,
        probs: np.ndarray,
        *,
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> dict[str, Any]:
        """
        Собирает результат классификации VideoMAE в контракт сервиса.
        """

        pred_idx = int(np.argmax(probs))
        confidence = float(np.asarray(probs, dtype=np.float32)[pred_idx])
        answer = self.mae_classes.get(pred_idx, str(pred_idx))

        return {
            "time": self.timerange.build_time_range(start_frame, end_frame, fps),
            "answer": answer,
            "confident": confidence,
        }

    def parse_yolo_detection(self, pred: np.ndarray) -> tuple[int, list[float]] | None:
        """
        Фильтрует и нормализует детекцию YOLO в bbox-формат сервиса.
        """

        scores = pred[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])

        if conf < self.yolo_conf or class_id not in self.allowed_classes:
            return None

        x, y, w, h = pred[:4]
        bbox = [
            float(x - w / 2),
            float(y - h / 2),
            float(x + w / 2),
            float(y + h / 2),
        ]
        return class_id, bbox

    def resolve_detected_objects(self, detected_class_ids: set[int]) -> list[str]:
        """
        Преобразует набор детектированных class_id в человекочитаемые имена.
        """

        return [
            self.yolo_classes[class_id]
            for class_id in detected_class_ids
            if class_id in self.yolo_classes
        ]
