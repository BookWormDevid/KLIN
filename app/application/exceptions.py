"""
Доменные исключения слоя application.
"""

from __future__ import annotations

import uuid


class KlinNotFoundError(Exception):
    """
    Ошибка отсутствия задачи в хранилище.
    """

    def __init__(self, klin_id: uuid.UUID) -> None:
        super().__init__(f"Klin {klin_id} not found")
        self.klin_id = klin_id


class KlinEnqueueError(Exception):
    """
    Ошибка постановки задачи в очередь обработки.
    """
