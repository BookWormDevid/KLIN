"""
Определение времени чанка для mae
"""


class TimeRangeHelper:  # pylint: disable=too-few-public-methods
    """
    Класс для вычисления времени чанков в mae
    """

    @staticmethod
    def build_time_range(start_frame: int, end_frame: int, fps: float) -> list[float]:
        """
        если fps ноль, то время 0.0, 0.0
        Возвращает начало и конец чанка где был обнаружен класс.
        """
        if fps <= 0:
            return [0.0, 0.0]
        return [start_frame / fps, (end_frame + 1) / fps]
