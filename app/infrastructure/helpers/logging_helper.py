"""
Логирование
"""

import logging as logger
from typing import Any


class LoggingHelper:
    """
    Класс для логирования чего-угодно
    """

    @staticmethod
    def log_processing(
        video_name: str, video_info: dict[str, Any], processing_time: float
    ) -> None:
        logger.info(
            "РЕЗУЛЬТАТЫ АНАЛИЗА ВИДЕО: video=%s "
            "duration=%.1fs processing=%.2fs frames=%d/%d",
            video_name,
            float(video_info["duration"]),
            processing_time,
            int(video_info["frames_read"]),
            int(video_info["total_frames"]),
        )

    @staticmethod
    def build_video_info(
        *,
        total_frames: int,
        fps: float,
        duration: float,
        frames_read: int,
    ) -> dict[str, Any]:
        """
        Объединяет:
        кадры, fps, длительность и прочитанные кадры в один словарь
        """
        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "frames_read": frames_read,
        }
