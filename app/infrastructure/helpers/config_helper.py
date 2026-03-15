"""
Класс с конфигами для процессора
"""

from dataclasses import dataclass, field


@dataclass
class StreamConfig:
    """Настройки потоковой обработки."""

    chunk_size: int = 16
    frame_size: tuple[int, int] = (224, 224)


@dataclass
class YoloConfig:
    """Настройки инференса YOLO."""

    yolo_stride: int = 2
    yolo_batch_size: int = 32
    yolo_conf: float = 0.6
    yolo_classes: dict[int, str] = field(default_factory=lambda: {0: "person"})
    allowed_classes: set[int] = field(default_factory=lambda: {0})


@dataclass
class MaeConfig:
    """Настройки классификации VideoMAE."""

    mae_classes: dict[int, str] = field(
        default_factory=lambda: {
            0: "Abuse",
            1: "Arrest",
            2: "Arson",
            3: "Assault",
            4: "Burglary",
            5: "Explosion",
            6: "Fighting",
            7: "Normal",
            8: "RoadAccident",
            9: "Robbery",
            10: "Shooting",
            11: "Shoplifting",
            12: "Stealing",
            13: "Vandalism",
        }
    )
