"""
Настройки приложения с валидацией
"""

from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from app.config.base import BaseSettings


load_dotenv()


@dataclass
class Settings(BaseSettings):
    """
    В классе представлены методы для передачи настроек
    """

    app_host: str = "0.0.0.0"
    app_port: int = 8008

    log_level: Any = "info"

    db_schema: str = "public"
    db_max_overflow: int = 30
    db_statement_timeout: int = 60000
    db_idle_in_transaction_session_timeout: int = 30000

    default_videomae_path: str = "models/videomae-UCF-crime"
    default_yolo_path: str = "models/yolov8x.pt"
    default_x3d_path: str = "models/pre_trained_x3d_model.pt"

    default_cors_allowed_origins: str = (
        "http://localhost,http://127.0.0.1,http://localhost:3000,http://127.0.0.1:3000"
    )
    default_max_retry_attemps: int = 1

    Klin_queue = "Klin-queue"

    @property
    def videomae_path(self) -> str:
        """
        Источник videomae
        """
        return self.resolve_env_property(
            "VIDEOMAE_PATH", str, default_value=self.default_videomae_path
        )

    @property
    def yolo_path(self) -> str:
        """
        Источник yolo
        """
        return self.resolve_env_property(
            "YOLO_PATH", str, default_value=self.default_yolo_path
        )

    @property
    def x3d_path(self) -> str:
        """
        Источник x3d
        """
        return self.resolve_env_property(
            "X3D_PATH", str, default_value=self.default_x3d_path
        )

    @property
    def database_url(self) -> str:
        """
        Ссылка базы данных
        """
        return self.resolve_env_property("DATABASE_URL", str)

    @property
    def db_pool_size(self) -> int:
        """
        Размер пула соединений
        """
        pool_size = self.resolve_env_property("DB_POOL_SIZE", int, default_value=5)
        if pool_size < 1:
            raise ValueError("DB_POOL_SIZE must be >= 1")
        return pool_size

    @property
    def broker_max_consumers(self) -> int | None:
        """
        Максимальное число подписчиков в broker
        """
        v = self.resolve_env_property("BROKER_MAX_CONSUMERS", int, default_value=0)

        return v or None

    @property
    def rabbit_url(self) -> str:
        """
        Ссылка на rabbit
        """
        return self.resolve_env_property("RABBIT_URL", str)

    @property
    def debug(self) -> bool:
        """
        Режим отладки
        """
        return bool(self.resolve_env_property("DEBUG", int, default_value=0))

    @property
    def cors_allowed_origins(self) -> list[str]:
        """
        Разрешенные источники CORS.
        """
        raw_origins = self.resolve_env_property(
            "CORS_ALLOWED_ORIGINS",
            str,
            default_value=self.default_cors_allowed_origins,
        )
        origins = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]
        if not origins:
            raise ValueError("CORS_ALLOWED_ORIGINS must contain at least one origin")
        return origins

    @property
    def max_retry_attempts(self) -> int:
        """
        Общая настройка для попыток выполнения
        """
        return self.resolve_env_property(
            "MAX_RETRY_ATTEMPTS", int, default_value=self.default_max_retry_attemps
        )

    @property
    def klin_secret(self) -> str:
        """
        Ссылка на секреты
        """
        return self.resolve_env_property("KLIN_SECRET", str)


app_settings = Settings()
