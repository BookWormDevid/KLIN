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
    app_port: int = 8000

    log_level: Any = "info"

    db_schema: str = "public"
    db_max_overflow: int = 30
    db_statement_timeout: int = 60000
    db_idle_in_transaction_session_timeout: int = 30000

    Klin_queue = "Klin-queue"

    @property
    def database_url(self) -> str:
        """
        Ссылка базы данных
        """
        return self.resolve_env_property("DATABASE_URL", str)

    @property
    def db_pool_size(self) -> bool:
        """
        Размер пула соединений
        """
        return bool(self.resolve_env_property("DB_POOL_SIZE", int, default_value=5))

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
    def klin_secret(self) -> str:
        """
        Ссылка на секреты
        """
        return self.resolve_env_property("KLIN_SECRET", str)


app_settings = Settings()
