from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from YOLO.appconfig.base import BaseSettings

load_dotenv()


@dataclass
class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8008

    log_level: Any = "info"

    db_schema: str = "public"
    db_max_overflow: int = 30
    db_statement_timeout: int = 60000
    db_idle_in_transaction_session_timeout: int = 30000

    yolo_queue = "yolo-queue"

    @property
    def database_url(self) -> str:
        return self.resolve_env_property("DATABASE_URL", str)

    @property
    def db_pool_size(self) -> bool:
        return bool(self.resolve_env_property("DB_POOL_SIZE", int, default_value=5))

    @property
    def broker_max_consumers(self) -> int | None:
        v = self.resolve_env_property("BROKER_MAX_CONSUMERS", int, default_value=0)

        return v or None

    @property
    def rabbit_url(self) -> str:
        return self.resolve_env_property("RABBIT_URL", str)

    @property
    def debug(self) -> bool:
        return bool(self.resolve_env_property("DEBUG", int, default_value=0))

    @property
    def yolo_secret(self) -> str:
        return self.resolve_env_property("YOLO_SECRET", str)


app_settings = Settings()
