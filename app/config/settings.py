# pylint: disable=R0904
"""
Настройки приложения с валидацией
"""

from datetime import timedelta
from typing import Any

from dotenv import load_dotenv

from app.config.base import BaseSettings


load_dotenv()


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
    default_db_connect_timeout: float = 10.0

    default_videomae_path: str = "models/videomae-ucf-crime"
    default_yolo_path: str = "models/yolov8x.pt"
    default_x3d_path: str = "models/pre_trained_x3d_model.pt"

    default_cors_allowed_origins: str = (
        "http://localhost,http://127.0.0.1,http://localhost:3000,http://127.0.0.1:3000"
    )
    default_max_retry_attemps: int = 1
    default_s3_region: str = "us-east-1"
    default_s3_addressing_style: str = "path"
    default_s3_key_prefix: str = "klin/uploads"
    default_keep_s3_source_objects: bool = True
    default_batch_s3_prefix: str = "klin/batch"
    default_batch_file_extensions: str = ".mp4,.avi,.mov,.mkv,.wmv,.webm"
    default_jwt_token_ttl_minutes: int = 60

    default_triton_grpc_url: str = "0.0.0.0:8001"

    Klin_queue = "Klin-queue"
    Klin_process_queue = "Klin-stream-queue"
    Klin_stream_event_queue = "Klin-stream-event-queue"

    @staticmethod
    def parse_bool_env(value: str) -> bool:
        """
        Парсит бред в bool
        """

        normalized = value.strip().lower()

        if normalized in {"1", "true", "yes", "on"}:
            return True

        if normalized in {"0", "false", "no", "off"}:
            return False

        raise ValueError(f"Unsupported boolean value: {value}")

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
    def db_connect_timeout(self) -> float:
        """
        Таймаут установки соединения с БД.
        """
        timeout = self.resolve_env_property(
            "DB_CONNECT_TIMEOUT",
            float,
            default_value=self.default_db_connect_timeout,
        )
        if timeout <= 0:
            raise ValueError("DB_CONNECT_TIMEOUT must be > 0")
        return timeout

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
    def s3_endpoint_url(self) -> str:
        """
        URL S3-совместимого endpoint'а.
        """
        return self.resolve_env_property("S3_ENDPOINT_URL", str)

    @property
    def s3_bucket_name(self) -> str:
        """
        Имя bucket'а для загруженных видео.
        """
        return self.resolve_env_property("S3_BUCKET_NAME", str)

    @property
    def s3_access_key_id(self) -> str:
        """
        Идентификатор access key для S3.
        """
        return self.resolve_env_property("S3_ACCESS_KEY_ID", str)

    @property
    def s3_secret_access_key(self) -> str:
        """
        Секретный ключ доступа для S3.
        """
        return self.resolve_env_property("S3_SECRET_ACCESS_KEY", str)

    @property
    def s3_region(self) -> str:
        """
        Регион S3 по умолчанию.
        """
        return self.resolve_env_property(
            "S3_REGION",
            str,
            default_value=self.default_s3_region,
        )

    @property
    def s3_addressing_style(self) -> str:
        """
        Стиль адресации S3 для boto3.
        """
        style = self.resolve_env_property(
            "S3_ADDRESSING_STYLE",
            str,
            default_value=self.default_s3_addressing_style,
        ).strip()
        if style not in {"path", "virtual"}:
            raise ValueError("S3_ADDRESSING_STYLE must be 'path' or 'virtual'")
        return style

    @property
    def s3_key_prefix(self) -> str:
        """
        Префикс для ключей загруженных объектов.
        """
        return (
            self.resolve_env_property(
                "S3_KEY_PREFIX",
                str,
                default_value=self.default_s3_key_prefix,
            )
            .strip()
            .strip("/")
        )

    @property
    def keep_s3_source_objects(self) -> bool:
        """
        Keep original S3 objects after offline processing.
        """
        return self.resolve_env_property(
            "KEEP_S3_SOURCE_OBJECTS",
            self.parse_bool_env,
            default_value=self.default_keep_s3_source_objects,
        )

    @property
    def batch_s3_prefix(self) -> str:
        """
        Base S3 prefix for date-partitioned batch runs.
        """
        return (
            self.resolve_env_property(
                "KLIN_BATCH_S3_PREFIX",
                str,
                default_value=self.default_batch_s3_prefix,
            )
            .strip()
            .strip("/")
        )

    @property
    def batch_file_extensions(self) -> tuple[str, ...]:
        """
        Allowed file extensions for S3 batch discovery.
        """
        raw_value = self.resolve_env_property(
            "KLIN_BATCH_FILE_EXTENSIONS",
            str,
            default_value=self.default_batch_file_extensions,
        )
        extensions = tuple(
            part.strip().lower() for part in raw_value.split(",") if part.strip()
        )
        if not extensions:
            raise ValueError(
                "KLIN_BATCH_FILE_EXTENSIONS must contain at least one value"
            )
        return extensions

    @property
    def triton_url(self) -> str:
        """
        URL сервера с моделями к которому будет обращаться воркер
        """
        return self.resolve_env_property(
            "TRITON_GRPC_URL", str, default_value=self.default_triton_grpc_url
        )

    @property
    def debug(self) -> bool:
        """
        Режим отладки
        """
        return self.resolve_env_property(
            "DEBUG", self.parse_bool_env, default_value=False
        )

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
    def jwt_secret(self) -> str:
        """
        Secret used to sign and validate API JWT tokens.
        """
        return self.resolve_env_property(
            "JWT_SECRET",
            str,
        )

    @property
    def jwt_token_ttl(self) -> timedelta:
        """
        JWT access token lifetime.
        """
        ttl_minutes = self.resolve_env_property(
            "JWT_TOKEN_TTL_MINUTES",
            int,
            default_value=self.default_jwt_token_ttl_minutes,
        )
        if ttl_minutes <= 0:
            raise ValueError("JWT_TOKEN_TTL_MINUTES must be > 0")
        return timedelta(minutes=ttl_minutes)


app_settings = Settings()
