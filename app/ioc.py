"""
Модуль конфигурации и провайдеров для приложения.
Определяет подключение к базе данных, RabbitMQ и сервисы через Dishka.
"""

from collections.abc import Iterator
from importlib import import_module
from typing import cast

from dishka import Provider, Scope, provide
from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.application.consumers.stream_event_consumer import StreamEventConsumer
from app.application.interfaces import (
    IKlinCallbackSender,
    IKlinEventProducer,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
    IKlinStream,
    IKlinStreamEventConsumer,
    IKlinVideoStorage,
)
from app.application.services import (  # ← добавили StreamService
    KlinService,
    StreamService,
)
from app.config import app_settings
from app.infrastructure.database import KlinRepository
from app.infrastructure.producers import KlinEventProducer, KlinProcessProducer
from app.infrastructure.services import StreamProcessor  # реализация IKlinStream
from app.infrastructure.services.callback_sender import KlinCallbackSender
from app.infrastructure.services.inference_stub import ApiInferenceStub
from app.infrastructure.services.s3_storage import S3ObjectStorage


class InfrastructureProvider(Provider):
    """
    Общие инфраструктурные зависимости (БД, Rabbit, репозитории, продюсеры).
    """

    scope = Scope.APP

    @provide
    def engine(self) -> Iterator[AsyncEngine]:
        """
        Основной async engine
        """
        db_idle_timeout = app_settings.db_idle_in_transaction_session_timeout
        engine = create_async_engine(
            app_settings.database_url,
            pool_size=app_settings.db_pool_size,
            max_overflow=app_settings.db_max_overflow,
            pool_pre_ping=True,
            connect_args={
                "timeout": app_settings.db_connect_timeout,
                "server_settings": {
                    "idle_in_transaction_session_timeout": f"{db_idle_timeout}",
                    "statement_timeout": f"{app_settings.db_statement_timeout}",
                },
            },
        )
        yield engine

    @provide
    def rabbit_broker(self) -> RabbitBroker:
        """
        Literally just a oneline provider
        """
        return RabbitBroker(app_settings.rabbit_url)

    @provide
    def session(self, engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
        """
        This one is 4 lines and makes a session with async engine...
        """
        return async_sessionmaker(
            bind=engine,
            expire_on_commit=True,
            autoflush=False,
        )

    KLIN_repository = provide(KlinRepository, provides=IKlinRepository)
    KLIN_event_producer = provide(KlinEventProducer, provides=IKlinEventProducer)
    KLIN_producer = provide(KlinProcessProducer, provides=IKlinProcessProducer)
    KLIN_callback_sender = provide(KlinCallbackSender, provides=IKlinCallbackSender)
    KLIN_video_storage = provide(S3ObjectStorage, provides=IKlinVideoStorage)

    KLIN_stream = provide(StreamProcessor, provides=IKlinStream)
    KLIN_stream_event_consumer = provide(
        StreamEventConsumer, provides=IKlinStreamEventConsumer
    )


class ApiApplicationProvider(Provider):
    """Сервисы для API-процесса."""

    scope = Scope.APP
    KLIN_service = provide(KlinService)


class ApiVideoProvider(Provider):
    """Заглушка инференса для API."""

    scope = Scope.APP
    InferenceProcessor = provide(ApiInferenceStub, provides=IKlinInference)


class WorkerApplicationProvider(Provider):
    """Сервисы для queue-worker процесса."""

    scope = Scope.APP
    KLIN_service = provide(KlinService)
    KLIN_stream_service = provide(StreamService)


class WorkerVideoProvider(Provider):
    """Реальный инференс только для worker."""

    scope = Scope.APP

    @provide(provides=IKlinInference)
    def inference_processor(self) -> IKlinInference:
        """Literally a shell"""
        processor_module = import_module(
            "app.infrastructure.services.target.video_processor"
        )
        return cast(IKlinInference, processor_module.InferenceProcessor())


# ====================== ФАБРИКИ ПРОВАЙДЕРОВ ======================
def get_api_providers() -> tuple[Provider, ...]:
    """Провайдеры для API-контейнера."""
    return (
        InfrastructureProvider(),
        ApiApplicationProvider(),
        ApiVideoProvider(),
    )


def get_worker_providers() -> tuple[Provider, ...]:
    """Провайдеры для FastStream worker"""
    return (
        InfrastructureProvider(),
        WorkerApplicationProvider(),
        WorkerVideoProvider(),
    )
