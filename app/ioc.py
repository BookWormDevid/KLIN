"""
Модуль конфигурации и провайдеров для приложения.
Определяет подключение к базе данных, RabbitMQ и сервисы
через систему инъекции зависимостей Dishka.
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

from app.application.interfaces import (
    IKlinCallbackSender,
    IKlinEventProducer,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
    IKlinVideoStorage,
)
from app.application.services import KlinService
from app.config import app_settings
from app.infrastructure.database import KlinRepository
from app.infrastructure.producers import KlinEventProducer, KlinProcessProducer
from app.infrastructure.services.callback_sender import KlinCallbackSender
from app.infrastructure.services.inference_stub import ApiInferenceStub
from app.infrastructure.services.s3_storage import S3ObjectStorage


class InfrastructureProvider(Provider):
    """
    Провайдер инфраструктуры приложения.
    Создает движок базы данных, сессии, брокера сообщений
    и инфраструктурные сервисы.
    """

    scope = Scope.APP

    @provide
    def engine(self) -> Iterator[AsyncEngine]:
        """
        Создает и возвращает асинхронный движок базы данных.
        """
        db_idle_timeout = app_settings.db_idle_in_transaction_session_timeout

        engine = create_async_engine(
            app_settings.database_url,
            pool_size=app_settings.db_pool_size,
            max_overflow=app_settings.db_max_overflow,
            pool_pre_ping=True,
            connect_args={
                "server_settings": {
                    "idle_in_transaction_session_timeout": f"{db_idle_timeout}",
                    "statement_timeout": f"{app_settings.db_statement_timeout}",
                }
            },
        )
        yield engine

    @provide
    def rabbit_broker(self) -> RabbitBroker:
        """
        Создает и возвращает брокера сообщений RabbitMQ.
        """
        return RabbitBroker(app_settings.rabbit_url)

    @provide
    def session(self, engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
        """
        Создает фабрику асинхронных сессий для работы с базой данных.
        """
        return async_sessionmaker(bind=engine, expire_on_commit=True, autoflush=False)

    KLIN_repository = provide(KlinRepository, provides=IKlinRepository)
    KLIN_event_producer = provide(KlinEventProducer, provides=IKlinEventProducer)
    KLIN_producer = provide(KlinProcessProducer, provides=IKlinProcessProducer)
    KLIN_callback_sender = provide(KlinCallbackSender, provides=IKlinCallbackSender)
    KLIN_video_storage = provide(S3ObjectStorage, provides=IKlinVideoStorage)


class ApiApplicationProvider(Provider):
    """Провайдер сервисов API-процесса."""

    scope = Scope.APP
    KLIN_service = provide(KlinService)


class ApiVideoProvider(Provider):
    """Провайдеры-заглушки для API, который не запускает инференс локально."""

    scope = Scope.APP
    InferenceProcessor = provide(ApiInferenceStub, provides=IKlinInference)


class WorkerApplicationProvider(Provider):
    """Провайдер сервисов queue-worker процесса."""

    scope = Scope.APP
    KLIN_service = provide(KlinService)


class WorkerVideoProvider(Provider):
    """Провайдеры реальных инференс-сервисов для queue-worker процесса."""

    scope = Scope.APP

    @provide(provides=IKlinInference)
    def inference_processor(self) -> IKlinInference:
        """Создает инференс-процессор только в worker-контейнере."""

        processor_module = import_module(
            "app.infrastructure.services.target.video_processor"
        )
        return cast(IKlinInference, processor_module.InferenceProcessor())


def get_api_providers() -> tuple[Provider, ...]:
    """Возвращает набор провайдеров для API-контейнера."""

    return (
        InfrastructureProvider(),
        ApiApplicationProvider(),
        ApiVideoProvider(),
    )


def get_worker_providers() -> tuple[Provider, ...]:
    """Возвращает набор провайдеров для queue-worker контейнера."""

    return (
        InfrastructureProvider(),
        WorkerApplicationProvider(),
        WorkerVideoProvider(),
    )
