"""
Модуль конфигурации и провайдеров для приложения.
Определяет подключение к базе данных, RabbitMQ и сервисы
через систему инъекции зависимостей Dishka.
"""

from collections.abc import Iterator

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
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
)
from app.application.services import KlinService
from app.config import app_settings
from app.infrastructure.database import KlinRepository
from app.infrastructure.producers import KlinProcessProducer
from app.infrastructure.services import InferenceProcessor, KlinCallbackSender


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
        return RabbitBroker(
            app_settings.rabbit_url,
        )

    @provide
    def session(self, engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
        """
        Создает фабрику асинхронных сессий для работы с базой данных.
        """
        return async_sessionmaker(bind=engine, expire_on_commit=True, autoflush=False)

    MAE_repository = provide(KlinRepository, provides=IKlinRepository)
    MAE_producer = provide(KlinProcessProducer, provides=IKlinProcessProducer)
    MAE_callback_sender = provide(KlinCallbackSender, provides=IKlinCallbackSender)


class ApplicationProvider(Provider):
    """
    Провайдер сервисов приложения.
    """

    scope = Scope.APP
    MAE_service = provide(KlinService)


class VideoProvider(Provider):
    """
    Провайдер для обработки видео и инференса MAE и Yolo.
    """

    scope = Scope.APP
    InferenceProcessor = provide(InferenceProcessor, provides=IKlinInference)
