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
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
)
from app.application.services import KlinService
from app.config import app_settings
from app.infrastructure.database import KlinRepository
from app.infrastructure.producers import KlinProcessProducer
from app.infrastructure.services import KlinProcessor


class InfrastructureProvider(Provider):
    scope = Scope.APP

    @provide
    def engine(self) -> Iterator[AsyncEngine]:
        engine = create_async_engine(
            app_settings.database_url,
            pool_size=app_settings.db_pool_size,
            max_overflow=app_settings.db_max_overflow,
            pool_pre_ping=True,
            connect_args={
                "server_settings": {
                    "idle_in_transaction_session_timeout": f"{app_settings.db_idle_in_transaction_session_timeout}",
                    "statement_timeout": f"{app_settings.db_statement_timeout}",
                }
            },
        )
        yield engine

    @provide
    def rabbit_broker(self) -> RabbitBroker:
        return RabbitBroker(
            app_settings.rabbit_url,
        )

    @provide
    def session(self, engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
        return async_sessionmaker(bind=engine, expire_on_commit=True, autoflush=False)

    klin_repository = provide(KlinRepository, provides=IKlinRepository)

    klin_producer = provide(KlinProcessProducer, provides=IKlinProcessProducer)

    # callbacksender


class ApplicationProvider(Provider):
    scope = Scope.APP
    klin_service = provide(KlinService)


class ImageProvider(Provider):
    scope = Scope.APP
    klin_processor = provide(KlinProcessor, provides=IKlinInference)
