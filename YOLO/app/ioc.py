from collections.abc import Iterator

from dishka import Provider, Scope, provide
from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from YOLO.app.application.interfaces import (
    IYoloInference,
    IYoloProcessProducer,
    IYoloRepository,
)
from YOLO.app.application.services import YoloService
from YOLO.app.config import app_settings
from YOLO.app.infrastructure.database import YoloRepository
from YOLO.app.infrastructure.producers import YoloProcessProducer
from YOLO.app.infrastructure.services import YoloProcessor


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

    yolo_repository = provide(YoloRepository, provides=IYoloRepository)

    yolo_producer = provide(YoloProcessProducer, provides=IYoloProcessProducer)

    # callbacksender


class ApplicationProvider(Provider):
    scope = Scope.APP
    yolo_service = provide(YoloService)


class ImageProvider(Provider):
    scope = Scope.APP
    yolo_processor = provide(YoloProcessor, provides=IYoloInference)
