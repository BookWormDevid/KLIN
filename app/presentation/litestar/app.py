# pylint: disable=redefined-outer-name
"""
Настройки litestar, prometheus, swagger, контейнера dishka
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dishka import make_async_container
from dishka.integrations.litestar import setup_dishka
from faststream.rabbit import RabbitBroker
from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.middleware.logging import LoggingMiddlewareConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import SwaggerRenderPlugin
from litestar.plugins.prometheus import PrometheusConfig, PrometheusController
from litestar.plugins.structlog import StructlogConfig, StructlogPlugin
from litestar.static_files import StaticFilesConfig

from app.ioc import ApplicationProvider, InfrastructureProvider, VideoProvider
from app.presentation.litestar.controllers import api_router


# Путь к фронтенду
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncIterator[None]:
    """
    Управляет жизненным циклом приложения.

    Выполняет инициализацию и закрытие ресурсов при запуске/остановке:
    Подключение к RabbitMQ
    Закрытие DI-контейнера
    """
    try:
        container = app.state.dishka_container
        rabbit_broker = await container.get(RabbitBroker)

        await rabbit_broker.connect()
        yield
    finally:
        await app.state.dishka_container.close()


def create_litestar_app(group_path: bool = False) -> Litestar:
    """
    Создаёт и настраивает экземпляр Litestar приложения.
    """
    container = make_async_container(
        InfrastructureProvider(), ApplicationProvider(), VideoProvider()
    )

    prometheus_config = PrometheusConfig(group_path=group_path)

    app = Litestar(
        route_handlers=[api_router, PrometheusController],
        middleware=[prometheus_config.middleware],
        request_max_body_size=200 * 1024 * 1024,
        cors_config=CORSConfig(allow_origins=["*"]),
        openapi_config=OpenAPIConfig(
            title="Klin Inference",
            version="1.0.0",
            path="/api/docs",
            render_plugins=[SwaggerRenderPlugin()],
        ),
        plugins=[
            StructlogPlugin(
                config=StructlogConfig(
                    middleware_logging_config=LoggingMiddlewareConfig(
                        response_log_fields=["status_code", "cookies", "headers"]
                    )
                )
            )
        ],
        static_files_config=[
            StaticFilesConfig(
                path="/frontend",
                directories=[str(FRONTEND_DIR)],
                html_mode=True,
            )
        ],
        lifespan=[lifespan],
        debug=True,
    )

    setup_dishka(container, app)
    app.state.dishka_container = container
    return app
