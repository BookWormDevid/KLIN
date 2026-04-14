"""
Настройки litestar, prometheus, swagger, контейнера dishka
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dishka import AsyncContainer
from dishka.integrations.litestar import setup_dishka as setup_litestar_dishka
from faststream.rabbit import RabbitBroker
from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.middleware.logging import LoggingMiddlewareConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import SwaggerRenderPlugin
from litestar.plugins.prometheus import PrometheusConfig, PrometheusController
from litestar.plugins.structlog import StructlogConfig, StructlogPlugin
from litestar.static_files import create_static_files_router

from app.config import app_settings
from app.presentation.litestar.controllers import api_router


logger = logging.getLogger(__name__)


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
        container: AsyncContainer = app.state.dishka_container
        rabbit_broker = await container.get(RabbitBroker)

        try:
            await rabbit_broker.connect()
        except Exception:
            logger.warning(
                "Rabbit broker is unavailable during API startup. "
                "HTTP endpoints will stay up, but enqueue operations may fail "
                "until RabbitMQ becomes reachable.",
                exc_info=True,
            )
        yield
    finally:
        await app.state.dishka_container.close()


def create_litestar_app(
    container: AsyncContainer, group_path: bool = False
) -> Litestar:
    """
    Создаёт и настраивает экземпляр Litestar приложения.
    """
    prometheus_config = PrometheusConfig(group_path=group_path)

    app = Litestar(
        route_handlers=[
            api_router,
            PrometheusController,
            create_static_files_router(
                path="/frontend",
                directories=[str(FRONTEND_DIR)],
                html_mode=True,
                name="frontend",
            ),
        ],
        middleware=[prometheus_config.middleware],
        request_max_body_size=200 * 1024 * 1024,
        cors_config=CORSConfig(allow_origins=app_settings.cors_allowed_origins),
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
        lifespan=[lifespan],
        debug=app_settings.debug,
    )

    setup_litestar_dishka(container, app)
    return app
