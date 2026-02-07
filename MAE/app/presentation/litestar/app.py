from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dishka import make_async_container
from dishka.integrations.litestar import setup_dishka
from faststream.rabbit import RabbitBroker
from litestar import Litestar, Router
from litestar.config.cors import CORSConfig
from litestar.middleware.logging import LoggingMiddlewareConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import SwaggerRenderPlugin
from litestar.plugins.structlog import StructlogConfig, StructlogPlugin
from litestar.static_files import StaticFilesConfig

from MAE.app.ioc import ApplicationProvider, InfrastructureProvider, VideoProvider
from MAE.app.presentation.litestar.controllers import MAEController

# Роутер с префиксом /api/v1
api_v1_router = Router(
    path="/api/v1",
    route_handlers=[MAEController],
)

# Путь к фронтенду
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncIterator[None]:
    container = app.state.dishka_container
    rabbit_broker = await container.get(RabbitBroker)
    await rabbit_broker.connect()
    try:
        yield
    finally:
        await rabbit_broker.close()
        container.close()


def create_litestar_app() -> Litestar:
    container = make_async_container(
        InfrastructureProvider(), ApplicationProvider(), VideoProvider()
    )

    app = Litestar(
        route_handlers=[api_v1_router],
        cors_config=CORSConfig(allow_origins=["*"]),
        openapi_config=OpenAPIConfig(
            title="MAE Inference",
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


app = create_litestar_app()
