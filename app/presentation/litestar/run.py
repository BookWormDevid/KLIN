from dishka import make_async_container

from app.ioc import ApplicationProvider, ImageProvider, InfrastructureProvider

from .app import create_litestar_app

container = make_async_container(
    InfrastructureProvider(), ApplicationProvider(), ImageProvider()
)

app = create_litestar_app()
