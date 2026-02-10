from dishka import make_async_container

from app.ioc import ApplicationProvider, InfrastructureProvider, VideoProvider

from .app import create_litestar_app

container = make_async_container(
    InfrastructureProvider(), ApplicationProvider(), VideoProvider()
)

app = create_litestar_app()
