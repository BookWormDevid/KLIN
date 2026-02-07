from litestar import Router

from .v1 import YoloController

api_router = Router(
    path="/api/v1",
    route_handlers=[YoloController],
)

__all__ = ("api_router", "YoloController")
