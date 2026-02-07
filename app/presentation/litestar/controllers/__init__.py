from litestar import Router

from .v1 import MAEController

api_router = Router(
    path="/api/v1",
    route_handlers=[MAEController],
)

__all__ = ("api_router", "MAEController")
