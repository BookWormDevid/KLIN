"""
Объединяет все контроллеры версии 1 API под общим префиксом /api/v1.
"""

from litestar import Router

from .v1 import KlinController, KlinStreamController


api_router = Router(
    path="/api/v1",
    route_handlers=[KlinController, KlinStreamController],
)

__all__ = ("api_router", "KlinController", "KlinStreamController")
