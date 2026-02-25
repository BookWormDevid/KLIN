"""
Объединяет все контроллеры версии 1 API под общим префиксом /api/v1.
"""

from litestar import Router

from .v1 import KlinController

api_router = Router(
    path="/api/v1",
    route_handlers=[KlinController],
)

__all__ = ("api_router", "KlinController")
