"""
Запуск приложения
"""

from dishka import make_async_container

from app.ioc import get_api_providers

from .app import create_litestar_app


container = make_async_container(*get_api_providers())

app = create_litestar_app(container)
