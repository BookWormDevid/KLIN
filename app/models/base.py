# pylint: disable=too-few-public-methods
# pylint: disable=unnecessary-lambda
"""
Базовая модель бд
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, registry


mapper_registry = registry(metadata=MetaData())


class Model(DeclarativeBase):
    """
    Инициализация базы данных
    """

    registry = mapper_registry
    metadata = mapper_registry.metadata

    __abstract__ = True


class BaseModel(Model):
    """
    Колонки базовой модели
    id
    is_removed поле мягкого удаления
    created_at когда создано новое поле
    updated_at когда поле обновлено последний раз
    """

    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, unique=True, default=uuid.uuid4
    )

    is_removed: Mapped[bool] = mapped_column(default=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: get_native_utc_now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: get_native_utc_now(), onupdate=lambda: get_native_utc_now()
    )


def get_native_utc_now() -> datetime:
    """
    Получает текущее время в UTC без часового пояса
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
