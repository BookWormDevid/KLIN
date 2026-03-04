"""
Предоставляет функциональность для чтения переменных окружения,
их парсинга и кэширования с обработкой ошибок.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar, cast

from .exceptions import EnvVariableNotExistError


Tprop = TypeVar("Tprop")


@dataclass
class BaseSettings:
    """
    Базовый класс для настроек приложения с поддержкой переменных окружения.
    """

    env_properties: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def get_env_variable(key: str) -> str:
        """
        Получает значение переменной окружения по ключу.
        """
        value = os.environ.get(key)

        if value is None:
            raise EnvVariableNotExistError(f"{key} does not exist in environment")

        return value

    def resolve_env_property(
        self,
        key: str,
        parser: Callable[[str], Tprop],
        default_value: None | Tprop = None,
    ) -> Tprop:
        """
        Получает и парсит значение переменной окружения с кэшированием.
        При первом обращении читает переменную из окружения, парсит её
        с помощью указанной функции и сохраняет результат в кэше.
        При повторных обращениях возвращает значение из кэша.
        """
        if key in self.env_properties:
            return cast(Tprop, self.env_properties[key])

        try:
            prop = parser(self.get_env_variable(key))
        except EnvVariableNotExistError as err:
            if default_value is None:
                raise err from err
            prop = default_value

        self.env_properties[key] = prop
        return prop
