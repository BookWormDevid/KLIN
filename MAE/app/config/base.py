import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .exceptions import EnvVariableNotExistError

TProp = TypeVar("TProp")


@dataclass
class BaseSettings:
    env_properties: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def get_env_variable(key: str) -> str:
        value = os.environ.get(key)

        if value is None:
            raise EnvVariableNotExistError(f"{key} does not exist in environment")

        return value

    def resolve_env_property(
        self,
        key: str,
        parser: Callable[[str], TProp],
        default_value: None | TProp = None,
    ) -> TProp:
        if key in self.env_properties:
            return self.env_properties[key]  # type: ignore

        try:
            prop = parser(self.get_env_variable(key))
        except EnvVariableNotExistError as err:
            if default_value is None:
                raise err from err
            prop = default_value

        self.env_properties[key] = prop
        return prop
