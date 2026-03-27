"""
Набор unit-тестов для класса BaseSettings.
"""

import os
import unittest
from unittest.mock import patch

from app.config.base import BaseSettings
from app.config.exceptions import EnvVariableNotExistError


class TestBaseSettings(unittest.TestCase):
    """
    Проверяем логику получения переменных окружения и обработки значений по умолчанию.
    """

    def test_get_env(self) -> None:
        """
        Проверяет, что метод get_env_variable корректно возвращает значение
        существующей переменной окружения.
        """
        # Мокаем окружение: только одна переменная TEST_KEY
        with patch.dict(os.environ, {"TEST_KEY": "value"}, clear=True):
            # Ожидаем, что метод вернёт именно это значение
            self.assertEqual(BaseSettings.get_env_variable("TEST_KEY"), "value")

    def test_get_env_missing(self) -> None:
        """
        Проверяет, что при отсутствии переменной в окружении
        выбрасывается ожидаемое исключение EnvVariableNotExistError.
        """
        # Полностью чистое окружение (ни одной переменной)
        with patch.dict(os.environ, {}, clear=True):
            # Ожидаем, что будет выброшено исключение
            with self.assertRaises(EnvVariableNotExistError):
                BaseSettings.get_env_variable("MISSING_KEY")

    def test_resolve_default(self) -> None:
        """
        Проверяет работу метода resolve_env_property:
        если переменная отсутствует → возвращается значение по умолчанию.
        """
        settings = BaseSettings()

        # Чистое окружение — переменной нет
        with patch.dict(os.environ, {}, clear=True):
            # Запрашиваем int-переменную, но с default_value=7
            value = settings.resolve_env_property("MISSING_KEY", int, default_value=7)

        # Должно вернуться значение по умолчанию
        self.assertEqual(value, 7)

    def test_resolve_missing(self) -> None:
        """
        Проверяет, что если переменная отсутствует И НЕ указан default_value,
        то выбрасывается исключение EnvVariableNotExistError.
        """
        settings = BaseSettings()

        with patch.dict(os.environ, {}, clear=True):
            # Без default_value → должно упасть
            with self.assertRaises(EnvVariableNotExistError):
                settings.resolve_env_property("MISSING_KEY", int)

    def test_resolve_cache(self) -> None:
        """
        Проверяет, что значение переменной кэшируется внутри экземпляра BaseSettings.
        Повторный вызов resolve_env_property с тем же ключом НЕ должен
        заново обращаться к os.environ (getter вызывается только один раз).
        """
        settings = BaseSettings()

        # Мокаем метод get_env_variable, чтобы посчитать, сколько раз он вызван
        with patch.object(settings, "get_env_variable", return_value="42") as getter:
            # Первый вызов → должен реально запросить переменную
            first = settings.resolve_env_property("NUMBER", int)

            # Второй вызов с тем же ключом → должен взять из кэша
            second = settings.resolve_env_property("NUMBER", int)

        # Оба значения должны быть одинаковыми
        self.assertEqual(first, 42)
        self.assertEqual(second, 42)

        # Самое важное: метод get_env_variable вызван ТОЛЬКО ОДИН раз
        getter.assert_called_once_with("NUMBER")
