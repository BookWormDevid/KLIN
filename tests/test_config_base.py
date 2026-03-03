import os
import unittest
from unittest.mock import patch

from app.config.base import BaseSettings
from app.config.exceptions import EnvVariableNotExistError


class TestBaseSettings(unittest.TestCase):
    def test_get_env(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "value"}, clear=True):
            self.assertEqual(BaseSettings.get_env_variable("TEST_KEY"), "value")

    def test_get_env_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvVariableNotExistError):
                BaseSettings.get_env_variable("MISSING_KEY")

    def test_resolve_default(self) -> None:
        settings = BaseSettings()
        with patch.dict(os.environ, {}, clear=True):
            value = settings.resolve_env_property("MISSING_KEY", int, default_value=7)
        self.assertEqual(value, 7)

    def test_resolve_missing(self) -> None:
        settings = BaseSettings()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvVariableNotExistError):
                settings.resolve_env_property("MISSING_KEY", int)

    def test_resolve_cache(self) -> None:
        settings = BaseSettings()

        with patch.object(settings, "get_env_variable", return_value="42") as getter:
            first = settings.resolve_env_property("NUMBER", int)
            second = settings.resolve_env_property("NUMBER", int)

        self.assertEqual(first, 42)
        self.assertEqual(second, 42)
        getter.assert_called_once_with("NUMBER")
