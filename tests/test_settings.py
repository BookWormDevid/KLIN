"""
Тестирование настроек приложения.
"""

import os
import unittest
from unittest.mock import patch

from app.config.exceptions import EnvVariableNotExistError
from app.config.settings import Settings


class TestSettings(unittest.TestCase):
    """Тесты для объекта настроек."""

    def test_videomae_default(self) -> None:
        settings = Settings()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.videomae_path, "models/videomae-ucf-crime")

    def test_videomae_env(self) -> None:
        settings = Settings()
        with patch.dict(os.environ, {"VIDEOMAE_PATH": "models/custom"}, clear=True):
            self.assertEqual(settings.videomae_path, "models/custom")

    def test_database_url(self) -> None:
        settings = Settings()
        with patch.dict(
            os.environ,
            {"DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/db"},
            clear=True,
        ):
            self.assertEqual(
                settings.database_url,
                "postgresql+asyncpg://user:pass@localhost:5432/db",
            )

    def test_rabbit_url(self) -> None:
        settings = Settings()
        with patch.dict(
            os.environ,
            {"RABBIT_URL": "amqp://user:pass@localhost:5672/"},
            clear=True,
        ):
            self.assertEqual(settings.rabbit_url, "amqp://user:pass@localhost:5672/")

    def test_broker_max_consumers_zero(self) -> None:
        settings = Settings()
        with patch.dict(os.environ, {"BROKER_MAX_CONSUMERS": "0"}, clear=True):
            self.assertIsNone(settings.broker_max_consumers)

    def test_broker_max_consumers_value(self) -> None:
        settings = Settings()
        with patch.dict(os.environ, {"BROKER_MAX_CONSUMERS": "4"}, clear=True):
            self.assertEqual(settings.broker_max_consumers, 4)

    def test_debug(self) -> None:
        settings = Settings()
        with patch.dict(os.environ, {"DEBUG": "1"}, clear=True):
            self.assertTrue(settings.debug)

        settings = Settings()
        with patch.dict(os.environ, {"DEBUG": "0"}, clear=True):
            self.assertFalse(settings.debug)

        settings = Settings()
        with patch.dict(os.environ, {"DEBUG": "on"}, clear=True):
            self.assertTrue(settings.debug)

        settings = Settings()
        with patch.dict(os.environ, {"DEBUG": "off"}, clear=True):
            self.assertFalse(settings.debug)

    def test_cors_allowed_origins(self) -> None:
        settings = Settings()
        with patch.dict(
            os.environ,
            {"CORS_ALLOWED_ORIGINS": "https://a.example, https://b.example"},
            clear=True,
        ):
            self.assertEqual(
                settings.cors_allowed_origins,
                ["https://a.example", "https://b.example"],
            )

    def test_db_pool_size(self) -> None:
        settings_default = Settings()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings_default.db_pool_size, 5)

        settings_custom = Settings()
        with patch.dict(os.environ, {"DB_POOL_SIZE": "30"}, clear=True):
            self.assertEqual(settings_custom.db_pool_size, 30)

        settings_invalid = Settings()
        with patch.dict(os.environ, {"DB_POOL_SIZE": "0"}, clear=True):
            with self.assertRaises(ValueError):
                _ = settings_invalid.db_pool_size

    def test_klin_secret_missing(self) -> None:
        settings = Settings()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvVariableNotExistError):
                _ = settings.klin_secret
