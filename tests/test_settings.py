# pylint: disable="import-error"
"""
Тестирование конфига приложения
"""

import os
import unittest
from unittest.mock import patch

from app.config.exceptions import EnvVariableNotExistError
from app.config.settings import Settings


class TestSettings(unittest.TestCase):
    """
    Тестируется:
    путь к модели,
    функции пути к модели,
    путь к бд,
    путь к rabbit,
    минимальное и максимальное число подписчиков,
    выполняется ли отладка,
    размер пула соединения,
    есть ли секретное поле

    """

    def test_videomae_default(self) -> None:
        """
        Тест путь к модели
        """
        settings = Settings()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.videomae_path, "models/videomae-UCF-crime")

    def test_videomae_env(self) -> None:
        """
        Тест функции пути к модели через функцию
        """
        settings = Settings()
        with patch.dict(os.environ, {"VIDEOMAE_PATH": "models/custom"}, clear=True):
            self.assertEqual(settings.videomae_path, "models/custom")

    def test_database_url(self) -> None:
        """
        Тест пути к бд
        """
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
        """
        Тест пути к rabbit
        """
        settings = Settings()
        with patch.dict(
            os.environ,
            {"RABBIT_URL": "amqp://user:pass@localhost:5672/"},
            clear=True,
        ):
            self.assertEqual(settings.rabbit_url, "amqp://user:pass@localhost:5672/")

    def test_broker_max_consumers_zero(self) -> None:
        """
        Тест если подписчиков 0
        """
        settings = Settings()
        with patch.dict(os.environ, {"BROKER_MAX_CONSUMERS": "0"}, clear=True):
            self.assertIsNone(settings.broker_max_consumers)

    def test_broker_max_consumers_value(self) -> None:
        """
        Тест если подписчиков 4
        """
        settings = Settings()
        with patch.dict(os.environ, {"BROKER_MAX_CONSUMERS": "4"}, clear=True):
            self.assertEqual(settings.broker_max_consumers, 4)

    def test_debug(self) -> None:
        """
        Тест если отладка включена и выключена
        """
        settings = Settings()
        with patch.dict(os.environ, {"DEBUG": "1"}, clear=True):
            self.assertTrue(settings.debug)

        settings = Settings()
        with patch.dict(os.environ, {"DEBUG": "0"}, clear=True):
            self.assertFalse(settings.debug)

    def test_db_pool_size(self) -> None:
        """
        Тест пула подключения
        """
        settings_default = Settings()
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(settings_default.db_pool_size)

        settings_zero = Settings()
        with patch.dict(os.environ, {"DB_POOL_SIZE": "0"}, clear=True):
            self.assertFalse(settings_zero.db_pool_size)

    def test_klin_secret_missing(self) -> None:
        """
        Тест на наличие секрета
        """

        settings = Settings()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvVariableNotExistError):
                _ = settings.klin_secret
