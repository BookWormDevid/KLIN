import asyncio
import logging
from dataclasses import dataclass

from app.infrastructure.services.target import StreamProcessor  # ваш путь


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class FakeModel:
    """Заглушка вместо KlinStreamingModel, чтобы быстро протестировать"""

    camera_url: str
    camera_id: str = "TEST_CAM_001"


async def test_stream():
    processor = StreamProcessor()

    # Вариант 1 — локальный видеофайл (самый надёжный для отладки)
    test_file = r"C:\Users\meksi\Documents\GitHub\fi004.mp4"
    # test_file = "short_test.mp4"   # положите короткий клип рядом со скриптом

    model = FakeModel(camera_url=test_file, camera_id="LOCAL_TEST_001")

    logging.info("Запуск тестового стриминга с файлом: %s", test_file)

    try:
        await processor.streaming_analyze(model)
    except KeyboardInterrupt:
        logging.info("Остановка по Ctrl+C — нормальное завершение")
    except Exception:
        logging.exception("Критическая ошибка во время теста")


if __name__ == "__main__":
    try:
        asyncio.run(test_stream())
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
