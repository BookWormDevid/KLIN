from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from app.application.dto import StreamEventDto


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)


class DummyEventProducer:
    """Просто логирует события — идеально для теста с видео-файлом."""

    async def send_event(self, event: StreamEventDto) -> None:
        logger.info(
            "📨 EVENT [%s] camera=%s stream=%s | %s",
            event.type,
            event.camera_id,
            event.stream_id,
            event.payload,
        )


async def test():
    from app.infrastructure.services import StreamProcessor
    from app.models.klin import KlinStreamingModel

    producer = DummyEventProducer()
    processor = StreamProcessor(event_producer=producer)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    test_file = "./tests/videos/svoi_hun_syn.avi"
    model = KlinStreamingModel(
        id=uuid.uuid4(),
        camera_url=test_file,
        camera_id="LOCAL_TEST_001",
        state="PENDING",
    )

    logging.info("Запуск тестового стриминга с файлом: %s", test_file)

    try:
        await processor.streaming_analyze(model)
    except KeyboardInterrupt:
        logging.info("Остановка по Ctrl+C")
        processor.stop("LOCAL_TEST_001")
    except Exception:
        logging.exception("Критическая ошибка во время теста")


if __name__ == "__main__":
    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
