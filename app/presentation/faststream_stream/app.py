"""
FastStream Worker — обработка стриминговых задач (реал-тайм CV).
Очередь: stream_process_queue
"""

import logging

import msgspec
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage
from prometheus_client import Counter, Histogram, start_http_server

from app.application.dto import StreamProcessDto
from app.application.services import StreamService
from app.config import app_settings
from app.ioc import get_worker_providers


logger = logging.getLogger(__name__)

# ====================== Dependency Injection ======================
container = make_container(*get_worker_providers())
broker: RabbitBroker = container.get(RabbitBroker)
app = FastStream(broker)  # ← вот эта переменная должна быть!

stream_service: StreamService = container.get(StreamService)

# ====================== Prometheus метрики ======================
STREAM_PROCESSED = Counter(
    "stream_processed_total",
    "Количество обработанных стриминговых задач",
    ["status"],
)

STREAM_PROCESSING_TIME = Histogram(
    "stream_processing_seconds",
    "Время обработки одной стриминговой задачи",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

start_http_server(8010)


@broker.subscriber(
    app_settings.Klin_stream_queue,
    consume_args={"prefetch_count": 1},
)
async def stream_start_handler(message: RabbitMessage) -> None:
    """
    Запускает стриминг-анализ камеры.
    """
    try:
        dto: StreamProcessDto = msgspec.json.decode(message.body, type=StreamProcessDto)

        logger.info(
            "Получена задача на запуск стрима",
            extra={"stream_id": str(dto.stream_id)},
        )

        with STREAM_PROCESSING_TIME.time():
            await stream_service.perform_stream(stream_id=dto.stream_id)

        STREAM_PROCESSED.labels(status="success").inc()

        logger.info(
            "Стрим успешно запущен в обработку",
            extra={"stream_id": str(dto.stream_id)},
        )

    except Exception as exc:
        STREAM_PROCESSED.labels(status="failed").inc()
        logger.exception(
            "Ошибка при запуске стрима",
            extra={
                "stream_id": getattr(dto, "stream_id", None),
                "error": str(exc),
            },
        )
        raise  # FastStream сделает NACK + dead letter (если настроен)
