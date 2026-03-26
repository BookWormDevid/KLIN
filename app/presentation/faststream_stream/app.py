"""
FastStream Worker — обработка стриминговых задач (реал-тайм CV).
Очередь: Klin_stream_queue
"""

import logging

import msgspec
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage
from prometheus_client import Counter, Histogram, start_http_server
from sqlalchemy.ext.asyncio import AsyncEngine

from app.application.dto import StreamEventDto, StreamProcessDto
from app.application.interfaces import IKlinStreamEventConsumer
from app.application.services import StreamService
from app.config import app_settings
from app.infrastructure.database.health import ping_database
from app.ioc import get_worker_providers


logger = logging.getLogger(__name__)

# ====================== Dependency Injection ======================
container = make_container(*get_worker_providers())
broker: RabbitBroker = container.get(RabbitBroker)
db_engine: AsyncEngine = container.get(AsyncEngine)

stream_service: StreamService = container.get(StreamService)

stream_event_consumer: IKlinStreamEventConsumer = container.get(
    IKlinStreamEventConsumer
)


async def verify_worker_database() -> None:
    """Fail startup early when Postgres is unavailable."""
    logger.info(
        "Checking stream worker database connectivity before consuming messages"
    )
    await ping_database(db_engine)
    logger.info("Stream worker database connectivity check passed")


app = FastStream(
    broker,
    on_startup=(verify_worker_database,),
)  # ← вот эта переменная должна быть!

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


@broker.subscriber(app_settings.Klin_stream_event_queue)
async def event_handler(message: RabbitMessage):
    event = msgspec.json.decode(message.body, type=StreamEventDto)
    await stream_event_consumer.handle(event=event)


@broker.subscriber(
    app_settings.Klin_process_queue,
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
        raise
