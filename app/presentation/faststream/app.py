"""
Запуск faststream.
Rabbit очередь и метрики к обработанным задачам.
"""

import logging

import msgspec
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage
from prometheus_client import Counter, Histogram, start_http_server
from sqlalchemy.ext.asyncio import AsyncEngine

from app.application.dto import KlinProcessDto
from app.application.services import KlinService
from app.config import app_settings
from app.infrastructure.database.health import ping_database
from app.ioc import get_worker_providers


logger = logging.getLogger(__name__)
container = make_container(*get_worker_providers())
broker = container.get(RabbitBroker)
db_engine = container.get(AsyncEngine)
Klin_service = container.get(KlinService)


async def verify_worker_database() -> None:
    """Fail startup early when Postgres is unavailable."""
    logger.info("Checking worker database connectivity before consuming messages")
    await ping_database(db_engine)
    logger.info("Worker database connectivity check passed")


app = FastStream(broker, on_startup=(verify_worker_database,))

KLIN_PROCESSED = Counter(
    "klin_processed_total", "Общее количество обработанных задач Klin"
)
KLIN_PROCESSING_TIME = Histogram(
    "klin_processing_seconds", "Время обработки одной задачи Klin"
)

start_http_server(8009)


@broker.subscriber(
    app_settings.Klin_queue,
    consume_args={"prefetch_count": 1},
)
async def base_handler(message: RabbitMessage) -> None:
    """
    Получает json из очереди Rabbit и декодирует его.
    Измеряет время обработки задачи.
    Увеличивает счётчик всех обработанных задач для Prometheus
    """

    data = msgspec.json.decode(message.body, type=KlinProcessDto)

    with KLIN_PROCESSING_TIME.time():
        await Klin_service.perform_klin(klin_id=data.klin_id)

    KLIN_PROCESSED.inc()
