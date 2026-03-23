"""
Запуск faststream.
Rabbit очередь и метрики к обработанным задачам.
"""

import msgspec
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage
from prometheus_client import Counter, Histogram, start_http_server

from app.application.dto import KlinProcessDto
from app.application.services import KlinService
from app.config import app_settings
from app.ioc import get_worker_providers


container = make_container(*get_worker_providers())
broker = container.get(RabbitBroker)
app = FastStream(broker)
Klin_service = container.get(KlinService)

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
