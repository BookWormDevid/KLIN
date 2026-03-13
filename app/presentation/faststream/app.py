"""FastStream приложение для обработки задач Klin через RabbitMQ."""

import msgspec
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage
from prometheus_client import Counter, Histogram, start_http_server

from app.application.dto import KlinProcessDto
from app.application.services import KlinService
from app.config import app_settings
from app.ioc import ApplicationProvider, InfrastructureProvider, VideoProvider


# Контейнер зависимостей
container = make_container(
    InfrastructureProvider(), ApplicationProvider(), VideoProvider()
)
broker = container.get(RabbitBroker)
app = FastStream(broker)
klin_service = container.get(KlinService)

# --- Prometheus метрики ---
KLIN_PROCESSED = Counter(
    "klin_processed_total", "Общее количество обработанных задач Klin"
)
KLIN_PROCESSING_TIME = Histogram(
    "klin_processing_seconds", "Время обработки одной задачи Klin"
)

# Запуск HTTP-сервера для Prometheus
start_http_server(8009)


@broker.subscriber(
    app_settings.Klin_queue,
    consume_args={"prefetch_count": 1},
)
async def base_handler(message: RabbitMessage) -> None:
    """Обрабатывает входящее сообщение и запускает обработку Klin-задачи."""

    data = msgspec.json.decode(message.body, type=KlinProcessDto)

    # Измеряем время обработки задач
    with KLIN_PROCESSING_TIME.time():
        await klin_service.perform_klin(klin_id=data.klin_id)

    # Увеличиваем счетчик обработанных задач
    KLIN_PROCESSED.inc()
