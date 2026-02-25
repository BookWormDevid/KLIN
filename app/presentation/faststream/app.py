import msgspec
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage

from app.application.dto import KlinProcessDto
from app.application.services import KlinService
from app.config import app_settings
from app.ioc import ApplicationProvider, InfrastructureProvider, VideoProvider

container = make_container(
    InfrastructureProvider(), ApplicationProvider(), VideoProvider()
)

broker = container.get(RabbitBroker)

app = FastStream(broker)

Klin_service = container.get(KlinService)


@broker.subscriber(
    app_settings.Klin_queue,
    consume_args={"prefetch_count": 1},
)
async def base_handler(message: RabbitMessage) -> None:
    data = msgspec.json.decode(message.body, type=KlinProcessDto)

    await Klin_service.perform_klin(klin_id=data.klin_id)
