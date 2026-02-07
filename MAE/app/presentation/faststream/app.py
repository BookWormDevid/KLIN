import msgspec
from app.application.dto import KlinProcessDto
from app.application.services import KlinService
from app.config import app_settings
from app.ioc import ApplicationProvider, ImageProvider, InfrastructureProvider
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage

container = make_container(
    InfrastructureProvider(), ApplicationProvider(), ImageProvider()
)

broker = container.get(RabbitBroker)

app = FastStream(broker)

klin_service = container.get(KlinService)


@broker.subscriber(app_settings.Klin_queue)
async def base_handler(message: RabbitMessage) -> None:
    data = msgspec.json.decode(message.body, type=KlinProcessDto)

    await klin_service.perform_klin(klin_id=data.klin_id)
