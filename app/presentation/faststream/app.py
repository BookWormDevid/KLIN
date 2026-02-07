import msgspec
from dishka import make_container
from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitMessage

from app.application.dto import MAEProcessDto
from app.application.services import MAEService
from app.config import app_settings
from app.ioc import ApplicationProvider, InfrastructureProvider, VideoProvider

container = make_container(
    InfrastructureProvider(), ApplicationProvider(), VideoProvider()
)

broker = container.get(RabbitBroker)

app = FastStream(broker)

MAE_service = container.get(MAEService)


@broker.subscriber(app_settings.MAE_queue)
async def base_handler(message: RabbitMessage) -> None:
    data = msgspec.json.decode(message.body, type=MAEProcessDto)

    await MAE_service.perform_MAE(MAE_id=data.MAE_id)
