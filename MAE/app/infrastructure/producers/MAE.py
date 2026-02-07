from dataclasses import dataclass

import msgspec
from faststream.rabbit import RabbitBroker

from MAE.app.application.dto import MAEProcessDto
from MAE.app.application.interfaces import IMAEProcessProducer
from MAE.app.config import app_settings


@dataclass()
class MAEProcessProducer(IMAEProcessProducer):
    _rabbit_broker: RabbitBroker

    async def send(self, data: MAEProcessDto) -> None:
        await self._rabbit_broker.publish(
            msgspec.json.encode(data), queue=app_settings.MAE_queue
        )
