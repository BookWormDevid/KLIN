from dataclasses import dataclass

import msgspec
from faststream.rabbit import RabbitBroker

from app.application.dto import KlinProcessDto
from app.application.interfaces import IKlinProcessProducer
from app.config import app_settings


@dataclass()
class KlinProcessProducer(IKlinProcessProducer):
    _rabbit_broker: RabbitBroker

    async def send(self, data: KlinProcessDto) -> None:
        await self._rabbit_broker.publish(
            msgspec.json.encode(data), queue=app_settings.Klin_queue
        )
