"""
Взаимодействие с брокером rabbitbroker
"""

from dataclasses import dataclass

import msgspec
from faststream.rabbit import RabbitBroker

from app.application.dto import KlinProcessDto, StreamProcessDto
from app.application.interfaces import IKlinProcessProducer
from app.config import app_settings


@dataclass()
class KlinProcessProducer(IKlinProcessProducer):
    """
    Класс для взаимодействия с брокером
    """

    _rabbit_broker: RabbitBroker

    async def send(self, data: KlinProcessDto) -> None:
        """
        Отправка данных для брокера
        """
        await self._rabbit_broker.publish(
            msgspec.json.encode(data), queue=app_settings.Klin_queue
        )

    async def send_streaming(self, data: StreamProcessDto) -> None:
        await self._rabbit_broker.publish(
            msgspec.json.encode(data), queue=app_settings.Klin_queue
        )
