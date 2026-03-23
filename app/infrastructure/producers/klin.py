"""
Взаимодействие с брокером rabbitbroker
"""

from dataclasses import dataclass

import msgspec
from faststream.rabbit import RabbitBroker

from app.application.dto import KlinProcessDto, StreamEventDto, StreamProcessDto
from app.application.interfaces import (
    IKlinEventProducer,
    IKlinProcessProducer,
    IKlinRepository,
)
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

    async def send_stream(self, data: StreamProcessDto) -> None:
        await self._rabbit_broker.publish(
            msgspec.json.encode(data), queue=app_settings.Klin_queue
        )

    async def send_streaming(self, data: StreamProcessDto) -> None:
        await self.send_stream(data)


@dataclass()
class KlinEventProducer(IKlinEventProducer):
    """Persists stream events through the repository layer."""

    _repository: IKlinRepository

    async def send_event(self, event: StreamEventDto) -> None:
        if event.type == "YOLO":
            await self._repository.save_yolo(event)
            return

        if event.type == "MAE":
            await self._repository.save_mae(event)
            return

        if event.type == "X3D_VIOLENCE":
            await self._repository.save_x3d(event)
            return

        raise ValueError(f"Unsupported stream event type: {event.type}")
