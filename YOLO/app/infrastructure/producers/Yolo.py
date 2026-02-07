from dataclasses import dataclass

import msgspec
from faststream.rabbit import RabbitBroker

from YOLO.app.application.dto import YoloProcessDto
from YOLO.app.application.interfaces import IYoloProcessProducer
from YOLO.app.config import app_settings


@dataclass()
class YoloProcessProducer(IYoloProcessProducer):
    _rabbit_broker: RabbitBroker

    async def send(self, data: YoloProcessDto) -> None:
        await self._rabbit_broker.publish(
            msgspec.json.encode(data), queue=app_settings.yolo_queue
        )
