import asyncio
import time
import uuid

import msgspec
from faststream.rabbit import RabbitBroker

from app.application.dto import StreamEventDto
from app.config import app_settings


async def send_x3d_event():
    broker = RabbitBroker(app_settings.rabbit_url)
    await broker.connect()

    event = StreamEventDto(
        id=str(uuid.uuid4()),
        type="X3D_VIOLENCE",
        camera_id="cam_1",
        stream_id="247b32f3-f4f5-478f-927e-18d67b088575",
        payload={
            "label": "violence",
            "prob": 0.91,
            "timestamp": time.time(),
        },
    )

    await broker.publish(
        msgspec.json.encode(event),
        queue=app_settings.Klin_stream_event_queue,
    )
    print("✅ X3D Event sent (с prob и float timestamp)")


asyncio.run(send_x3d_event())
