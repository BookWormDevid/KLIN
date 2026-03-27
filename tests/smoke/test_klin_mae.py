import asyncio
import uuid

import msgspec
from faststream.rabbit import RabbitBroker

from app.application.dto import StreamEventDto
from app.config import app_settings


async def send_mae_event() -> None:
    broker = RabbitBroker(app_settings.rabbit_url)
    await broker.connect()

    event_id = f"mae-{uuid.uuid4()}"
    event = StreamEventDto(
        id=str(uuid.uuid4()),
        type="MAE",
        camera_id="cam_1",
        stream_id=uuid.UUID("247b32f3-f4f5-478f-927e-18d67b088575"),
        payload={
            "event_id": event_id,
            "label": "Burglary",
            "confidence": 0.95,
            "start_ts": 0.64,
            "end_ts": 0.43,
            "probs": [
                {"class_name": "Assault", "probability": 0.9132182002067566},
                {"class_name": "Burglary", "probability": 0.049689892679452896},
                {"class_name": "Fighting", "probability": 0.02134312503039837},
            ],
        },
    )

    await broker.publish(
        msgspec.json.encode(event),
        queue=app_settings.Klin_stream_event_queue,
    )


if __name__ == "__main__":
    asyncio.run(send_mae_event())
