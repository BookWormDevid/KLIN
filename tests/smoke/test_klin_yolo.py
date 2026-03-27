import asyncio
import uuid

import msgspec
from faststream.rabbit import RabbitBroker

from app.application.dto import StreamEventDto
from app.config import app_settings


async def send_event() -> None:
    broker = RabbitBroker(app_settings.rabbit_url)
    await broker.connect()

    event = StreamEventDto(
        id=str(uuid.uuid4()),
        type="YOLO",
        camera_id="cam_1",
        stream_id=uuid.UUID("247b32f3-f4f5-478f-927e-18d67b088575"),
        payload={
            "frame_idx": 1,
            "timestamp": 1774543689.6240273,
            "detections": [
                {
                    "class_id": 0,
                    "label": "person",
                    "bbox": [
                        230.47079467773438,
                        262.52813720703125,
                        323.5611267089844,
                        592.9786987304688,
                    ],
                    "confidence": 0.6510257124900818,
                    "timestamp": 1774543689.6240273,
                }
            ],
        },
    )

    await broker.publish(
        msgspec.json.encode(event),
        queue=app_settings.Klin_stream_event_queue,
    )


if __name__ == "__main__":
    asyncio.run(send_event())
