import logging

from app.application.consumers.stream_event_service import StreamEventService
from app.application.dto import StreamEventDto


logger = logging.getLogger(__name__)


class StreamEventConsumer:
    def __init__(self, service: StreamEventService) -> None:
        self.service = service

    async def handle(self, event: StreamEventDto) -> None:
        await self.service.process(event)
        return
