"""
Endpoint's для стрима
"""

from collections.abc import Sequence
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, get, post
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED

from app.application.dto import StreamReadDto, StreamUploadDto
from app.application.exceptions import KlinNotFoundError
from app.application.mappers import to_stream_read_dto
from app.application.services import StreamService
from app.presentation.litestar.controllers.helpers import LitestarErrors


litestar_errors: LitestarErrors = LitestarErrors()


class KlinStreamController(Controller):
    """
    HTTP-endpoint'ы сервиса стриминга.
    """

    path = "/Klin_Stream"
    tags: Sequence[str] | None = ["Klin_Stream"]

    @post("/upload", status_code=HTTP_201_CREATED)
    @inject
    async def start_stream(
        self,
        klin_stream_service: FromDishka[StreamService],
        data: StreamUploadDto,
    ) -> StreamReadDto:
        """
        Запуск стрима
        """
        stream = await klin_stream_service.start_stream(data)
        return to_stream_read_dto(stream)

    @post("/{stream_id:uuid}/stop", status_code=HTTP_200_OK)
    @inject
    async def stop_stream(
        self,
        stream_id: UUID,
        stream_service: FromDishka[StreamService],
    ) -> dict:
        """
        Остановка стрима
        """
        await stream_service.stop_stream(stream_id)

        return {
            "stream_id": str(stream_id),
            "status": "stop_requested",
        }

    @get("/{stream_id:uuid}", status_code=HTTP_200_OK)
    @inject
    async def stream_check(
        self, stream_service: FromDishka[StreamService], stream_id: UUID
    ) -> Response[StreamReadDto]:
        """
        Получения информации о стриме по его id
        """
        try:
            inference = await stream_service.get_stream_status(stream_id)

        except KlinNotFoundError as e:
            litestar_errors.raise_404(e)
        return Response(inference)

    class HealthController(Controller):
        """
        Класс для проверки жизни сервиса
        """

        path = "/health"

        @get("/live", opt={"exclude_from_auth": True})
        async def live(self) -> str:
            return "healthy"
