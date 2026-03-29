from collections.abc import Sequence
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, MediaType, Response, get, post
from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_404_NOT_FOUND

from app.application.dto import StreamReadDto, StreamUploadDto
from app.application.exceptions import KlinNotFoundError
from app.application.mappers import to_stream_read_dto
from app.application.services import StreamService


class KlinStreamController(Controller):
    path = "/Klin_Stream"
    tags: Sequence[str] | None = ["Klin_Stream"]

    @post("/upload", status_code=HTTP_201_CREATED)
    @inject
    async def start_stream(
        self,
        klin_stream_service: FromDishka[StreamService],
        data: StreamUploadDto,
    ) -> StreamReadDto:
        stream = await klin_stream_service.start_stream(data)
        return to_stream_read_dto(stream)

    @post("/{stream_id:uuid}/stop", status_code=HTTP_200_OK)
    @inject
    async def stop_stream(
        self,
        stream_id: UUID,
        stream_service: FromDishka[StreamService],
    ) -> dict:
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
        try:
            inference = await stream_service.get_stream_status(stream_id)

        except KlinNotFoundError as exc:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return Response(inference)

    @get(path="/health/live", media_type=MediaType.TEXT)
    async def health_check(self) -> str:
        """
        Проверка liveness для API-процесса.
        """

        return "healthy"
