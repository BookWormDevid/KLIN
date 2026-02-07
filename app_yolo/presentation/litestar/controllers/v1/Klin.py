from collections.abc import Sequence
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, get
from litestar.status_codes import HTTP_200_OK

from app.application.dto import KlinReadDto
from app.application.services import KlinService


class KlinController(Controller):
    class TranscriptionController(Controller):
        path = "/klin"
        tags: Sequence[str] | None = ["klin"]

        # пиши метод пост

        @get("/{klin_id:uuid}", status_code=HTTP_200_OK)
        @inject
        async def get_inference_status(
            self,
            klin_service: FromDishka[KlinService],
            klin_id: UUID,
        ) -> Response[KlinReadDto]:
            inference = await klin_service.get_inference_status(klin_id)
            return Response(inference)
