from collections.abc import Sequence
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, get
from litestar.status_codes import HTTP_200_OK

from MAE.app.application.dto import MAEReadDto
from MAE.app.application.services import MAEService


class MAEController(Controller):
    class TranscriptionController(Controller):
        path = "/MAE"
        tags: Sequence[str] | None = ["MAE"]

        # пиши метод пост

        @get("/{MAE_id:uuid}", status_code=HTTP_200_OK)
        @inject
        async def get_inference_status(
            self,
            MAE_service: FromDishka[MAEService],
            MAE_id: UUID,
        ) -> Response[MAEReadDto]:
            inference = await MAE_service.get_inference_status(MAE_id)
            return Response(inference)
