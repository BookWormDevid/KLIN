from collections.abc import Sequence
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, get
from litestar.status_codes import HTTP_200_OK

from YOLO.app.application.dto import YoloReadDto
from YOLO.app.application.services import YoloService


class YoloController(Controller):
    class TranscriptionController(Controller):
        path = "/yolo"
        tags: Sequence[str] | None = ["yolo"]

        # пиши метод пост

        @get("/{yolo_id:uuid}", status_code=HTTP_200_OK)
        @inject
        async def get_inference_status(
            self,
            yolo_service: FromDishka[YoloService],
            yolo_id: UUID,
        ) -> Response[YoloReadDto]:
            inference = await yolo_service.get_inference_status(yolo_id)
            return Response(inference)
