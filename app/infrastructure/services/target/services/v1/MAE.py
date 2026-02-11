from collections.abc import Sequence

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, post
from litestar.status_codes import HTTP_201_CREATED

from app.application.dto import MAEReadDto, MAEUploadDto
from app.application.services import MAEService


class MAEController(Controller):
    path = "/MAE"
    tags: Sequence[str] | None = ["MAE"]

    @post("/upload", status_code=HTTP_201_CREATED)
    @inject
    async def MAE_image(
        self,
        MAE_service: FromDishka[MAEService],
        data: MAEUploadDto,
    ) -> Response[MAEReadDto]:
        MAE = await MAE_service.MAE_image(data)

        return Response(MAEReadDto.from_model(MAE))
