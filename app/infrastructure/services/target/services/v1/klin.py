from collections.abc import Sequence

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, post
from litestar.status_codes import HTTP_201_CREATED

from app.application.dto import KlinReadDto, KlinUploadDto
from app.application.services import KlinService


class MAEController(Controller):
    path = "/MAE"
    tags: Sequence[str] | None = ["MAE"]

    @post("/upload", status_code=HTTP_201_CREATED)
    @inject
    async def MAE_image(
        self,
        Klin_service: FromDishka[KlinService],
        data: KlinUploadDto,
    ) -> Response[KlinReadDto]:
        MAE = await Klin_service.MAE_image(data)

        return Response(KlinReadDto.from_model(MAE))
