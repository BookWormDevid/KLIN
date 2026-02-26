"""
HTTP контроллер для загрузки видео и запуска обработки
"""

from collections.abc import Sequence

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, post
from litestar.status_codes import HTTP_201_CREATED

from app.application.dto import KlinReadDto, KlinUploadDto
from app.application.services import KlinService


class MAEController(Controller):
    """
    Обрабатывает POST-запрос загрузки и запускает бизнес-логику через сервис.
    """

    path = "/MAE"
    tags: Sequence[str] | None = ["MAE"]

    @post("/upload", status_code=HTTP_201_CREATED)
    @inject
    async def mae_image(
        self,
        klin_service: FromDishka[KlinService],
        data: KlinUploadDto,
    ) -> Response[KlinReadDto]:
        """
        Принимает данные загрузки, передаёт их в KlinService
        и возвращает результат обработки.
        """
        mae = await klin_service.klin_image(data)

        return Response(KlinReadDto.from_model(mae))
