from collections.abc import Sequence
from uuid import UUID
from typing import Annotated
from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, Response, get, post, MediaType
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from app.application.dto import MAEReadDto, MAEUploadDto
from app.application.services import MAEService
import os

class MAEController(Controller):
    path = "/MAE"
    tags: Sequence[str] | None = ["MAE"]

    @post("/upload", status_code=HTTP_201_CREATED, media_type=MediaType.JSON)
    @inject
    async def file_upload(
            self,
            data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
            response_url: Annotated[str, Body(media_type=RequestEncodingType.MULTI_PART)],
            mae_service: FromDishka[MAEService],
    ) -> MAEReadDto:
        # Сохраняем файл
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, data.filename)
        content = await data.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        upload_dto = MAEUploadDto(response_url=response_url, video_path=tmp_path)

        # Передаем в сервис
        mae_model = await mae_service.MAE_image(upload_dto)

        # Возвращаем статус
        return MAEReadDto.from_model(mae_model)

    @get("/{MAE_id:uuid}", status_code=HTTP_200_OK)
    @inject
    async def get_inference_status(
        self,
        mae_service: FromDishka[MAEService],
        MAE_id: UUID,
    ) -> Response[MAEReadDto]:
        inference = await mae_service.get_inference_status(MAE_id)
        return Response(inference)