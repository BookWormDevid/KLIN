import os
import uuid
from collections.abc import Sequence
from typing import Annotated, List
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, MediaType, Response, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException
from litestar.params import Body
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED

from app.application.dto import MAEReadDto, MAEUploadDto
from app.application.services import MAEService


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
        MAX_SIZE = 200 * 2048 * 2048
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        safe_name = f"{uuid.uuid4()}.mp4"
        tmp_path = os.path.join(tmp_dir, safe_name)

        size = 0

        with open(tmp_path, "wb") as f:
            while chunk := await data.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_SIZE:
                    f.close()
                    os.remove(tmp_path)
                    raise HTTPException(status_code=413, detail="File too large")

                f.write(chunk)

        upload_dto = MAEUploadDto(response_url=response_url, video_path=tmp_path)

        mae_model = await mae_service.MAE_image(upload_dto)

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

    @get(path="/health/live", media_type=MediaType.TEXT)
    def health_check(self) -> str:
        return "healthy"

    @get("/health/ready")
    @inject
    async def readiness_check(
        self, mae_service: FromDishka[MAEService], MAE_id: UUID
    ) -> Response[MAEReadDto]:
        try:
            check = await mae_service.get_inference_status(MAE_id)
            return Response(check)
        except ConnectionError as e:
            raise HTTPException(
                status_code=503, detail=f"Database unavailable: {str(e)}"
            ) from e

    @get("/", status_code=HTTP_200_OK)
    @inject
    async def get_all(
        self, mae_service: FromDishka[MAEService]
    ) -> Response[list[MAEReadDto]]:
        imfer_list = await mae_service.get_n_imferences(100)
        dto_ready_list = []
        for imference in imfer_list:
            dto_ready_list.append(MAEReadDto.from_model(imference))

        return Response(dto_ready_list)
