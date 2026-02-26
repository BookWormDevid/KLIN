"""
endpoint'ы приложения
"""

import os
import uuid
from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, MediaType, Response, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException
from litestar.params import Body
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED

from app.application.dto import KlinReadDto, KlinUploadDto
from app.application.services import KlinService


class KlinController(Controller):
    """
    Класс с endpoint'ами
    """

    path = "/Klin"
    tags: Sequence[str] | None = ["Klin"]

    @post("/upload", status_code=HTTP_201_CREATED, media_type=MediaType.JSON)
    @inject
    async def file_upload(
        self,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
        response_url: Annotated[str, Body(media_type=RequestEncodingType.MULTI_PART)],
        klin_service: FromDishka[KlinService],
    ) -> KlinReadDto:
        """
        Скачивает видео, преобразует его в формат mp4 и сохраняет в папку tmp.
        """
        max_size = 200 * 2048 * 2048
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        safe_name = f"{uuid.uuid4()}.mp4"
        tmp_path = os.path.join(tmp_dir, safe_name)

        size = 0

        with open(tmp_path, "wb") as f:
            while chunk := await data.read(1024 * 1024):
                size += len(chunk)

                if size > max_size:
                    f.close()
                    os.remove(tmp_path)
                    raise HTTPException(status_code=413, detail="File too large")

                f.write(chunk)

        upload_dto = KlinUploadDto(response_url=response_url, video_path=tmp_path)

        klin_model = await klin_service.klin_image(upload_dto)

        return KlinReadDto.from_model(klin_model)

    @get("/{klin_id:uuid}", status_code=HTTP_200_OK)
    @inject
    async def get_inference_status(
        self,
        klin_service: FromDishka[KlinService],
        klin_id: UUID,
    ) -> Response[KlinReadDto]:
        """
        Получает всю информацию об конкретном id
        """
        inference = await klin_service.get_inference_status(klin_id)
        return Response(inference)

    @get(path="/health/live", media_type=MediaType.TEXT)
    async def health_check(self) -> str:
        """
        Проверяет функционирует ли приложение litestar
        """
        return "healthy"

    @get("/health/ready")
    @inject
    async def readiness_check(
        self, klin_service: FromDishka[KlinService], klin_id: UUID
    ) -> Response[KlinReadDto]:
        """
        Проверяет функционирует ли база данных
        """
        try:
            check = await klin_service.get_inference_status(klin_id)
            return Response(check)
        except ConnectionError as e:
            raise HTTPException(
                status_code=503, detail=f"Database unavailable: {str(e)}"
            ) from e

    @get("/", status_code=HTTP_200_OK)
    @inject
    async def get_all(
        self, klin_service: FromDishka[KlinService]
    ) -> Response[list[KlinReadDto]]:
        """
        Получает вывод последних 100 записей в базе данных
        """
        imfer_list = await klin_service.get_n_imferences(100)
        dto_ready_list = []
        for imference in imfer_list:
            dto_ready_list.append(KlinReadDto.from_model(imference))

        return Response(dto_ready_list)
