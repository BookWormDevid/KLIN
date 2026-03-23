"""
Endpoint'ы API сервиса Klin.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, BinaryIO, cast
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.litestar import inject
from litestar import Controller, MediaType, Response, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException
from litestar.params import Body
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_404_NOT_FOUND

from app.application.dto import KlinReadDto, KlinUploadDto
from app.application.exceptions import KlinEnqueueError, KlinNotFoundError
from app.application.interfaces import IKlinVideoStorage
from app.application.services import KlinService
from app.config import app_settings


class KlinController(Controller):
    """
    HTTP-endpoint'ы сервиса Klin.
    """

    path = "/Klin"
    tags: Sequence[str] | None = ["Klin"]

    @staticmethod
    def _build_object_key(filename: str | None) -> str:
        prefix = app_settings.s3_key_prefix
        suffix = Path(filename or "").suffix.lower() or ".mp4"
        object_name = f"{uuid.uuid4()}{suffix}"
        return f"{prefix}/{object_name}" if prefix else object_name

    @staticmethod
    async def _cleanup_uploaded_object(
        klin_video_storage: IKlinVideoStorage,
        object_uri: str,
    ) -> None:
        try:
            await klin_video_storage.delete(object_uri)
        except Exception:
            pass

    @post("/upload", status_code=HTTP_201_CREATED, media_type=MediaType.JSON)
    @inject
    async def file_upload(
        self,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
        klin_service: FromDishka[KlinService],
        klin_video_storage: FromDishka[IKlinVideoStorage],
        response_url: Annotated[
            str | None, Body(media_type=RequestEncodingType.MULTI_PART)
        ] = None,
    ) -> KlinReadDto:
        """
        Загружает исходное видео в S3-совместимое хранилище и ставит задачу в очередь.
        """

        max_size = 200 * 1024 * 1024
        object_key = self._build_object_key(data.filename)

        try:
            await data.seek(0)
            object_uri = await klin_video_storage.upload_fileobj(
                fileobj=cast(BinaryIO, data.file),
                object_key=object_key,
                content_type=data.content_type,
                max_size_bytes=max_size,
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 413 if detail == "File too large" else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
        finally:
            await data.close()

        upload_dto = KlinUploadDto(response_url=response_url, video_path=object_uri)

        try:
            klin_model = await klin_service.klin_image(upload_dto)
        except KlinEnqueueError as exc:
            await self._cleanup_uploaded_object(klin_video_storage, object_uri)
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc
        except Exception:
            await self._cleanup_uploaded_object(klin_video_storage, object_uri)
            raise

        return KlinReadDto.from_model(klin_model)

    @get("/{klin_id:uuid}", status_code=HTTP_200_OK)
    @inject
    async def get_inference_status(
        self,
        klin_service: FromDishka[KlinService],
        klin_id: UUID,
    ) -> Response[KlinReadDto]:
        """
        Возвращает полное состояние инференса по идентификатору.
        """

        try:
            inference = await klin_service.get_inference_status(klin_id)
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

    @get("/health/ready")
    @inject
    async def readiness_check(
        self, klin_service: FromDishka[KlinService]
    ) -> Response[dict[str, str]]:
        """
        Проверка readiness для зависимостей сервиса.
        """

        try:
            await klin_service.get_n_imferences(1)
            return Response({"status": "ready"})
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: {str(exc)}",
            ) from exc

    @get("/", status_code=HTTP_200_OK)
    @inject
    async def get_all(
        self, klin_service: FromDishka[KlinService]
    ) -> Response[list[KlinReadDto]]:
        """
        Возвращает последние 100 задач.
        """

        imfer_list = await klin_service.get_n_imferences(100)
        return Response([KlinReadDto.from_model(imference) for imference in imfer_list])
