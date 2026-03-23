"""
Интеграция с S3-совместимым объектным хранилищем для загруженных видео.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, BinaryIO, cast
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import app_settings


class S3ObjectStorage:
    """
    Блокирующий boto3-клиент, обернутый в асинхронные вспомогательные методы.
    """

    def __init__(self) -> None:
        self._client_error = ClientError
        self._bucket_name = app_settings.s3_bucket_name
        self._region_name = app_settings.s3_region
        self._bucket_ready = False
        self._bucket_lock = asyncio.Lock()
        s3_config = cast(
            Any,
            {"addressing_style": app_settings.s3_addressing_style},
        )

        self._client = boto3.client(
            "s3",
            endpoint_url=app_settings.s3_endpoint_url,
            aws_access_key_id=app_settings.s3_access_key_id,
            aws_secret_access_key=app_settings.s3_secret_access_key,
            region_name=self._region_name,
            config=Config(
                signature_version="s3v4",
                s3=s3_config,
            ),
        )

    async def upload_fileobj(
        self,
        *,
        fileobj: BinaryIO,
        object_key: str,
        content_type: str | None = None,
        max_size_bytes: int | None = None,
    ) -> str:
        """
        Загружает файловый объект в S3-совместимое хранилище.
        """

        await self._ensure_bucket_exists()
        file_size = await self._run_sync(self._get_file_size, fileobj)

        if file_size <= 0:
            raise ValueError("Uploaded file is empty")

        if max_size_bytes is not None and file_size > max_size_bytes:
            raise ValueError("File too large")

        extra_args: dict[str, Any] | None = None
        if content_type:
            extra_args = {"ContentType": content_type}

        def _upload() -> None:
            fileobj.seek(0)
            self._client.upload_fileobj(
                Fileobj=fileobj,
                Bucket=self._bucket_name,
                Key=object_key,
                ExtraArgs=extra_args,
            )

        await self._run_sync(_upload)
        return self.build_uri(bucket_name=self._bucket_name, object_key=object_key)

    async def download_to_path(self, *, source_uri: str, destination_path: str) -> None:
        """
        Скачивает объект из хранилища в локальный путь.
        """

        bucket_name, object_key = self.parse_uri(source_uri)
        await self._run_sync(
            self._client.download_file,
            bucket_name,
            object_key,
            destination_path,
        )

    async def delete(self, source_uri: str) -> None:
        """
        Удаляет объект из хранилища.
        """

        bucket_name, object_key = self.parse_uri(source_uri)
        await self._run_sync(
            self._client.delete_object,
            Bucket=bucket_name,
            Key=object_key,
        )

    async def _ensure_bucket_exists(self) -> None:
        if self._bucket_ready:
            return

        async with self._bucket_lock:
            if self._bucket_ready:
                return

            try:
                await self._run_sync(self._client.head_bucket, Bucket=self._bucket_name)
            except self._client_error as exc:
                error_code = self._extract_error_code(exc)
                if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                    raise

                create_kwargs: dict[str, Any] = {"Bucket": self._bucket_name}
                if self._region_name and self._region_name != "us-east-1":
                    create_kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": self._region_name
                    }

                try:
                    await self._run_sync(self._client.create_bucket, **create_kwargs)
                except self._client_error as create_exc:
                    create_error_code = self._extract_error_code(create_exc)
                    if create_error_code not in {
                        "BucketAlreadyOwnedByYou",
                        "BucketAlreadyExists",
                    }:
                        raise

            self._bucket_ready = True

    @staticmethod
    def build_uri(*, bucket_name: str, object_key: str) -> str:
        """
        Literally just builds a url with an f string
        """
        return f"s3://{bucket_name}/{object_key}"

    @staticmethod
    def parse_uri(source_uri: str) -> tuple[str, str]:
        """
        Literally just parses a url with urlparse and an error handler
        """
        parsed = urlparse(source_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(f"Unsupported S3 URI: {source_uri}")
        return parsed.netloc, parsed.path.lstrip("/")

    @staticmethod
    def _get_file_size(fileobj: BinaryIO) -> int:
        current_position = fileobj.tell()
        fileobj.seek(0, 2)
        file_size = fileobj.tell()
        fileobj.seek(current_position)
        return file_size

    def _extract_error_code(self, exc: Exception) -> str:
        response = getattr(exc, "response", None) or {}
        error = response.get("Error", {})
        return str(error.get("Code", ""))

    async def _run_sync(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
