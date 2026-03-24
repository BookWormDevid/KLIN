# pylint: disable=too-few-public-methods
"""
Передача информации из инфраструктурного слоя.
Содержит классы для взаимодействия инфраструктурного слоя со слоями выше.
"""

import uuid
from abc import abstractmethod
from typing import BinaryIO, Protocol

from app.application.dto import (
    KlinProcessDto,
    KlinResultDto,
    StreamEventDto,
    StreamProcessDto,
)
from app.models import (
    KlinMaeResult,
    KlinModel,
    KlinStreamState,
    KlinX3DResult,
    KlinYoloResult,
)


class IKlinEventProducer(Protocol):
    @abstractmethod
    async def send_event(self, event: StreamEventDto) -> None: ...


class IKlinStream(Protocol):
    @abstractmethod
    async def streaming_analyze(self, model: KlinStreamState) -> None: ...

    @abstractmethod
    async def stop(self, camera_id: str) -> None: ...

    @abstractmethod
    async def wait_stopped(self, camera_id: str, timeout: float = 5) -> bool: ...


class IKlinInference(Protocol):
    """
    Класс для передачи данных процессора
    """

    @abstractmethod
    async def analyze(self, model: KlinModel) -> KlinResultDto:
        """
        Метод передачи данных процессора и запуска анализа
        """


class IKlinVideoStorage(Protocol):
    """
    Контракт объектного хранилища для загруженных видео.
    """

    @abstractmethod
    async def upload_fileobj(
        self,
        *,
        fileobj: BinaryIO,
        object_key: str,
        content_type: str | None = None,
        max_size_bytes: int | None = None,
    ) -> str:
        """
        Сохраняет в s3 хранилище и сохраняет URI
        """

    @abstractmethod
    async def download_to_path(self, *, source_uri: str, destination_path: str) -> None:
        """
        Загружает из s3 на локальный путь
        """

    @abstractmethod
    async def delete(self, source_uri: str) -> None:
        """
        Удаляет хранящийся объект
        """


class IKlinRepository(Protocol):
    """
    Класс для взаимодействия с базой данных
    """

    @abstractmethod
    async def save_yolo(self, event: KlinYoloResult) -> None: ...

    @abstractmethod
    async def save_mae(self, event: KlinMaeResult) -> None: ...

    @abstractmethod
    async def save_x3d(self, event: KlinX3DResult) -> None: ...

    @abstractmethod
    async def get_by_id(self, klin_id: uuid.UUID) -> KlinModel:
        """
        Метод передачи данных из бд по id
        """

    @abstractmethod
    async def get_by_id_stream(self, stream_id: uuid.UUID) -> KlinStreamState:
        """
        Метод передачи данных из бд по id
        """

    @abstractmethod
    async def claim_for_processing(self, klin_id: uuid.UUID) -> KlinModel | None:
        """
        Атомарно переводит задачу из PENDING в PROCESSING.
        Возвращает модель, если захват выполнен, иначе None.
        """

    async def claim_for_processing_stream(
        self, klin_id: uuid.UUID
    ) -> KlinStreamState | None:
        """
        Атомарно переводит задачу из PENDING в PROCESSING.
        Возвращает модель, если захват выполнен, иначе None.
        """

    @abstractmethod
    async def create(
        self, model: KlinModel | KlinStreamState
    ) -> KlinModel | KlinStreamState:
        """
        Метод для создания запроса в бд
        """

    @abstractmethod
    async def update(self, model: KlinModel | KlinStreamState) -> None:
        """
        Метод для обновления запроса в бд
        """

    @abstractmethod
    async def get_first_n(self, count: int) -> list[KlinModel]:
        """
        Метод для получения запроса к бд - последние n строк по id
        """


class IKlinProcessProducer(Protocol):
    """
    Класс для взаимодействия с брокером
    """

    @abstractmethod
    async def send(self, data: KlinProcessDto) -> None:
        """
        Метод для отправки сообщений в брокер
        """

    @abstractmethod
    async def send_stream(self, data: StreamProcessDto) -> None:
        """
        Метод для отправки сообщений в брокер
        """


class IKlinCallbackSender(Protocol):
    """
    Класс для взаимодействия с отправкой вывода процессора
    """

    @abstractmethod
    async def post_consumer(self, model: KlinModel) -> None:
        """
        Метод для отправки json с выводом процессора
        """
