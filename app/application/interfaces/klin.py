# pylint: disable=too-few-public-methods
"""
Передача информации из инфраструктурного слоя.
Содержит классы для взаимодействия инфраструктурного слоя со слоями выше.
"""

import uuid
from abc import abstractmethod
from typing import Protocol

from app.application.dto import (
    KlinProcessDto,
    KlinResultDto,
    StreamEventDto,
    StreamProcessDto,
)
from app.models import KlinModel, KlinStreamingModel


class IKlinEventProducer(Protocol):
    @abstractmethod
    async def send_event(self, event: StreamEventDto) -> None: ...


class IKlinStream(Protocol):
    @abstractmethod
    async def streaming_analyze(self, model: KlinStreamingModel) -> None: ...


class IKlinInference(Protocol):
    """
    Класс для передачи данных процессора
    """

    @abstractmethod
    async def analyze(self, model: KlinModel) -> KlinResultDto:
        """
        Метод передачи данных процессора и запуска анализа
        """


class IKlinRepository(Protocol):
    """
    Класс для взаимодействия с базой данных
    """

    @abstractmethod
    async def save_yolo(self, event: StreamEventDto) -> None: ...

    @abstractmethod
    async def save_mae(self, event: StreamEventDto) -> None: ...

    @abstractmethod
    async def save_x3d(self, event: StreamEventDto) -> None: ...

    @abstractmethod
    async def get_by_id(self, klin_id: uuid.UUID) -> KlinModel:
        """
        Метод передачи данных из бд по id
        """

    @abstractmethod
    async def get_by_id_stream(self, stream_id: uuid.UUID) -> KlinStreamingModel:
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
    ) -> KlinStreamingModel | None:
        """
        Атомарно переводит задачу из PENDING в PROCESSING.
        Возвращает модель, если захват выполнен, иначе None.
        """

    @abstractmethod
    async def create(self, model: KlinModel) -> KlinModel:
        """
        Метод для создания запроса в бд
        """

    @abstractmethod
    async def create_stream(self, model: KlinStreamingModel) -> KlinStreamingModel:
        """
        Метод для создания запроса в бд
        """

    @abstractmethod
    async def update(self, model: KlinModel) -> None:
        """
        Метод для обновления запроса в бд
        """

    @abstractmethod
    async def update_stream(self, model: KlinStreamingModel) -> None:
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
