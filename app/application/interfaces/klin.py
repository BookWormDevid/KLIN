# pylint: disable=too-few-public-methods
"""
Передача информации из инфраструктурного слоя.
Содержит классы для взаимодействия инфраструктурного слоя со слоями выше.
"""

import uuid
from abc import abstractmethod
from typing import Protocol

from app.application.dto import KlinProcessDto, KlinResultDto, StreamResultDto
from app.models import KlinModel, KlinStreamingModel


class IKlinStream(Protocol):
    @abstractmethod
    async def streaming_analyze(self, model: KlinStreamingModel) -> StreamResultDto:
        """ """


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
    async def get_by_id(self, klin_id: uuid.UUID) -> KlinModel:
        """
        Метод передачи данных из бд по id
        """

    @abstractmethod
    async def claim_for_processing(self, klin_id: uuid.UUID) -> KlinModel | None:
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
    async def update(self, model: KlinModel) -> None:
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


class IKlinCallbackSender(Protocol):
    """
    Класс для взаимодействия с отправкой вывода процессора
    """

    @abstractmethod
    async def post_consumer(self, model: KlinModel) -> None:
        """
        Метод для отправки json с выводом процессора
        """
