import uuid
from abc import abstractmethod
from typing import Protocol

from app.application.dto import YoloProcessDto
from app.models import YoloModel


class IYoloInference(Protocol):
    @abstractmethod
    async def analyze(self, model: YoloModel) -> str: ...


class IYoloRepository(Protocol):
    @abstractmethod
    async def get_by_id(self, klin_id: uuid.UUID) -> YoloModel: ...

    @abstractmethod
    async def create(self, model: YoloModel) -> YoloModel: ...

    @abstractmethod
    async def update(self, model: YoloModel) -> None: ...


class IYoloProcessProducer(Protocol):
    @abstractmethod
    async def send(self, data: YoloProcessDto) -> None: ...


# class IKlinCallbackSender(Protocol): ...
