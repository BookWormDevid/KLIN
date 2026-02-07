import uuid
from abc import abstractmethod
from typing import Protocol

from YOLO.appapplication.dto import YoloProcessDto
from YOLO.appmodels import YoloModel


class IYoloInference(Protocol):
    @abstractmethod
    async def analyze(self, model: YoloModel) -> str: ...


class IYoloRepository(Protocol):
    @abstractmethod
    async def get_by_id(self, yolo_id: uuid.UUID) -> YoloModel: ...

    @abstractmethod
    async def create(self, model: YoloModel) -> YoloModel: ...

    @abstractmethod
    async def update(self, model: YoloModel) -> None: ...


class IYoloProcessProducer(Protocol):
    @abstractmethod
    async def send(self, data: YoloProcessDto) -> None: ...


# class IYoloCallbackSender(Protocol): ...
