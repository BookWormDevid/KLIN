import uuid
from abc import abstractmethod
from typing import Protocol

from app.application.dto import MAEProcessDto, MAEResultDto
from app.models import MAEModel


class IMAEInference(Protocol):
    @abstractmethod
    async def analyze(self, model: MAEModel) -> MAEResultDto: ...


class IMAERepository(Protocol):
    @abstractmethod
    async def get_by_id(self, MAE_id: uuid.UUID) -> MAEModel: ...

    @abstractmethod
    async def create(self, model: MAEModel) -> MAEModel: ...

    @abstractmethod
    async def update(self, model: MAEModel) -> None: ...


class IMAEProcessProducer(Protocol):
    @abstractmethod
    async def send(self, data: MAEProcessDto) -> None: ...


class IMAECallbackSender(Protocol):
    @abstractmethod
    async def post_consumer(self, model: MAEModel) -> None: ...
