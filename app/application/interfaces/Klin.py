import uuid
from abc import abstractmethod
from typing import Protocol

from app.application.dto import KlinProcessDto
from app.models import KlinModel


class IKlinInference(Protocol):
    @abstractmethod
    async def analyze(self, model: KlinModel) -> str: ...


class IKlinRepository(Protocol):
    @abstractmethod
    async def get_by_id(self, klin_id: uuid.UUID) -> KlinModel: ...

    @abstractmethod
    async def create(self, model: KlinModel) -> KlinModel: ...

    @abstractmethod
    async def update(self, model: KlinModel) -> None: ...


class IKlinProcessProducer(Protocol):
    @abstractmethod
    async def send(self, data: KlinProcessDto) -> None: ...
